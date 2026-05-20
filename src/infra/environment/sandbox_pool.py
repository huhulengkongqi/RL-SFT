"""Docker container pool for sandboxed code execution with async concurrency support."""

import asyncio
import logging
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import docker
from docker.models.containers import Container

from infra.sandbox.execution_manager import SandboxExecutor
from infra.sandbox.models import (
    ExecutionResult,
    SandboxConfig,
    SandboxExecutionRequest,
    SandboxExecutionResponse,
    ExecutionStatus,
)

from .models import PoolMetrics, PooledContainerStats

logger = logging.getLogger(__name__)


@dataclass
class PooledContainer:
    """Represents a container in the pool with usage metadata."""

    container: Container
    container_id: str
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    is_busy: bool = False
    created_at: datetime = field(default_factory=datetime.now)


class SandboxPool:
    """
    Managed pool of Docker containers for concurrent sandbox execution.

    Features:
    - Async semaphore for concurrency control
    - Container reuse to avoid cold start overhead
    - Idle container cleanup for resource management
    - Configurable pool size and reuse limits
    """

    def __init__(
        self,
        pool_size: int = 5,
        max_container_reuse: int = 50,
        idle_timeout: int = 600,  # 10 minutes in seconds
        config: Optional[SandboxConfig] = None,
        pre_warm_count: int = 2,
    ):
        self.pool_size = pool_size
        self.max_container_reuse = max_container_reuse
        self.idle_timeout = idle_timeout
        self.pre_warm_count = min(pre_warm_count, pool_size)

        # Default config: 30s timeout, 512MB memory, no network access
        self.config = config or SandboxConfig(
            timeout=30,
            memory_limit=512,
            disk_limit=1024,
            network_access=False,
            image="lab-vllm-build:latest",
        )

        self._pool: List[PooledContainer] = []
        self._semaphore = asyncio.Semaphore(pool_size)
        self._docker_client = docker.from_env()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._total_executions = 0
        self._initialized = False
        self._lock = asyncio.Lock()
        self._executor = SandboxExecutor(self.config)

    async def initialize(self) -> None:
        """Initialize the pool and pre-warm containers."""
        if self._initialized:
            return

        logger.info(
            f"Initializing SandboxPool with {self.pre_warm_count} pre-warmed containers"
        )

        # Pre-warm containers
        warmup_tasks = [self._create_new_container() for _ in range(self.pre_warm_count)]
        await asyncio.gather(*warmup_tasks)

        # Start idle cleanup background task
        self._cleanup_task = asyncio.create_task(self._idle_cleanup_loop())
        self._initialized = True

        logger.info(f"SandboxPool initialized with {len(self._pool)} containers")

    async def shutdown(self) -> None:
        """Shutdown the pool and cleanup all containers."""
        logger.info("Shutting down SandboxPool")

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cleanup all containers
        async with self._lock:
            for pooled in self._pool:
                try:
                    await self._retire_container_no_lock(pooled)
                except Exception as e:
                    logger.warning(
                        f"Error retiring container {pooled.container_id}: {e}"
                    )
            self._pool.clear()

        self._initialized = False
        logger.info("SandboxPool shutdown complete")

    async def execute(
        self, request: SandboxExecutionRequest
    ) -> SandboxExecutionResponse:
        """
        Execute code in an available container from the pool.

        Uses semaphore to limit concurrent execution to pool_size.
        """
        if not self._initialized:
            await self.initialize()

        async with self._semaphore:
            pooled = await self._get_available_container()
            if not pooled:
                pooled = await self._create_new_container()

            try:
                pooled.is_busy = True
                pooled.last_used = datetime.now()
                pooled.use_count += 1
                self._total_executions += 1

                logger.debug(
                    f"Executing in container {pooled.container_id[:12]} "
                    f"(use count: {pooled.use_count})"
                )

                return await self._execute_in_container(pooled.container, request)
            finally:
                pooled.is_busy = False

                # Retire if exceeded reuse limit
                if pooled.use_count >= self.max_container_reuse:
                    async with self._lock:
                        await self._retire_container_no_lock(pooled)

    async def _execute_in_container(
        self, container: Container, request: SandboxExecutionRequest
    ) -> SandboxExecutionResponse:
        """Execute code directly in a pooled container."""
        start_time = time.time()
        execution_id = f"pooled_{int(start_time)}_{id(container)}"

        try:
            # Install requirements if needed
            if request.requirements:
                success = await self._executor.install_requirements(
                    container, request.requirements
                )
                if not success:
                    return SandboxExecutionResponse(
                        execution_result=ExecutionResult.error(
                            "Failed to install requirements",
                            time.time() - start_time,
                        ),
                        test_results=[],
                        passed_tests=0,
                        total_tests=len(request.test_cases),
                        overall_passed=False,
                        execution_id=execution_id,
                    )

            # Execute the main code
            main_code = self._executor._prepare_main_code(
                request.code, request.test_cases
            )
            result = await self._executor._execute_in_container(
                container, main_code, request.config.timeout
            )

            # Execute test cases if present
            test_results = []
            if request.test_cases and result.status == ExecutionStatus.SUCCESS:
                test_results = await self._executor._execute_test_cases(
                    container, request.test_cases
                )

            return SandboxExecutionResponse(
                execution_result=result,
                test_results=test_results,
                passed_tests=sum(1 for t in test_results if t.passed),
                total_tests=len(request.test_cases),
                overall_passed=result.status == ExecutionStatus.SUCCESS
                and all(t.passed for t in test_results),
                execution_id=execution_id,
                finished_at=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Execution error in pooled container: {e}")
            return SandboxExecutionResponse(
                execution_result=ExecutionResult.error(
                    str(e), time.time() - start_time
                ),
                test_results=[],
                passed_tests=0,
                total_tests=len(request.test_cases),
                overall_passed=False,
                execution_id=execution_id,
                finished_at=datetime.now(),
            )

    async def _get_available_container(self) -> Optional[PooledContainer]:
        """Get an available (idle, not retired) container from the pool."""
        async with self._lock:
            for pooled in self._pool:
                if not pooled.is_busy and pooled.use_count < self.max_container_reuse:
                    return pooled
        return None

    async def _create_new_container(self) -> PooledContainer:
        """Create a new Docker container and add to pool."""
        # Create container directly (not using context manager)
        docker_client = self._executor.docker_client
        container_config = {
            'image': self._executor.config.image,
            'detach': True,
            'stdin_open': True,
            'tty': False,
            'working_dir': '/app',
            'mem_limit': f"{self._executor.config.memory_limit}m",
            'memswap_limit': f"{self._executor.config.disk_limit}m",
            'entrypoint': ['tail', '-f', '/dev/null'],
        }

        if not self._executor.config.network_access:
            container_config['network_disabled'] = True

        container = await asyncio.to_thread(docker_client.containers.run, **container_config)

        # Wait for container to be ready
        await asyncio.sleep(0.5)

        pooled = PooledContainer(
            container=container,
            container_id=container.id,
        )

        async with self._lock:
            # Remove oldest idle if pool is full
            if len(self._pool) >= self.pool_size:
                idle_containers = [p for p in self._pool if not p.is_busy]
                if idle_containers:
                    oldest = min(idle_containers, key=lambda p: p.last_used)
                    await self._retire_container_no_lock(oldest)

            self._pool.append(pooled)

        logger.debug(f"Created new container: {pooled.container_id[:12]}")
        return pooled

    async def _retire_container_no_lock(self, pooled: PooledContainer) -> None:
        """Retire and cleanup a container (caller must hold lock)."""
        if pooled in self._pool:
            self._pool.remove(pooled)

        try:
            # Remove the container
            await asyncio.to_thread(pooled.container.remove, force=True)
        except Exception as e:
            logger.warning(
                f"Error cleaning up container {pooled.container_id[:12]}: {e}"
            )

        logger.debug(f"Retired container: {pooled.container_id[:12]}")

    async def _idle_cleanup_loop(self) -> None:
        """Background task to cleanup idle containers."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_containers()
        except asyncio.CancelledError:
            logger.debug("Idle cleanup loop cancelled")
        except Exception as e:
            logger.error(f"Error in idle cleanup loop: {e}")

    async def _cleanup_idle_containers(self) -> None:
        """Remove containers that have been idle for longer than idle_timeout."""
        cutoff = datetime.now() - timedelta(seconds=self.idle_timeout)

        async with self._lock:
            to_retire = [
                pooled
                for pooled in self._pool
                if not pooled.is_busy
                and pooled.last_used < cutoff
                and len(self._pool) > self.pre_warm_count
            ]

            for pooled in to_retire:
                await self._retire_container_no_lock(pooled)

        if to_retire:
            logger.info(f"Cleaned up {len(to_retire)} idle containers")

    def get_metrics(self) -> PoolMetrics:
        """Get current pool metrics."""
        total_use_count = sum(p.use_count for p in self._pool)
        avg_lifetime = 0.0
        if self._pool:
            lifetimes = [
                (datetime.now() - p.created_at).total_seconds() for p in self._pool
            ]
            avg_lifetime = sum(lifetimes) / len(lifetimes)

        reuse_rate = (
            (self._total_executions - len(self._pool)) / self._total_executions
            if self._total_executions > 0
            else 0.0
        )

        return PoolMetrics(
            pool_size=self.pool_size,
            active_containers=len(self._pool),
            busy_containers=sum(1 for p in self._pool if p.is_busy),
            total_executions=self._total_executions,
            container_reuse_rate=reuse_rate,
            avg_container_lifetime=avg_lifetime,
        )

    def get_container_stats(self) -> List[PooledContainerStats]:
        """Get detailed stats for each container in the pool."""
        return [
            PooledContainerStats(
                container_id=p.container_id,
                use_count=p.use_count,
                last_used=p.last_used,
                created_at=p.created_at,
                is_busy=p.is_busy,
            )
            for p in self._pool
        ]

    async def __aenter__(self) -> "SandboxPool":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()
