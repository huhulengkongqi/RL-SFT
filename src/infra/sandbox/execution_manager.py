"""
Docker-based sandbox execution manager for code execution.
"""

import asyncio
import docker
import os
import tempfile
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from .models import (
    ExecutionResult, SandboxConfig, SandboxExecutionRequest,
    SandboxExecutionResponse, TestCaseExecution, ExecutionStatus,
    ValidationErrorType
)


class SandboxExecutor:
    """Docker container-based code execution manager"""

    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self.docker_client = None
        self._initialize_docker()

    def _initialize_docker(self):
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
            # Test Docker connection
            self.docker_client.ping()
            print("Docker client initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Docker: {e}")

    @asynccontextmanager
    async def _create_container(self):
        """Create and manage Docker container lifecycle"""
        container = None
        try:
            # Prepare container configuration
            container_config = {
                'image': self.config.image,
                'detach': True,
                'stdin_open': True,
                'tty': False,
                'working_dir': '/app',
                'mem_limit': f"{self.config.memory_limit}m",
                'memswap_limit': f"{self.config.disk_limit}m",
            }

            if not self.config.network_access:
                container_config['network_disabled'] = True

            # Create container
            container = self.docker_client.containers.run(**container_config)

            # Wait for container to be ready
            await asyncio.sleep(0.5)
            yield container

        except Exception as e:
            if container:
                container.remove(force=True)
            raise e
        finally:
            if container:
                container.remove(force=True)

    async def install_requirements(self, container, requirements: List[str]) -> bool:
        """Install Python packages in container"""
        if not requirements:
            return True

        requirements_str = "\n".join(requirements)
        install_cmd = f"""
        python -m pip install --no-cache-dir -r /tmp/requirements.txt &&
        rm -f /tmp/requirements.txt
        """

        # Create requirements file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements_str)
            requirements_file = f.name

        try:
            # Copy requirements to container
            with open(requirements_file, 'rb') as f:
                container.put_archive('/tmp', f.name)

            # Install requirements
            result = container.exec_run(
                cmd=install_cmd,
                stdout=True,
                stderr=True,
                demux=True
            )

            if result.exit_code != 0:
                error_output = result.stderr.decode('utf-8')
                print(f"Failed to install requirements: {error_output}")
                return False

            return True

        except Exception as e:
            print(f"Error installing requirements: {e}")
            return False
        finally:
            os.unlink(requirements_file)

    async def execute_code(self, request: SandboxExecutionRequest) -> SandboxExecutionResponse:
        """Execute code in sandbox"""
        start_time = time.time()
        execution_id = f"exec_{int(start_time)}"

        try:
            async with self._create_container() as container:
                # Install requirements if needed
                if request.requirements:
                    success = await self.install_requirements(container, request.requirements)
                    if not success:
                        return SandboxExecutionResponse(
                            execution_result=ExecutionResult.error(
                                "Failed to install requirements",
                                time.time() - start_time
                            ),
                            test_results=[],
                            passed_tests=0,
                            total_tests=len(request.test_cases),
                            overall_passed=False,
                            execution_id=execution_id
                        )

                # Prepare test files
                if request.test_cases:
                    test_files = self._prepare_test_files(request.test_cases)
                    for file_path, content in test_files.items():
                        with open(content, 'w') as f:
                            f.write(file_path)
                        # Copy to container
                        container.put_archive('/app', file_path)

                # Execute the main code
                main_code = self._prepare_main_code(request.code, request.test_cases)
                result = await self._execute_in_container(container, main_code, request.config.timeout)

                # Execute test cases if present
                test_results = []
                if request.test_cases and result.status == ExecutionStatus.SUCCESS:
                    test_results = await self._execute_test_cases(container, request.test_cases)

                return SandboxExecutionResponse(
                    execution_result=result,
                    test_results=test_results,
                    passed_tests=sum(1 for t in test_results if t.passed),
                    total_tests=len(request.test_cases),
                    overall_passed=result.status == ExecutionStatus.SUCCESS and
                                   all(t.passed for t in test_results),
                    execution_id=execution_id,
                    finished_at=datetime.now()
                )

        except Exception as e:
            return SandboxExecutionResponse(
                execution_result=ExecutionResult.error(str(e), time.time() - start_time),
                test_results=[],
                passed_tests=0,
                total_tests=len(request.test_cases),
                overall_passed=False,
                execution_id=execution_id,
                finished_at=datetime.now()
            )

    async def _execute_in_container(self, container, code: str, timeout: int) -> ExecutionResult:
        """Execute code in container with timeout"""
        script_path = "/tmp/execute.py"

        # Write code to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Copy script to container
            with open(temp_file, 'rb') as f:
                container.put_archive('/tmp', (script_path, f.read()))

            # Execute script
            exec_cmd = f"python {script_path}"
            result = container.exec_run(
                cmd=exec_cmd,
                stdout=True,
                stderr=True,
                demux=True,
                socket=True,
                workdir='/app'
            )

            # Read output with timeout
            stdout = []
            stderr = []
            start_time = time.time()

            try:
                for chunk in result.output.iter_bytes():
                    if time.time() - start_time > timeout:
                        result.close()
                        break

                    chunk_str = chunk.decode('utf-8', errors='replace')
                    if result.output._stdout is not None:
                        stdout.append(chunk_str)
                    if result.output._stderr is not None:
                        stderr.append(chunk_str)

            except Exception:
                # Timeout or other error
                result.close()
                return ExecutionResult.timeout(time.time() - start_time)

            # Check exit code
            if result.exit_code is not None:
                if result.exit_code == 0:
                    output = ''.join(stdout)
                    return ExecutionResult.success(
                        output=output,
                        execution_time=time.time() - start_time
                    )
                else:
                    error = ''.join(stderr) or 'Execution failed'
                    return ExecutionResult.failure(
                        error=error,
                        execution_time=time.time() - start_time,
                        traceback=''.join(stderr)
                    )

        except Exception as e:
            return ExecutionResult.error(str(e), time.time() - start_time)

        finally:
            os.unlink(temp_file)

    async def _execute_test_cases(self, container, test_cases: List[Dict]) -> List[TestCaseExecution]:
        """Execute test cases in container"""
        results = []

        for i, test_case in enumerate(test_cases):
            test_code = self._generate_test_code(test_case, i)
            start_time = time.time()

            try:
                exec_cmd = f"python -c \"{test_code}\""
                result = container.exec_run(
                    cmd=exec_cmd,
                    stdout=True,
                    stderr=True,
                    demux=True
                )

                execution_time = time.time() - start_time

                if result.exit_code == 0:
                    # Test passed
                    results.append(TestCaseExecution(
                        passed=True,
                        output="",
                        execution_time=execution_time,
                        test_case_index=i
                    ))
                else:
                    # Test failed
                    error = result.stderr.decode('utf-8', errors='replace')
                    results.append(TestCaseExecution(
                        passed=False,
                        error=error,
                        execution_time=execution_time,
                        test_case_index=i,
                        error_type=ValidationErrorType.ASSERTION_FAILURE
                    ))

            except Exception as e:
                execution_time = time.time() - start_time
                results.append(TestCaseExecution(
                    passed=False,
                    error=str(e),
                    execution_time=execution_time,
                    test_case_index=i,
                    error_type=ValidationErrorType.RUNTIME_ERROR
                ))

        return results

    def _prepare_test_files(self, test_cases: List[Dict]) -> Dict[str, str]:
        """Prepare test files for execution"""
        files = {}

        # Create test data files if needed
        for i, test_case in enumerate(test_cases):
            if 'input_data' in test_case:
                filename = f"/tmp/test_input_{i}.json"
                import json
                with open(filename, 'w') as f:
                    json.dump(test_case['input_data'], f)
                files[filename] = filename

        return files

    def _prepare_main_code(self, code: str, test_cases: List[Dict]) -> str:
        """Prepare main execution code"""
        main_code = code

        # If there are test cases, wrap the code to capture output
        if test_cases:
            main_code = f"""
import sys
import contextlib

@contextlib.contextmanager
def capture_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_output = []

    class Capture:
        def write(self, data):
            captured_output.append(data)
        def flush(self):
            pass

    sys.stdout = Capture()
    sys.stderr = Capture()

    try:
        yield captured_output
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Execute the main code
with capture_output() as output:
{chr(10).join('    ' + line for line in code.splitlines())}

# Print output for capture
if output:
    print(''.join(output))
"""

        return main_code

    def _generate_test_code(self, test_case: Dict, index: int) -> str:
        """Generate test case execution code"""
        if 'validator_code' in test_case:
            return test_case['validator_code']
        else:
            # Simple test case generation
            input_data = test_case.get('input', {})
            expected = test_case.get('expected_output', None)

            if expected is not None:
                return f"""
# Test case {index}
result = execute_main({input_data})
assert result == {repr(expected)}, f"Expected {repr(expected)}, got {{result}}"
print("Test passed")
"""
            else:
                return f"""
# Test case {index}
result = execute_main({input_data})
print("Test completed:", result)
"""

    async def health_check(self) -> bool:
        """Check if sandbox is healthy"""
        try:
            async with self._create_container() as container:
                # Simple ping test
                result = container.exec_run(cmd="python -c 'print(\"OK\")'", stdout=True)
                return result.exit_code == 0
        except Exception:
            return False

    async def close(self):
        """Clean up resources"""
        if self.docker_client:
            self.docker_client.close()