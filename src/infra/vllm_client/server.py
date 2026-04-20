"""vLLM Server management utilities."""

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class VLLMServerConfig:
    """Configuration for vLLM server."""
    model: str
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    port: int = 8000
    host: str = "0.0.0.0"
    dtype: str = "auto"
    quantization: Optional[str] = None
    enable_prefix_caching: bool = True
    disable_log_stats: bool = False
    trust_remote_code: bool = True
    extra_env_vars: Dict[str, str] = field(default_factory=dict)


class VLLMServer:
    """vLLM server process manager."""

    def __init__(self, config: VLLMServerConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._log_file: Optional[Path] = None

    def _build_command(self) -> List[str]:
        """Build vLLM api server command."""
        cmd = [
            "vllm", "serve", self.config.model,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--dtype", self.config.dtype,
        ]

        if self.config.max_model_len:
            cmd.extend(["--max-model-len", str(self.config.max_model_len)])

        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])

        if self.config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        if self.config.disable_log_stats:
            cmd.append("--disable-log-stats")

        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")

        return cmd

    def start(self, log_file: Optional[str] = None, block: bool = False) -> None:
        """Start the vLLM server.

        Args:
            log_file: Path to write server logs
            block: If True, block until server exits
        """
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("vLLM server is already running")

        cmd = self._build_command()

        env = os.environ.copy()
        env.update(self.config.extra_env_vars)

        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT

        if log_file:
            self._log_file = Path(log_file)
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            stdout = open(log_file, "w", encoding="utf-8")

        self._process = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            env=env,
            text=True,
        )

        if block:
            self._process.wait()

    def stop(self) -> int:
        """Stop the vLLM server. Returns exit code."""
        if self._process is None:
            return -1

        self._process.terminate()
        try:
            return self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            return self._process.wait()

    def is_running(self) -> bool:
        """Check if server process is running."""
        return self._process is not None and self._process.poll() is None

    @property
    def base_url(self) -> str:
        """Get server base URL for OpenAI API."""
        return f"http://{self.config.host}:{self.config.port}/v1"


def start_local_server(
    model: str,
    port: int = 8000,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    log_file: Optional[str] = None,
    **kwargs,
) -> VLLMServer:
    """Convenience function to start a local vLLM server."""
    config = VLLMServerConfig(
        model=model,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        **kwargs,
    )
    server = VLLMServer(config)
    server.start(log_file=log_file, block=False)
    return server


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start vLLM Server")
    parser.add_argument("model", help="Model name or path")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--gpu-mem", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--quantization", help="Quantization method (awq, gptq, fp8, etc.)")
    parser.add_argument("--max-model-len", type=int, help="Maximum model context length")

    args = parser.parse_args()

    print(f"Starting vLLM server with model: {args.model}")
    server = start_local_server(
        model=args.model,
        port=args.port,
        gpu_memory_utilization=args.gpu_mem,
        tensor_parallel_size=args.tensor_parallel,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        log_file=args.log_file,
    )

    print(f"Server starting on port {args.port}... Press Ctrl+C to stop")
    try:
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
        print("Server stopped.")
