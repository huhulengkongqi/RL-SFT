"""vLLM Inference Client - OpenAI API compatible wrapper."""

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


class Message(BaseModel):
    """Chat message."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    model: str
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False


class VLLMClient:
    """vLLM inference client supporting both sync and async OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self.base_url = base_url
        self.api_key = api_key or "dummy"
        self.timeout = timeout

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous chat completion."""
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def achat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronous chat completion."""
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def achat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Asynchronous streaming chat completion."""
        stream = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def batch_chat(
        self,
        requests: List[Dict[str, Any]],
        max_concurrency: int = 5,
    ) -> List[str]:
        """Batch chat completions with concurrency limit."""
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_chat(req: Dict[str, Any]) -> str:
            async with semaphore:
                return await self.achat(**req)

        async def run_all() -> List[str]:
            tasks = [bounded_chat(req) for req in requests]
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())

    def is_server_ready(self) -> bool:
        """Check if vLLM server is ready and responding."""
        try:
            health_url = self.base_url.replace("/v1", "/health")
            response = httpx.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_server(self, timeout_seconds: int = 300, check_interval: int = 5) -> bool:
        """Wait for vLLM server to be ready."""
        start = time.time()
        while time.time() - start < timeout_seconds:
            if self.is_server_ready():
                return True
            time.sleep(check_interval)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Client Test")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM server base URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    args = parser.parse_args()

    client = VLLMClient(base_url=args.base_url)

    if client.is_server_ready():
        print(f"vLLM server at {args.base_url} is READY")
    else:
        print(f"vLLM server at {args.base_url} is NOT available")
