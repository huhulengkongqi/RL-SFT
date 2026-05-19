"""Anthropic Native API Client - 火山Claude原生协议客户端."""

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from anthropic import Anthropic, AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Anthropic native API client supporting sync/async calls and Thinking mode.

    专为火山 Claude API 设计，完整支持:
    - Thinking 扩展思考模式
    - Prompt Caching
    - 原生工具调用格式
    - 流式响应
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        sleep_before_request: float = 0.0,
        sleep_after_request: float = 0.0,
    ):
        """初始化客户端.

        Args:
            base_url: API 端点, 例如: "https://ark.cn-beijing.volces.com/api/coding/v3"
            api_key: API 密钥, 如未提供则从环境变量 ANTHROPIC_AUTH_TOKEN 读取
            timeout: 请求超时时间(秒)
            sleep_before_request: 每次请求前的延迟(秒), 用于避免速率限制
            sleep_after_request: 每次请求后的延迟(秒), 用于避免速率限制
        """
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        self.timeout = timeout
        self.sleep_before_request = sleep_before_request
        self.sleep_after_request = sleep_after_request

        if not self.api_key:
            raise ValueError("请提供 api_key 或设置环境变量 ANTHROPIC_AUTH_TOKEN")

        # HTTP client for health checks
        self._http_client = httpx.Client(base_url=base_url, timeout=timeout)

        # Sync client
        self._client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        # Async client
        self._async_client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _apply_sleep_before(self) -> None:
        """请求前延迟."""
        if self.sleep_before_request > 0:
            time.sleep(self.sleep_before_request)

    def _apply_sleep_after(self) -> None:
        """请求后延迟."""
        if self.sleep_after_request > 0:
            time.sleep(self.sleep_after_request)

    @staticmethod
    def _extract_text_from_content(content: List[Any]) -> str:
        """从 Anthropic response content 中提取文本, 处理 Thinking 模式."""
        full_response = []
        has_thinking = False

        for block in content:
            block_type = getattr(block, "type", "unknown")
            if block_type == "text":
                full_response.append(block.text)
            elif block_type == "thinking":
                has_thinking = True
            else:
                full_response.append(str(block))

        if has_thinking and full_response:
            # 如果有思考内容，在回复开头标记（可选）
            pass

        return "\n".join(full_response) if full_response else ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """同步聊天完成.

        Args:
            model: 模型名称, 例如: "ark-code-latest"
            messages: 消息列表, 格式: [{"role": "user", "content": "Hello"}]
            temperature: 温度参数
            max_tokens: 最大生成token数
        """
        self._apply_sleep_before()

        # 转换 messages 格式为 Anthropic 原生格式
        # Anthropic messages 格式和 OpenAI 兼容
        message = self._client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        self._apply_sleep_after()

        return self._extract_text_from_content(message.content)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def achat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """异步聊天完成."""
        if self.sleep_before_request > 0:
            import asyncio
            await asyncio.sleep(self.sleep_before_request)

        message = await self._async_client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if self.sleep_after_request > 0:
            import asyncio
            await asyncio.sleep(self.sleep_after_request)

        return self._extract_text_from_content(message.content)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def achat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """异步流式聊天完成."""
        if self.sleep_before_request > 0:
            import asyncio
            await asyncio.sleep(self.sleep_before_request)

        stream = await self._async_client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                yield event.delta.text

        if self.sleep_after_request > 0:
            import asyncio
            await asyncio.sleep(self.sleep_after_request)

    def is_server_ready(self) -> bool:
        """检查 API 服务是否可用."""
        try:
            # 尝试访问 health 端点或根路径
            response = self._http_client.get("/health", timeout=5)
            return response.status_code == 200
        except Exception:
            try:
                # 如果没有 health 端点，尝试根路径
                response = self._http_client.get("/", timeout=5)
                return response.status_code < 500
            except Exception:
                return False


# 便捷工厂函数
def create_volcano_claude_client(
    model: str = "ark-code-latest",
    sleep_before: float = 1.0,
    sleep_after: float = 2.0,
) -> AnthropicClient:
    """创建预配置的火山 Claude 客户端.

    从环境变量读取配置:
    - ANTHROPIC_AUTH_TOKEN: API 密钥
    - VOLCANO_CLAUDE_BASE_URL: API 端点

    Args:
        model: 默认模型名称
        sleep_before: 请求前延迟(秒)
        sleep_after: 请求后延迟(秒)
    """
    base_url = os.environ.get(
        "VOLCANO_CLAUDE_BASE_URL",
        "https://ark.cn-beijing.volces.com/api/coding/v3",
    )

    return AnthropicClient(
        base_url=base_url,
        sleep_before_request=sleep_before,
        sleep_after_request=sleep_after,
    )
