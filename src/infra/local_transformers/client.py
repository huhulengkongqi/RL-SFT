"""本地 Transformers 客户端 - 不依赖 vLLM，Windows 直接运行."""

import os
import time
from typing import Any, Dict, List, Optional
from threading import Lock

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LocalLLMClient:
    """本地 LLM 客户端，使用 HuggingFace Transformers 直接加载模型运行.

    特点:
    - ✅ 纯 Python，不依赖 vLLM
    - ✅ Windows/Linux/Mac 全平台支持
    - ✅ 自动 GPU/CPU  fallback
    - ✅ 与 VLLMClient / AnthropicClient 相同的 API 接口

    适合本地开发测试，使用小模型 (0.5B, 1.5B)。
    """

    _instance_lock = Lock()
    _model_instance = None
    _tokenizer_instance = None

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = None,
        max_model_len: int = 2048,
        temperature: float = 0.7,
        sleep_before_request: float = 0.0,
        sleep_after_request: float = 0.0,
    ):
        """初始化本地 LLM 客户端.

        Args:
            model: 模型名称或路径, 推荐:
                   - Qwen/Qwen2.5-0.5B-Instruct (~1GB) 🔥
                   - Qwen/Qwen2.5-1.5B-Instruct (~3GB)
            device: 设备, None=自动选择 (cuda → cpu)
            max_model_len: 最大上下文长度
            temperature: 默认温度
            sleep_before_request: 请求前延迟(秒)
            sleep_after_request: 请求后延迟(秒)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "请先安装 transformers 和 torch: "
                "pip install transformers torch accelerate"
            )

        self.model_name = model
        self.temperature = temperature
        self.max_model_len = max_model_len
        self.sleep_before_request = sleep_before_request
        self.sleep_after_request = sleep_after_request

        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._load_model()

    def _load_model(self) -> None:
        """懒加载模型，单例模式，避免重复加载."""
        with self._instance_lock:
            if self._model_instance is not None and self._tokenizer_instance is not None:
                self.tokenizer = self._tokenizer_instance
                self.model = self._model_instance
                return

            print(f"📥 正在加载模型: {self.model_name}")
            print(f"   设备: {self.device}")

            start = time.time()

            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            elapsed = time.time() - start
            print(f"✅ 模型加载完成，耗时 {elapsed:.1f}s")

            # 缓存实例
            LocalLLMClient._model_instance = self.model
            LocalLLMClient._tokenizer_instance = self.tokenizer

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """应用对话模板，兼容 OpenAI 格式的 messages."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def chat(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """同步聊天完成，API 与 VLLMClient 兼容.

        Args:
            model: 忽略，使用初始化时指定的模型
            messages: 消息列表，格式: [{"role": "user", "content": "Hello"}]
            temperature: 温度
            max_tokens: 最大生成 token 数
        """
        if self.sleep_before_request > 0:
            time.sleep(self.sleep_before_request)

        # 应用对话模板
        prompt = self._apply_chat_template(messages or [])

        # tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or 512,
                temperature=temperature or self.temperature,
                do_sample=(temperature or self.temperature) > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 解码，只取新生成的部分
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if self.sleep_after_request > 0:
            time.sleep(self.sleep_after_request)

        return response

    def is_server_ready(self) -> bool:
        """检查模型是否已加载."""
        return self.model is not None and self.tokenizer is not None


# 便捷工厂函数
def create_local_llm_client(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    sleep_before: float = 0.0,
    sleep_after: float = 0.0,
) -> LocalLLMClient:
    """创建本地 LLM 客户端.

    Args:
        model: 模型名称
        sleep_before: 请求前延迟(秒)
        sleep_after: 请求后延迟(秒)
    """
    return LocalLLMClient(
        model=model,
        sleep_before_request=sleep_before,
        sleep_after_request=sleep_after,
    )
