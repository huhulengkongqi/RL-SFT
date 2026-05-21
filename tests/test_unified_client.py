"""统一客户端示例 - 本地 vLLM 和火山 Claude 使用同一套接口."""

import os
import sys
sys.path.insert(0, "src")

# 两个客户端有相同的 API 设计，可以无缝切换！
from src.infra.vllm_client import VLLMClient          # 本地模型
from src.infra.anthropic_client import AnthropicClient  # 火山 Claude


def test_vllm_local():
    """测试本地 vLLM 模型."""
    print("\n" + "=" * 60)
    print("🔷 使用本地 vLLM 模型 (Qwen2.5-0.5B)")
    print("=" * 60)

    # 前提: 先运行 python start_local_vllm.py 启动本地服务器
    client = VLLMClient(
        base_url="http://localhost:8000/v1",
        api_key="dummy",  # 本地不需要 API key
    )

    if not client.is_server_ready():
        print("❌ 本地 vLLM 服务器未启动")
        print("   请先运行: python start_local_vllm.py")
        return

    response = client.chat(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": "用中文介绍一下 Python 编程语言"}],
        temperature=0.7,
        max_tokens=300,
    )

    print(f"✅ 回复: {response}")


def test_volcano_claude():
    """测试火山 Claude API."""
    print("\n" + "=" * 60)
    print("🔷 使用火山 Claude API (ark-code-latest)")
    print("=" * 60)

    # ANTHROPIC_AUTH_TOKEN 从环境变量读取, 不要硬编码!

    if not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        print("❌ 请先设置 ANTHROPIC_AUTH_TOKEN 环境变量")
        return

    client = AnthropicClient(
        base_url=os.environ.get("VOLCANO_CLAUDE_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3"),
        sleep_before_request=1,
        sleep_after_request=1,
    )

    response = client.chat(
        model="ark-code-latest",
        messages=[{"role": "user", "content": "用中文介绍一下 Python 编程语言"}],
        temperature=0.7,
        max_tokens=300,
    )

    print(f"✅ 回复: {response}")


def main():
    print("🚀 统一客户端测试")
    print("   两个客户端使用几乎相同的 API！")

    # 测试 1: 火山 Claude (需要网络)
    test_volcano_claude()

    # 测试 2: 本地 vLLM (需要先启动服务器)
    # test_vllm_local()


if __name__ == "__main__":
    main()
