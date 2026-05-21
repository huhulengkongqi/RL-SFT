"""Simple test script for Volcano Claude API."""

import os
import time

# 首先设置环境变量
os.environ["API_TIMEOUT_MS"] = "3000000"
os.environ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

from src.infra.vllm_client.client import VLLMClient

# ==============================================
# 配置信息 (从环境变量读取, 或在 .env 文件中设置)
# ==============================================
BASE_URL = os.environ.get("VOLCANO_CLAUDE_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
API_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")  # 从环境变量读取 API Key
MODEL_NAME = "ark-code-latest"  # 模型名称, 例如: "claude-3-sonnet"
# ==============================================


def main():
    # 检查配置
    if not BASE_URL or not API_KEY or not MODEL_NAME:
        print("❌ 请先在脚本中填入 BASE_URL, API_KEY 和 MODEL_NAME!")
        return

    # 1. 初始化客户端
    client = VLLMClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=120,
    )

    # 2. 发送 Hello 测试请求
    messages = [
        {"role": "user", "content": "Hello"}
    ]

    print(f"🚀 正在请求火山Claude API...")
    print(f"   Base URL: {BASE_URL}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Message: {messages[0]['content']}")
    print("-" * 60)

    # 请求前延迟: 避免速率限制
    time.sleep(10)
    print(f"   ⏱️  请求前预热延迟 10s 完成")

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )

        print("✅ 响应成功!")
        print(f"📝 回复: {response}")

        # 请求后冷却延迟
        time.sleep(10)
        print(f"   ⏱️  请求后冷却延迟 10s 完成")

    except Exception as e:
        print(f"❌ 请求失败: {type(e).__name__}: {e}")
        print("\n💡 排查建议:")
        print("   - 401: 检查 API_KEY 是否正确")
        print("   - 404: 检查 BASE_URL 是否正确")
        print("   - 超时: 增加 timeout 值或检查网络连接")
        print("   - 模型不存在: 确认火山Claude支持的模型名称")


if __name__ == "__main__":
    main()
