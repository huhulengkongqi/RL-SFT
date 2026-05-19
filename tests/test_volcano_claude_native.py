"""Simple test script for Volcano Claude API - 使用封装后的客户端."""

import os

# ==============================================
# 环境变量配置 (也可以在系统环境中设置)
# ==============================================
os.environ["ANTHROPIC_AUTH_TOKEN"] = "***REMOVED***"
os.environ["VOLCANO_CLAUDE_BASE_URL"] = "https://ark.cn-beijing.volces.com/api/coding/v3"
os.environ["API_TIMEOUT_MS"] = "3000000"
os.environ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

# ==============================================
# 一行代码导入并创建客户端！
# ==============================================
from src.infra.anthropic_client import AnthropicClient, create_volcano_claude_client

MODEL_NAME = "ark-code-latest"


def main():
    print(f"🚀 正在请求火山Claude API (使用封装客户端)...")
    print(f"   Base URL: {os.environ['VOLCANO_CLAUDE_BASE_URL']}")
    print(f"   Model: {MODEL_NAME}")
    print("-" * 60)

    try:
        # 方式1: 使用工厂函数 (最简单，推荐)
        client = create_volcano_claude_client(sleep_before=10, sleep_after=10)

        # 方式2: 手动创建 (更灵活)
        # client = AnthropicClient(
        #     base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        #     api_key="***REMOVED***",
        #     sleep_before_request=10,
        #     sleep_after_request=10,
        # )

        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=4096,
        )

        print("✅ 响应成功!")
        print(f"📝 回复: {response}")

    except Exception as e:
        print(f"❌ 请求失败: {type(e).__name__}: {e}")
        print("\n💡 排查建议:")
        print("   - 401: 检查 ANTHROPIC_AUTH_TOKEN 是否正确")
        print("   - 404: 检查 BASE_URL 路径和版本")
        print("   - 超时: 增加 timeout 值或检查网络连接")
        print("   - 模型不存在: 确认火山Claude支持的模型名称")


if __name__ == "__main__":
    main()
