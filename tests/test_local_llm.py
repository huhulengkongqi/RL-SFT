"""测试本地 LLM - 纯 Transformers 版本，Windows 直接运行，不依赖 vLLM！"""

import os

# 国内 HuggingFace 镜像 - 必须在导入 transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.insert(0, "src")

from src.infra.local_transformers import LocalLLMClient, create_local_llm_client


def main():
    print("=" * 60)
    print("🚀 本地 LLM 测试 - Qwen2.5-0.5B")
    print("   纯 Transformers 实现，Windows 直接运行")
    print("   不依赖 vLLM，不需要 Docker")
    print("=" * 60)

    # 可选模型:
    #   "Qwen/Qwen2.5-0.5B-Instruct"  ~1GB  🔥 最快，推荐
    #   "Qwen/Qwen2.5-1.5B-Instruct"  ~3GB
    #   "Qwen/Qwen2.5-3B-Instruct"    ~6GB

    try:
        # 创建客户端（首次运行会自动从 HuggingFace 下载模型）
        print("\n📦 初始化客户端...")
        client = create_local_llm_client(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            sleep_before=0,
            sleep_after=0,
        )

        print("\n🧪 开始推理测试...")
        print("-" * 60)

        # 测试 1: 简单问候
        prompt = "Hello! 请用中文介绍一下你自己。"
        print(f"\n👤 用户: {prompt}")

        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )

        print(f"🤖 模型: {response}")

        # 测试 2: 多轮对话
        print("\n" + "-" * 60)
        prompt2 = "1+1等于几？"
        print(f"\n👤 用户: {prompt2}")

        response2 = client.chat(
            messages=[{"role": "user", "content": prompt2}],
            temperature=0.1,  # 低温度，更确定
            max_tokens=100,
        )

        print(f"🤖 模型: {response2}")

        print("\n" + "=" * 60)
        print("✅ 测试完成！")
        print("   你可以使用 client.chat() 进行任意对话")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        print("\n💡 解决方法:")
        print("   1. 安装依赖: pip install transformers torch accelerate")
        print("   2. 如果下载慢，设置 HuggingFace 镜像:")
        print("      set HF_ENDPOINT=https://hf-mirror.com")
        print("   3. 确保有至少 2GB 可用内存")


if __name__ == "__main__":
    main()
