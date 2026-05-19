"""启动本地 vLLM 服务器，使用小模型测试."""

import sys
sys.path.insert(0, "src")

from src.infra.vllm_client.server import start_local_server, VLLMServerConfig
from src.infra.vllm_client.client import VLLMClient
import time

# ==============================================
# 配置 - 选择你想跑的模型
# ==============================================
# 推荐的小模型 (越小越快):
# - Qwen/Qwen2.5-0.5B-Instruct  ~1GB   🔥 最快，推荐测试用
# - Qwen/Qwen2.5-1.5B-Instruct   ~3GB
# - Qwen/Qwen2.5-3B-Instruct     ~6GB
# - Qwen/Qwen2.5-7B-Instruct-AWQ ~4GB   (项目默认，需要 8GB 显存)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8000
GPU_MEM_UTIL = 0.5  # 0.5 = 使用 50% GPU 显存


def main():
    print(f"🚀 启动本地 vLLM 服务器")
    print(f"   模型: {MODEL}")
    print(f"   端口: {PORT}")
    print(f"   GPU 显存利用率: {GPU_MEM_UTIL}")
    print("-" * 60)
    print("   首次启动会自动从 HuggingFace 下载模型...")
    print("   请耐心等待 (0.5B 模型约 1GB)")
    print("-" * 60)

    # 1. 启动服务器 (非阻塞)
    server = start_local_server(
        model=MODEL,
        port=PORT,
        gpu_memory_utilization=GPU_MEM_UTIL,
        quantization=None,  # 小模型不需要量化
        log_file="logs/vllm_server.log",
    )

    print(f"\n⏳ 等待服务器启动完成... (约 10-30 秒)")

    # 2. 等待服务器就绪
    client = VLLMClient(base_url=f"http://localhost:{PORT}/v1")
    max_wait = 300  # 最多等 5 分钟
    waited = 0
    check_interval = 5

    while waited < max_wait:
        if client.is_server_ready():
            print(f"✅ vLLM 服务器已就绪！")
            break
        time.sleep(check_interval)
        waited += check_interval
        print(f"   已等待 {waited}s...")
    else:
        print(f"❌ 服务器启动超时，请检查日志: logs/vllm_server.log")
        server.stop()
        return

    # 3. 测试推理
    print("\n" + "=" * 60)
    print("🧪 开始测试推理...")
    print("=" * 60)

    try:
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "Hello! 请用中文介绍一下你自己。"}],
            temperature=0.7,
            max_tokens=500,
        )
        print(f"\n📝 模型回复:")
        print(response)
        print("\n" + "=" * 60)
        print("✅ 测试成功！服务器正在运行")
        print(f"   API 地址: http://localhost:{PORT}/v1")
        print(f"   兼容 OpenAI 协议，可以用 VLLMClient 调用")
        print("   按 Ctrl+C 停止服务器")
        print("=" * 60)
    except Exception as e:
        print(f"❌ 测试失败: {type(e).__name__}: {e}")
        server.stop()
        return

    # 保持运行
    try:
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务器...")
        server.stop()
        print("✅ 服务器已停止")


if __name__ == "__main__":
    main()
