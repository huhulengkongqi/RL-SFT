#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 加载环境变量
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

# 默认配置
VLLM_MODE=${VLLM_MODE:-pull}

echo "========================================"
echo "  启动 vLLM Docker 容器"
echo "  模式: $VLLM_MODE"
echo "========================================"

# 检查 NVIDIA Docker 支持
if ! docker info | grep -q "nvidia"; then
  echo "⚠️  警告: 未检测到 NVIDIA Docker 运行时"
  echo "   请确保已安装 NVIDIA Container Toolkit"
  read -p "   继续? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# 根据模式选择服务
case "$VLLM_MODE" in
  "pull")
    SERVICE="vllm"
    COMPOSE_CMD=""
    ;;
  "build")
    SERVICE="vllm-build"
    COMPOSE_CMD="--build"
    ;;
  "full")
    SERVICE="vllm-full"
    COMPOSE_CMD="--build"
    ;;
  *)
    echo "❌ 未知模式: $VLLM_MODE"
    echo "   支持的模式: pull, build, full"
    exit 1
    ;;
esac

echo "🔧 启动服务: $SERVICE"

# 启动容器
docker compose up -d $COMPOSE_CMD $SERVICE

echo "⏳ 等待服务就绪..."
sleep 10

# 检查容器状态
if ! docker compose ps | grep -q "Up"; then
  echo "❌ 容器启动失败"
  docker compose logs --tail=20
  exit 1
fi

echo "✅ 容器已启动"
echo "📝 日志输出:"
docker compose logs -f $SERVICE
