#!/bin/bash
set -e

# 默认镜像
DEFAULT_IMAGE="ghcr.io/huhulengkongqi/rl-sft-vllm:v0.6.3-rlsft.1"
VLLM_IMAGE=${VLLM_IMAGE:-$DEFAULT_IMAGE}

echo "========================================"
echo "  预拉取 vLLM Docker 镜像"
echo "  镜像: $VLLM_IMAGE"
echo "========================================"

# 检查是否已存在
if docker inspect "$VLLM_IMAGE" > /dev/null 2>&1; then
  echo "✅ 镜像已存在于本地"
  docker inspect --format='{{.Id}}' "$VLLM_IMAGE"
  exit 0
fi

echo "⬇️  开始拉取镜像..."
docker pull "$VLLM_IMAGE"

echo ""
echo "✅ 镜像拉取完成！"
docker inspect --format='镜像 ID: {{.Id}}' "$VLLM_IMAGE"
docker inspect --format='大小: {{.Size}} 字节' "$VLLM_IMAGE"

echo ""
echo "接下来可以运行:"
echo "  ./scripts/start_vllm_docker.sh"
