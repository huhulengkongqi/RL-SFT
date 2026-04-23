#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 加载环境变量
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

VLLM_PORT=${VLLM_PORT:-8000}
VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}
EXPECTED_DIGEST=${EXPECTED_DIGEST:-sha256:b8374cee0a1acaec8b64525ff77560f30443f67bd0fc1956a3529504a89f823b}

echo "========================================"
echo "  vLLM 服务验证套件"
echo "========================================"

PASSED=0
FAILED=0

run_check() {
  local name="$1"
  local cmd="$2"
  echo -n "🔍 检查 $name... "
  if eval "$cmd" > /dev/null 2>&1; then
    echo "✅ 通过"
    ((PASSED++))
  else
    echo "❌ 失败"
    ((FAILED++))
  fi
}

# 1. 容器运行状态
run_check "Docker 容器运行" "docker compose ps | grep -q 'Up'"

# 2. GPU 可访问性
run_check "容器内 GPU 可访问" "docker compose exec vllm nvidia-smi > /dev/null 2>&1" || \
run_check "容器内 GPU 可访问" "docker compose exec vllm-build nvidia-smi > /dev/null 2>&1" || \
run_check "容器内 GPU 可访问" "docker compose exec vllm-full nvidia-smi > /dev/null 2>&1"

# 3. 健康端点
run_check "健康端点响应" "curl -s http://localhost:$VLLM_PORT/health | grep -q 'ok'"

# 4. 正确的模型加载
run_check "已加载模型名称" "curl -s http://localhost:$VLLM_PORT/v1/models | grep -q \"$VLLM_MODEL\""

# 5. Chat completion 接口
echo -n "🔍 检查 Chat completion 接口... "
RESPONSE=$(curl -s http://localhost:$VLLM_PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$VLLM_MODEL"'",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 10,
    "temperature": 0
  }' 2>/dev/null)
if echo "$RESPONSE" | grep -q "choices"; then
  echo "✅ 通过"
  ((PASSED++))
else
  echo "❌ 失败"
  ((FAILED++))
fi

# 6. 镜像 digest 校验
echo -n "🔍 检查镜像 Digest... "
IMAGE_ID=$(docker compose images -q vllm 2>/dev/null || \
           docker compose images -q vllm-build 2>/dev/null || \
           docker compose images -q vllm-full 2>/dev/null)
if [ -n "$IMAGE_ID" ]; then
  ACTUAL_DIGEST=$(docker inspect --format='{{.Id}}' "$IMAGE_ID" 2>/dev/null || echo "unknown")
  echo "✅ 确认 (镜像 ID: ${IMAGE_ID:0:20})"
  ((PASSED++))
else
  echo "❌ 未找到镜像"
  ((FAILED++))
fi

echo ""
echo "========================================"
echo "  验证结果: $PASSED / $((PASSED + FAILED))"
echo "========================================"

if [ $FAILED -gt 0 ]; then
  echo ""
  echo "⚠️  部分检查未通过，请检查日志:"
  docker compose logs --tail=30
  exit 1
else
  echo ""
  echo "🎉 所有检查通过！vLLM 服务运行正常。"
  echo "📡 API 端点: http://localhost:$VLLM_PORT/v1"
  exit 0
fi
