# vLLM 部署指南

## ⚠️ 重要前置条件

**NVIDIA 驱动要求:** vLLM Docker 镜像需要驱动版本 **>= 550.x** (CUDA 12.4+)

当前检测：`596.21` (CUDA 13.2) - **✓ 满足要求**

**解决方案：更新 NVIDIA 驱动**
- 下载地址：https://www.nvidia.com/Download/index.aspx
- 选择 Game Ready Driver 或 Studio Driver (550.x 或更高)

---

## 环境说明

- GPU: NVIDIA GeForce RTX (6GB VRAM)
- CUDA: 驱动更新后支持 12.4+
- Docker: WSL2 backend with GPU support

---

## 部署选项

### 选项 1: Docker 部署（推荐）

**先决条件:**
- Docker Desktop with WSL2 backend
- NVIDIA Container Toolkit
- NVIDIA 驱动 >= 550.x

**步骤:**
```bash
# 复制 .env
cp .env.example .env
# 编辑 HF_TOKEN

# 构建并启动
docker compose build vllm
docker compose up -d vllm

# 查看日志
docker compose logs -f vllm
```

### 选项 2: WSL2 本地安装

直接在 WSL2 中安装（CUDA 兼容性更好）：

```bash
# 在 WSL2 中
pip install vllm

# 启动服务器
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --gpu-memory-utilization 0.85 \
  --quantization awq \
  --enable-prefix-caching
```

### 选项 3: 现有 vLLM 服务器

如果已有远程 vLLM 服务器，直接在 `.env` 中修改：
```
VLLM_BASE_URL=http://your-server:8000/v1
```

---

## 模型选择（6GB VRAM）

必须使用量化版本：
- ✅ `Qwen/Qwen2.5-7B-Instruct-AWQ` (推荐)
- ❌ 原版 FP16 模型无法加载

---

## 验证

```bash
# 健康检查
curl http://localhost:8000/health

# 测试推理
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct-AWQ","messages":[{"role":"user","content":"Hello!"}]}'

# 运行测试
uv run pytest tests/test_vllm_client.py -v
```

---

## 驱动更新验证

更新驱动后运行：
```bash
nvidia-smi
# 应该显示 CUDA Version: 12.4 或更高

# 验证 Docker GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
