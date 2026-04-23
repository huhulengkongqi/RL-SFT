# vLLM 部署指南

## ⚠️ 重要前置条件

**NVIDIA 驱动要求:** vLLM Docker 镜像需要驱动版本 **>= 550.x** (CUDA 12.4+)

**解决方案：更新 NVIDIA 驱动**
- 下载地址：https://www.nvidia.com/Download/index.aspx
- 选择 Game Ready Driver 或 Studio Driver (550.x 或更高)

---

## 环境说明

- GPU: NVIDIA GeForce RTX (6GB+ VRAM)
- CUDA: 驱动更新后支持 12.4+
- Docker: WSL2 backend with GPU support

---

## 部署选项

### 三种 Docker 部署模式对比

| 模式 | 描述 | 启动时间 | 推荐场景 |
|------|------|----------|----------|
| **pull** (默认) | 直接拉取 GHCR 上的预构建镜像 | ⚡ ~30秒 | ✅ **大多数用户首选** |
| **build** | 基于官方 vLLM 镜像添加依赖 | ⏱️ ~2分钟 | 需要自定义依赖时 |
| **full** | 从 CUDA 基础镜像完整构建 | ⏳ ~10分钟 | 最高控制度 / 开发环境 |

---

### 🚀 选项 1: 快速部署 (Pull 模式 - 推荐)

**最简单、最快的方式** - 使用项目预构建的可复现镜像

```bash
# 1. 配置环境
cp .env.example .env
# 编辑 .env 设置 HF_TOKEN
# VLLM_MODE=pull (默认)

# 2. 启动服务
./scripts/start_vllm_docker.sh
```

---

### 🛠️ 选项 2: 构建模式 (Build 模式)

基于官方 vLLM 镜像构建，可复现的构建过程

```bash
# 1. 配置环境
cp .env.example .env
# 编辑 .env:
#   VLLM_MODE=build

# 2. 启动服务（自动构建）
./scripts/start_vllm_docker.sh
```

---

### 🔧 选项 3: 完整构建 (Full 模式)

从 CUDA 基础镜像开始完整构建，最高控制度

```bash
# 1. 配置环境
cp .env.example .env
# 编辑 .env:
#   VLLM_MODE=full

# 2. 启动服务（自动构建）
./scripts/start_vllm_docker.sh
```

---

### 📡 选项 4: 远程 vLLM 服务器

如果已有远程 vLLM 服务器，直接在 `.env` 中修改：
```
VLLM_BASE_URL=http://your-server:8000/v1
```

---

### 💻 选项 5: WSL2 本地安装

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

---

## 模型选择（6GB VRAM）

必须使用量化版本：
- ✅ `Qwen/Qwen2.5-7B-Instruct-AWQ` (推荐)
- ❌ 原版 FP16 模型无法加载

---

## ✅ 自动验证

运行完整的 6 项验证套件：

```bash
./scripts/verify_vllm.sh
```

将自动检查：
1. ✅ Docker 容器运行状态
2. ✅ 容器内 GPU 可访问性
3. ✅ 健康端点响应
4. ✅ 已加载模型名称正确
5. ✅ Chat completion 接口可用
6. ✅ 镜像 ID 校验

---

## 🔍 手动验证

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

## 🛡️ 可复现性保障

本项目通过以下机制确保环境一致性：

1. **基础镜像 Digest 锁定**
   - vLLM 镜像: `sha256:b8374cee0a1acaec8b64525ff77560f30443f67bd0fc1956a3529504a89f823b`
   - CUDA 镜像: `sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6`

2. **依赖 Hash 锁定**
   - `docker/requirements-vllm.txt` 包含所有依赖的 SHA256 哈希
   - 使用 `--require-hashes` 强制验证

3. **GitHub Actions 构建证明**
   - 每次构建自动生成 provenance attestation
   - 可验证构建来源和完整性

---

## 驱动更新验证

更新驱动后运行：
```bash
nvidia-smi
# 应该显示 CUDA Version: 12.4 或更高

# 验证 Docker GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 📝 常用命令

```bash
# 查看日志
docker compose logs -f vllm

# 停止服务
./scripts/stop_vllm_docker.sh

# 仅拉取镜像（预先准备）
./scripts/pull_vllm_image.sh

# 切换构建模式
# 编辑 .env 中的 VLLM_MODE=build 或 full
```

---

## 故障排除

### 容器启动失败
```bash
# 查看详细日志
docker compose logs --tail=50

# 检查 GPU 可用性
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 模型加载失败
- 检查 VRAM 充足（需要 ~5GB）
- 确认 `VLLM_GPU_MEMORY` 设置（建议 0.85-0.90）
- 检查 HuggingFace 网络连接

### 端口冲突
修改 `.env` 中的 `VLLM_PORT` 为其他端口
