# Agentic RL SFT Data Synthesis Pipeline

用于 Agentic RL 冷启动的 SFT 数据合成管线，涵盖任务生成、轨迹采样（Agent Harness）、质量过滤和数据集构建。

## 功能模块

- `task_generator` - 多样化任务生成（编码、推理、工具使用等）
- `trajectory_sampler` - Agent 轨迹采样，带沙箱执行环境
- `quality_filter` - 基于 LLM 的轨迹质量过滤与打分
- `dataset_builder` - 数据集格式化（ShareGPT、OpenAI 格式等）

## 环境要求

- Python 3.11+
- uv 包管理器
- **NVIDIA 驱动 >= 550.x** (CUDA 12.4+)
- **Docker + NVIDIA Container Toolkit** (推荐部署方式)
- (可选) E2B / Docker 代码沙箱

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件配置 HF_TOKEN 和其他参数
```

### 2. 启动 vLLM 服务（推荐 Docker）

```bash
./scripts/start_vllm_docker.sh
```

### 3. 验证服务

```bash
./scripts/verify_vllm.sh
```

### 4. 安装 Python 依赖（用于管道运行）

```bash
uv sync
```

### 备选：本地 Python 启动 vLLM

```bash
uv run python -m src.infra.vllm_client.server Qwen/Qwen2.5-7B-Instruct --port 8000
uv run python -m src.infra.vllm_client.client --base-url http://localhost:8000/v1
```

## 部署方式

项目提供 **3 种 vLLM Docker 部署模式**，通过 `docker-compose.yml` profiles 切换：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `pull` | 从 GHCR 拉取预构建镜像 | 快速启动、生产环境 |
| `build` | 基于官方 vLLM 镜像构建 | 自定义 vLLM 版本 |
| `full` | 从 CUDA 基础镜像完整构建 | 最大可控性 |

完整部署文档请参考：[docs/VLLM_DEPLOYMENT.md](docs/VLLM_DEPLOYMENT.md)

### 可用脚本

| 脚本 | 说明 |
|------|------|
| `./scripts/start_vllm_docker.sh` | 启动 vLLM Docker 服务 |
| `./scripts/stop_vllm_docker.sh` | 停止 vLLM Docker 服务 |
| `./scripts/verify_vllm.sh` | 6 点验证套件（健康检查、模型加载、推理测试等） |
| `./scripts/pull_vllm_image.sh` | 预先从 GHCR 拉取镜像 |

### 可重现性保障

- **SHA256 锁定基础镜像** - Dockerfile 中所有基础镜像通过 digest 固定
- **Hash-pinned Python 依赖** - `requirements-vllm.txt` 包含完整 hash
- **GHCR Provenance Attestation** - GitHub Actions 自动发布并附带构建证明

## 项目结构

```
src/
├── agent_sft/
│   ├── task_generator/      # 任务生成模块
│   ├── trajectory_sampler/  # 轨迹采样（Agent Harness）
│   ├── quality_filter/      # 质量过滤
│   └── dataset_builder/     # 数据集构建
└── infra/
    ├── vllm_client/         # vLLM 推理客户端
    └── sandbox/             # 沙箱封装（E2B/Docker）
```

## 故障排查

常见问题和解决方案请参考：[docs/VLLM_DEPLOYMENT.md](docs/VLLM_DEPLOYMENT.md) 中的 **Troubleshooting** 章节

## 开发工具

```bash
# 代码格式化
uv run ruff format src/ tests/

# 代码检查
uv run ruff check src/ tests/

# 类型检查
uv run mypy src/

# 运行测试
uv run pytest
```

## License

MIT
