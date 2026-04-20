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
- vLLM 推理服务（本地或远程）
- (可选) E2B / Docker 代码沙箱

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件配置 API keys 和 vLLM 地址
```

### 3. 启动 vLLM 服务（可选）

```bash
# 使用 vLLM 本地服务脚本
uv run python -m src.infra.vllm_client.server Qwen/Qwen2.5-7B-Instruct --port 8000
```

### 4. 验证 vLLM 客户端

```bash
uv run python -m src.infra.vllm_client.client --base-url http://localhost:8000/v1
```

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
