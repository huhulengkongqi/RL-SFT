# Agentic RL SFT Data Synthesis Pipeline

用于 Agentic RL 冷启动的 SFT 数据合成管线。当前项目的核心目标不是直接训练模型，而是把真实/合成任务转化为可用于训练 Agent 的高质量轨迹数据：

```text
raw data → seed prompts → evolved tasks → test/reference enrichment → AgentLoop trajectories → SFT JSON
```

## 当前端到端流程

### 1. 原始数据采集与存放

原始爬取/收集数据放在 `data/raw/`：

```text
data/raw/
├── Posts.xml                         # StackOverflow 等问答/调试类原始数据
├── train-00000-of-00001.parquet      # 数学/推理类原始数据
├── OpenAPI-Specification-main.zip    # API / OpenAPI 规范数据
├── fastapi_examples.zip              # API 编排示例
├── sdk_boto3.zip                     # SDK/API 使用示例
├── sdk_google.zip
├── sdk_requests.zip
├── ansible-examples-master.zip       # 多步规划 / DevOps 示例
└── starter-workflows-main.zip        # CI/CD 工作流示例
```

这些 raw 数据用于构造四个主要 domain 的初始任务来源：

| Domain | 来源示例 | 目标任务类型 |
|---|---|---|
| `code_debug` | StackOverflow、真实 Python bug、SDK 示例 | 调试、修复、解释 bug |
| `math_reasoning` | 数学推理数据集 | 逐步推理、数值答案 |
| `api_orchestration` | OpenAPI、SDK、FastAPI 示例 | API 调用顺序、错误处理、鉴权流程 |
| `multi_step_planning` | Ansible、CI/CD workflow、部署样例 | 多阶段计划、依赖、风险控制 |

### 2. 用 LLM / 规则整理成 Seed Prompt

Seed Prompt 是结构化任务的最小单元，通常包含：

```json
{
  "id": "uuid",
  "domain": "code_debug",
  "difficulty": "medium",
  "prompt": "任务描述",
  "test_cases": [...],
  "validator_code": "...",
  "source": "crawled | llm_generated | human_curated",
  "quality_score": 0.8,
  "tags": [...]
}
```

已有/相关输出文件包括：

```text
data/final_seed_pool_181_real.json
data/code_debug_augmented.json
data/code_debug_augmented_final.json
```

可用脚本：

```bash
uv run python scripts/generate_seed_prompts.py
```

说明：`generate_seed_prompts.py` 当前主要用于构造 4 个 domain 的基础 seed pool；后续可接入 `data/raw/` 的解析器，把爬取数据和 LLM 清洗结果统一转成 SeedPrompt 结构。

### 3. Evol-Instruct 进化任务

Seed Prompt 经过 Evol-Instruct 扩展，生成更复杂、更强约束、更接近真实 Agent 训练场景的任务。

核心脚本：

```bash
# 本地/服务端 vLLM
uv run python scripts/run_evolution.py \
  --seed-file data/final_seed_pool_181_real.json \
  --generations 4 \
  --evolutions-per-seed 3 \
  --output-dir data/evolved

# 使用火山 API（OpenAI-compatible coding/v3）
set ANTHROPIC_AUTH_TOKEN=your_api_key
uv run python scripts/run_evolution.py \
  --use-claude \
  --claude-model ark-code-latest \
  --min-sleep 12 \
  --max-sleep 18
```

进化策略在 `src/agent_sft/evol_instruct/` 中实现，覆盖：

- 加深约束
- 增加复杂输入
- Chain-of-Thought 结构化要求
- 广度变异
- in-context learning 风格变体
- 多轮 generation 递进演化

进化结果可用：

```bash
uv run python scripts/assess_evolution.py data/evolved/final_evolved.json
```

### 4. 给进化后的 Task 补充测试例与参考答案修正

进化后的 task 需要进一步补全/校正：

1. `test_cases`
2. `expected_output.final_answer`
3. `reference_solution` / `final_answer`
4. 任务验证方式所需的结构字段

当前已形成的数据文件示例：

```text
data/claude_evolved_4gen/final_evolved_v1.0_complete.json
data/final_evolved_v1.0_complete_math_fixed_20260522_154556.json
```

数学参考答案批量修正脚本：

```bash
set ANTHROPIC_AUTH_TOKEN=your_api_key
uv run python scripts/fix_math_references_with_llm.py \
  --input data/claude_evolved_4gen/final_evolved_v1.0_complete.json \
  --sleep-min 10 \
  --sleep-max 18
```

该脚本会：

- 遍历 `math_reasoning` tasks
- 调用火山 LLM 独立解题
- 判断原参考答案是否正确
- 修正错误参考答案
- 尽量把答案标准化为单独数字字符串
- 输出新的完整 dataset JSON
- 输出 audit report 和 checkpoint

输出示例：

```text
data/reference_checks/final_evolved_v1.0_complete_math_fixed_YYYYMMDD_HHMMSS.json
data/reference_checks/math_reference_fix_report_YYYYMMDD_HHMMSS.json
data/reference_checks/math_reference_fix_progress_YYYYMMDD_HHMMSS.jsonl
```

### 5. AgentLoop 轨迹生成

完成 task 后，用强模型作为 teacher agent，放入真实环境中交互，录制完整轨迹。

核心链路：

```text
Task
  ↓
AgentLoop
  ↓
LLM 生成 Action
  ↓
Environment 执行工具 / 验证答案
  ↓
Observation
  ↓
TrajectoryRecorder 录制
  ↓
raw trajectory + SFT format JSON
```

主脚本：

```bash
set ANTHROPIC_AUTH_TOKEN=your_api_key
uv run python scripts/generate_single_trajectory_real_env.py \
  --domain code_debug \
  --max-steps 20 \
  --sleep-min 10 \
  --sleep-max 18
```

支持 domain：

```text
math_reasoning
code_debug
api_orchestration
multi_step_planning
```

输出：

```text
data/sft_trajectories/realenv_<task_id>_<timestamp>_raw.json
data/sft_trajectories/realenv_<task_id>_<timestamp>_sft.json
```

### 6. 当前验证策略

真实环境入口：

```python
from infra.environment.environment import Environment
```

它会使用 `SandboxPool` / Docker 执行工具调用和代码验证。

#### `math_reasoning`

- 工具：`eval`
- 验证：`MATH_EQUATION`
- 支持：
  - 精确字符串匹配
  - 数值容差匹配
  - 小数位不同的四舍五入/数值匹配

示例：

```text
88 == 88.0
15.4 == 15.400000
0.154 == 0.1540
```

#### `code_debug`

先尝试严格代码验证：

1. 从 final answer 中提取 Python 代码块
2. 自动推断函数名
3. 若是脚本式代码，尝试封装成 `solution(**kwargs)`
4. 使用 Docker sandbox 执行测试

如果数据集 test case 是 StackOverflow 调试报告结构，例如：

```json
{
  "root_cause": true,
  "fixed_code": true,
  "explanation": true
}
```

则走混合验证：

```text
format validation
+ LLM-as-Judge
+ 历史成功 exec 工具调用证据
```

也就是说，不能只因为回答里有字段就成功，还必须存在历史工具执行成功的相关代码证据。

#### `api_orchestration`

- 当前以格式验证 + LLM-as-Judge 为主
- 检查 API 调用顺序、鉴权、错误处理、payload/response 处理完整性

#### `multi_step_planning`

- 当前以格式验证 + LLM-as-Judge 为主
- 检查步骤顺序、依赖、风险、资源、时间线完整性

## 主要模块结构

```text
src/
├── agent_sft/
│   ├── task_generator/          # SeedPrompt / Task 结构、生成、校验
│   ├── evol_instruct/           # Evol-Instruct 策略与多代进化
│   ├── quality_filter/          # 去重、质量判别、统计
│   ├── trajectory_sampler/      # AgentLoop / AgentState / TrajectoryRecorder
│   └── dataset_builder/         # SFT 数据格式化（待扩展）
└── infra/
    ├── vllm_client/             # OpenAI-compatible 客户端；也用于火山 coding/v3
    ├── anthropic_client/        # Anthropic native protocol 客户端
    ├── environment/             # Agent 交互环境、AnswerVerifier、SandboxPool
    └── sandbox/                 # Docker sandbox 执行管理
```

## 环境要求

- Python 3.11+
- `uv`
- Docker（用于真实 Environment / sandbox）
- 可选：NVIDIA GPU + vLLM Docker
- 火山 API Key（使用火山模型时）

设置火山 API Key：

```powershell
$env:ANTHROPIC_AUTH_TOKEN="your_api_key"
```

或者 cmd：

```cmd
set ANTHROPIC_AUTH_TOKEN=your_api_key
```

## 常用命令

### 安装依赖

```bash
uv sync
```

### 运行测试

```bash
uv run pytest
uv run pytest tests/test_agent_loop.py -q
uv run pytest tests/infra/environment/test_environment.py -v
```

### Lint / Format / Type Check

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run mypy src/
```

### vLLM Docker

```bash
./scripts/start_vllm_docker.sh
./scripts/verify_vllm.sh
./scripts/stop_vllm_docker.sh
```

## 当前保留脚本

| 脚本 | 用途 |
|---|---|
| `scripts/run_evolution.py` | SeedPrompt 多代进化 |
| `scripts/assess_evolution.py` | 进化结果统计 |
| `scripts/generate_seed_prompts.py` | 生成基础 seed pool |
| `scripts/fix_math_references_with_llm.py` | 批量修正数学参考答案 |
| `scripts/generate_single_trajectory_real_env.py` | 使用真实 Environment 生成单条轨迹 |
| `scripts/quality_assessment.py` | seed/evolved 数据质量评估 |
| `scripts/demo_task_generator.py` | task generation demo |
| `scripts/demo_task_validation.py` | sandbox validation demo |

## 数据产物路径

| 路径 | 含义 |
|---|---|
| `data/raw/` | 原始爬取/收集数据 |
| `data/final_seed_pool_181_real.json` | 真实整理后的 seed pool 示例 |
| `data/evolved/` | Evol-Instruct 输出目录 |
| `data/claude_evolved_4gen/final_evolved_v1.0_complete.json` | 4 代进化后的完整 task 数据 |
| `data/reference_checks/` | LLM 参考答案审核/修正报告 |
| `data/sft_trajectories/` | AgentLoop 生成的 raw/SFT 轨迹 |

## 许可证

MIT
