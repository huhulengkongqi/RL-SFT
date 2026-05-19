"""
处理真实数据集，解压zip文件，提取API Orchestration和Multi-step Planning种子prompt

数据源:
- GitHub Actions workflows → multi-step_planning
- Ansible playbooks → multi-step_planning
- OpenAPI specs → api_orchestration
- SDK examples (boto3, google, requests) → api_orchestration
- FastAPI examples → api_orchestration
"""
import json
import re
import uuid
import zipfile
from pathlib import Path
from collections import Counter


def extract_zip(zip_path: Path, extract_to: Path):
    """解压zip文件"""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


def extract_github_workflows_scenarios(base_path: Path) -> list:
    """从GitHub Actions workflows中提取多步规划场景"""
    scenarios = []

    workflow_files = list(base_path.rglob("*.yml")) + list(base_path.rglob("*.yaml"))

    print(f"\nProcessing GitHub Actions workflows: Found {len(workflow_files)} files")

    for wf_file in workflow_files[:80:  # 处理前80个
        try:
            with open(wf_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 提取workflow名称和步骤
            name_match = re.search(r'name:\s*([^\n]+)', content)
            workflow_name = name_match.group(1).strip() if name_match else wf_file.stem

            # 统计步骤数（- name: pattern）
            step_count = content.count("- name:") + content.count("- uses:")

            if step_count >= 3:  # 只保留有3步以上的workflow
                if any(k in wf_file.name.lower() for k in ["deploy", "publish", "release"]):
                    category = "Deployment"
                elif any(k in wf_file.name.lower() for k in ["test", "lint", "check", "test"]):
                    category = "Testing"
                elif any(k in wf_file.name.lower() for k in ["security", "security", "automation"]):
                    category = "Security/Automation"
                else:
                    category = "CI/CD"

                scenarios.append({
                    "name": workflow_name,
                    "steps_count": step_count,
                    "category": category,
                    "file": str(wf_file.relative_to(base_path)),
                })
        except Exception as e:
            pass

    print(f"Extracted {len(scenarios)} workflow scenarios")
    return scenarios


def extract_ansible_playbooks(base_path: Path) -> list:
    """从Ansible playbooks中提取场景"""
    scenarios = []
    playbook_files = list(base_path.rglob("*.yml")) + list(base_path.rglob("*.yaml"))

    print(f"\nProcessing Ansible playbooks: Found {len(playbook_files)} files")

    for pb_file in playbook_files[:100]:
        try:
            with open(pb_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            task_count = content.count("- name:") + content.count("- import_playbook:")

            if task_count >= 3:
                name = pb_file.stem.replace("_", " ").replace("-", " ").title()
                scenarios.append({
                    "name": name,
                    "steps_count": task_count,
                    "category": "Infrastructure Automation",
                    "file": str(pb_file.relative_to(base_path)),
                })
        except:
            pass

    print(f"Extracted {len(scenarios)} ansible playbook scenarios")
    return scenarios


def extract_api_scenarios_from_sdk(base_path: Path, sdk_name: str) -> list:
    """从SDK示例代码中提取API调用场景"""
    scenarios = []

    # 找README和示例
    readme_files = list(base_path.rglob("README*.md")) + list(base_path.rglob("README*.rst"))

    print(f"\nProcessing {sdk_name} SDK: Found {len(readme_files)} README files")

    for readme in readme_files:
        try:
            with open(readme, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 提取包含代码块和使用说明
            code_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)

            for block in code_blocks[:50]:  # 每个README取前50个代码块
                api_calls = block.count("client") + block.count(".get(") + block.count(".post(") + block.count(".put(") + block.count(".delete(") + block.count("session.")

                if api_calls >= 2:  # 至少2个API调用
                    scenarios.append({
                        "name": f"{sdk_name} API调用示例",
                        "api_count": api_calls,
                        "category": "API Orchestration",
                        "code_snippet": block[:200],
                        "file": str(readme.relative_to(base_path)),
                    })
        except:
            pass

    print(f"Extracted {len(scenarios)} {sdk_name} API scenarios")
    return scenarios


def extract_openapi_scenarios(base_path: Path) -> list:
    """从OpenAPI规范中提取场景"""
    scenarios = []
    spec_files = list(base_path.rglob("*.json")) + list(base_path.rglob("*.yaml")) + list(base_path.rglob("*.yml"))

    print(f"\nProcessing OpenAPI specs: Found {len(spec_files)} files")

    for spec_file in spec_files[:50]:
        try:
            with open(spec_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 数API endpoint数量
            endpoint_count = content.count('"get"') + content.count("post:") + content.count("put:") + content.count("delete:")

            if endpoint_count >= 3:
                name = spec_file.stem.replace("_", " ").replace("-", " ").title()
                scenarios.append({
                    "name": name,
                    "endpoint_count": endpoint_count,
                    "category": "REST API Design",
                    "file": str(spec_file.relative_to(base_path)),
                })
        except:
            pass

    print(f"Extracted {len(scenarios)} OpenAPI scenarios")
    return scenarios


def convert_to_seed_prompt(scenario: dict, domain: str, difficulty: str = "medium") -> dict:
    """把场景转换为标准种子prompt格式"""
    scenario_name = scenario.get("name", "Unnamed workflow")
    category = scenario.get("category", "General")

    if domain == "multi_step_planning":
        steps_count = scenario.get("steps_count", 5)

        prompt = f"""Create a detailed multi-step execution plan for: {scenario_name}

This is a real-world {category} workflow with approximately {steps_count} steps.

Provide:
1. Clear pre-requisite dependencies between steps
2. Detailed instructions for each step
3. Error handling and rollback plan
4. Success criteria for each step
5. Validation steps that can be parallelizable for critical failure recovery procedures"""

    else:  # api_orchestration
        api_count = scenario.get("api_count", scenario.get("endpoint_count", 3))
        steps_count = scenario.get("api_count", 3)

        prompt = f"""Design an API orchestration workflow for: {scenario_name}

This requires coordinating {api_count} API calls in {category} domain.

Provide:
1. Sequence diagram of the API call flow
2. Error handling strategy for each API call
3. Retry and fallbacks for each endpoint
4. Data transformation between steps
5. Transaction boundaries if applicable
6. Final success validation"""

    return {
        "id": str(uuid.uuid4()),
        "domain": domain,
        "difficulty": difficulty,
        "prompt": prompt,
        "test_cases": [{
            "input": {"scenario_name": scenario_name},
            "expected_output": {"plan": True, "steps": True},
            "is_public": True
        }],
        "validator_code": "def validate(sol): return isinstance(sol, dict) and 'steps' in sol and len(sol['steps']) >= 3",
        "source": "crawled",
        "quality_score": min(0.9, 0.75 + min(0.15, steps_count / 20)),
        "tags": ["real-world-scenario", domain, category.lower().replace(" ", "_")],
    }


def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Processing Real Datasets for API Orchestration & Multi-step Planning")
    print("=" * 70)

    # 1. 解压所有zip文件
    extract_dir = processed_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)

    zip_files = list(raw_dir.glob("*.zip"))
    print(f"\nFound {len(zip_files)} zip files to extract")

    for zip_file in zip_files:
        try:
            extract_zip(zip_file, extract_dir / zip_file.stem)
        except Exception as e:
            print(f"Error extracting {zip_file}: {e}")

    # 2. 提取所有场景，构建种子
    print("\n" + "=" * 70)
    print("Extracting scenarios from datasets...")
    print("=" * 70)

    # Multi-step Planning 来源
    ms_planning_scenarios = []

    # GitHub Actions workflows
    wf_dir = extract_dir / "starter-workflows-main"
    if wf_dir.exists():
        ms_planning_scenarios.extend([
            (s, "github_workflows") for s in extract_github_workflows_scenarios(wf_dir)
        ])

    # Ansible playbooks
    ansible_dir = extract_dir / "ansible-examples-master"
    if ansible_dir.exists():
        ms_planning_scenarios.extend([
            (s, "ansible") for s in extract_ansible_playbooks(ansible_dir)
        ])
            (s, "ansible") for s in extract_ansible_playbooks(ansible_dir)
        ])

    # API Orchestration 来源
    api_scenarios = []

    # boto3 SDK
    boto3_dir = extract_dir / "sdk_boto3"
    if boto3_dir.exists():
        api_scenarios.extend([
            (s, "boto3") for s in extract_api_scenarios_from_sdk(boto3_dir, "AWS Boto3")
        ])

    # Google API SDK
    google_dir = extract_dir / "sdk_google"
    if google_dir.exists():
        api_scenarios.extend([
            (s, "google") for s in extract_api_scenarios_from_sdk(google_dir, "Google Cloud")
        ])

    # Requests SDK
    requests_dir = extract_dir / "sdk_requests"
    if requests_dir.exists():
        api_scenarios.extend([
            (s, "requests") for s in extract_api_scenarios_from_sdk(requests_dir, "HTTP Requests")
        ])

    # FastAPI examples
    fastapi_dir = extract_dir / "fastapi_examples"
    if fastapi_dir.exists():
        api_scenarios.extend([
            (s, "fastapi") for s in extract_api_scenarios_from_sdk(fastapi_dir, "FastAPI Service")
        ])

    # OpenAPI specs
    openapi_dir = extract_dir / "OpenAPI-Specification-main"
    if openapi_dir.exists():
        api_scenarios.extend([
            (s, "openapi") for s in extract_openapi_scenarios(openapi_dir)
        ])

    # 3. 转换为种子prompt格式
    print("\n" + "=" * 70)
    print("Converting to seed prompt format...")
    print("=" * 70)

    ms_seeds = []
    for scenario, source in ms_planning_scenarios:
        seed = convert_to_seed_prompt(scenario, "multi_step_planning")
        seed["tags"].append(source)
        ms_seeds.append(seed)

    api_seeds = []
    for scenario, source in api_scenarios:
        seed = convert_to_seed_prompt(scenario, "api_orchestration")
        seed["tags"].append(source)
        api_seeds.append(seed)

    # 4. 质量过滤和去重
    print(f"\nMulti-step Planning: {len(ms_seeds)} raw seeds")
    print(f"API Orchestration: {len(api_seeds)} raw seeds")

    # 按质量分排序
    ms_seeds.sort(key=lambda x: x["quality_score"], reverse=True)
    api_seeds.sort(key=lambda x: x["quality_score"], reverse=True)
    seen = set()
    ms_final = []
    for seed in ms_seeds:
        fp = seed["prompt"][:80].lower()
        if fp not in seen:
            seen.add(fp)
            ms_final.append(seed)

    seen = set()
    api_final = []
    for seed in api_seeds:
        fp = seed["prompt"][:80].lower()
        if fp not in seen:
            seen.add(fp)
            api_final.append(seed)

    # 每个domain精选50个
    ms_final = ms_final[:50]
    api_final = api_final[:50]

    # 5. 难度校准难度
    print(f"\nFinal selection:")
    print(f"Multi-step Planning: {len(ms_final)} seeds")
    print(f"API Orchestration: {len(api_final)} seeds")

    # 保存
    # 难度分布校准
    def calibrate_difficulty(seeds):
        for seed in seeds:
            steps_count = seed["prompt"].count("\n")
            if steps_count < 10:
                seed["difficulty"] = "easy"
            elif steps_count < 20:
                seed["difficulty"] = "medium"
            else:
                seed["difficulty"] = "hard"

    calibrate_difficulty(ms_final)
    calibrate_difficulty(api_final)

    # 6. 保存
    output = {
        "prompts": ms_final + api_final
    }

    output_file = processed_dir / "api_and_ms_100_seeds.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(output['prompts'])} seeds to {output_file}")

    # 7. 统计
    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)

    all_seeds = ms_final + api_final

    print(f"\nTotal seeds: {len(all_seeds)}")
    print(f"By domain: {dict(Counter(s['domain'] for s in all_seeds))}")
    print(f"By source: {dict(Counter(s['source'] for s in all_seeds))}")
    print(f"By difficulty: {dict(Counter(s['difficulty'] for s in all_seeds))}")
    print(f"Average quality: {sum(s['quality_score'] for s in all_seeds) / len(all_seeds):.2f}")

    # 抽样展示
    print("\n" + "=" * 70)
    print("Sample seeds:")
    print("=" * 70)
    for i, seed in enumerate(all_seeds[:3]):
        print(f"\n{i+1}. [{seed['domain']}] [{seed['difficulty']}] [{seed['quality_score']:.2f}]")
        print(f"   {seed['prompt'][:80]}...")

    return output


if __name__ == "__main__":
    main()
