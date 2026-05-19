"""
从真实的GitHub Actions workflow YAML文件内容中，
生成多样化的Multi-step Planning种子
"""
import json
import uuid
import re
from pathlib import Path
from collections import Counter


def extract_workflow_features(yaml_content: str):
    """从真实YAML中提取workflow特征"""
    features = {
        'jobs': len(re.findall(r'^\s{2,4}[a-zA-Z_]+:', yaml_content, re.MULTILINE)),
        'steps': yaml_content.count('- uses:') + yaml_content.count('- name:'),
        'has_secrets': 'secrets' in yaml_content.lower(),
        'has_matrix': 'matrix:' in yaml_content,
        'has_needs': 'needs:' in yaml_content,
        'has_if': 'if:' in yaml_content,
        'has_concurrency': 'concurrency:' in yaml_content,
        'has_environment': 'environment:' in yaml_content,
        'deploy': any(k in yaml_content.lower() for k in ['deploy', 'publish', 'release']),
        'test': any(k in yaml_content.lower() for k in ['test', 'unit', 'integration', 'lint']),
        'build': any(k in yaml_content.lower() for k in ['build', 'compile', 'package']),
        'has_approval': 'environment' in yaml_content and ('production' in yaml_content.lower() or 'prod' in yaml_content.lower()),
    }
    return features


def generate_prompt_from_features(workflow_name: str, features: dict):
    """基于真实workflow特征，生成多样化的prompt"""

    # ====== 不同的prompt前缀模板（10种不同开头） ======
    prefix_templates = [
        "Design a multi-step execution workflow for {name}. The workflow should include:",
        "Create a comprehensive automation playbook for {name} with:",
        "You are a DevOps engineer. Implement a production-grade workflow for {name}:",
        "Design a resilient {name} pipeline with failure handling and rollback:",
        "Build an end-to-end {name} process with explicit dependencies between steps:",
        "Architect a production {name} workflow that includes:",
        "Outline step-by-step the {name} automation with proper error handling:",
        "Design for {name}: a multi-phase process with validation gates between phases:",
        "Create an automated {name} procedure that handles edge cases and failures:",
        "Plan the {name} workflow with clear prerequisites, execution steps, and post-validation:",
    ]

    # ====== 不同的要求/约束组合（8种不同组合）======
    requirement_sets = [
        [
            "Dependency graph showing which steps depend on others",
            "Error handling strategy for each step with specific failure modes",
            "Rollback and cleanup procedures for partial failures",
            "Idempotency guarantees for safe re-runs",
            "Progress tracking and completion criteria",
        ],
        [
            "Sequential and parallel step execution design",
            "Retry policies with backoff for flaky operations",
            "Timeout and watchdog mechanisms for hung steps",
            "Resource cleanup on both success and failure paths",
            "Monitoring and alerting integration points",
        ],
        [
            "Pre-flight validation checks before any changes",
            "Gradual rollout with canary or phased execution",
            "Health checks and success criteria for each phase",
            "Manual approval gates for high-risk operations",
            "Audit logging and compliance tracking requirements",
        ],
        [
            "Input validation and schema enforcement",
            "Branching logic based on conditional outcomes",
            "Rate limiting and throttling for external API calls",
            "Circuit breaker pattern for dependent services",
            "Fallback and graceful degradation strategies",
        ],
        [
            "Data persistence and state management between steps",
            "Transaction boundaries with ACID guarantees",
            "Compensation actions for each step in reverse order",
            "Idempotency keys for exactly-once delivery",
            "Dead letter queue for failed items handling",
        ],
        [
            "Performance objectives and SLA requirements",
            "Cost optimization steps for resource usage",
            "Scaling considerations based on input size",
            "Caching strategy to avoid redundant computation",
            "Cleanup of temporary resources on completion",
        ],
        [
            "Security scanning and vulnerability checks",
            "Permission model and least-privilege access design",
            "Secret management and credential rotation handling",
            "Audit trail generation for compliance requirements",
            "Threat modeling and attack surface reduction",
        ],
        [
            "Testing strategy for the workflow itself",
            "Dry-run mode that simulates without side effects",
            "Shadow traffic testing for production changes",
            "Metrics collection and observability design",
            "Alert thresholds and on-call escalation paths",
        ],
    ]

    # ====== 根据features选择合适的难度和质量分======
    complexity_score = features['steps'] + features['jobs'] * 2 + sum([
        features[k] for k in ['has_matrix', 'has_needs', 'has_secrets',
                             'has_approval', 'has_concurrency', 'has_if']
    ])

    if complexity_score <= 6:
        difficulty = 'easy'
        quality = 0.78
    elif complexity_score <= 12:
        difficulty = 'medium'
        quality = 0.84
    else:
        difficulty = 'hard'
        quality = 0.90

    # 选择模板和要求集合
    import random
    prefix = random.choice(prefix_templates)
    reqs = random.choice(requirement_sets)

    # 构建prompt
    prompt = prefix.format(name=workflow_name) + "\n\n"
    for i, req in enumerate(reqs, 1):
        prompt += f"{i}. {req}\n"

    # 根据features添加领域特定要求
    if features['deploy']:
        prompt += "\nAdditional deployment-specific requirements:\n"
        prompt += "- Canary or blue-green deployment strategy\n"
        prompt += "- Health checks and traffic shifting logic\n"
        prompt += "- Instant rollback trigger conditions\n"
        quality += 0.02

    if features['test']:
        prompt += "\nAdditional testing-specific requirements:\n"
        prompt += "- Flaky test detection and automatic retry\n"
        prompt += "- Test parallelization with resource isolation\n"
        prompt += "- Failure grouping and reporting strategy\n"
        quality += 0.02

    if features['has_matrix']:
        prompt += "\nMatrix-specific requirements:\n"
        prompt += "- Matrix expansion and combination logic\n"
        prompt += "- Fail-fast vs continue-on-error configuration\n"
        prompt += "- Fast-fail and cancellation propagation\n"
        quality += 0.01

    if features['has_needs']:
        prompt += "\nDependency management requirements:\n"
        prompt += "- Directed acyclic graph validation\n"
        prompt += "- Dependency failure propagation handling\n"
        prompt += "- Optional vs required dependency distinction\n"
        quality += 0.01

    if features['has_approval']:
        prompt += "\nGovernance and compliance requirements:\n"
        prompt += "- Manual approval gates with timeouts\n"
        prompt += "- Multi-person review for production changes\n"
        prompt += "- Audit log generation for compliance\n"
        quality += 0.02

    return prompt.strip(), difficulty, min(0.95, quality)


def main():
    extract_dir = Path("data/processed/extracted")

    # 查找所有workflow文件
    workflow_files = []
    for pattern in ['starter-workflows-main/**/*.yml',
                    'starter-workflows-main/**/*.yaml',
                    'ansible-examples-master/**/*.yml',
                    'ansible-examples-master/**/*.yaml']:
        workflow_files.extend(list(extract_dir.glob(pattern)))

    print(f"Found {len(workflow_files)} workflow files")

    # 处理每个文件
    all_seeds = []
    seen_prompts = set()

    for wf_file in workflow_files:
        try:
            content = wf_file.read_text(encoding='utf-8', errors='ignore')

            # 跳过太小的文件
            if len(content) < 200:
                continue

            features = extract_workflow_features(content)

            # 从文件名提取合理的workflow名称
            name = wf_file.stem.replace('_', ' ').replace('-', ' ').title()

            # 生成多样化的prompt
            for variant in range(2):  # 每个文件生成2个不同的变体
                prompt, difficulty, quality = generate_prompt_from_features(name, features)

                # 去重：用前120字符作为指纹
                fp = prompt[:120].lower()
                if fp in seen_prompts:
                    continue
                seen_prompts.add(fp)

                category = 'deployment' if features['deploy'] else \
                          'testing' if features['test'] else \
                          'automation'

                seed = {
                    'id': str(uuid.uuid4()),
                    'domain': 'multi_step_planning',
                    'difficulty': difficulty,
                    'prompt': prompt,
                    'test_cases': [{
                        'input': {'workflow': name},
                        'expected_output': {'plan': True, 'dependencies': True},
                        'is_public': True
                    }],
                    'validator_code': 'def validate(sol): return isinstance(sol, dict) and len(sol.get(\"steps\", [])) >= 5',
                    'source': 'crawled_real_workflow',
                    'quality_score': quality,
                    'tags': ['real-workflow', category, 'complexity:' + str(features['steps']) + 'steps']
                }
                all_seeds.append(seed)

        except Exception as e:
            continue

    print(f"\nGenerated {len(all_seeds)} unique multi-step planning seeds")

    # 按质量排序，精选50个
    all_seeds.sort(key=lambda x: x['quality_score'], reverse=True)
    selected = all_seeds[:50]

    # 统计
    print(f"\nQuality distribution:")
    diff_count = Counter(s['difficulty'] for s in selected)
    print(f"  Difficulty: {dict(diff_count)}")
    print(f"  Avg quality: {sum(s['quality_score'] for s in selected)/len(selected):.2f}")

    # 抽样展示
    print(f"\nSample of 5 diverse prompts:")
    print("=" * 70)
    for i, s in enumerate(selected[:5]):
        print(f"\n{i+1}. [{s['difficulty']}, q={s['quality_score']:.2f}]")
        print(f"   {s['prompt'][:120]}...")

    # 保存
    output_file = Path("data/processed/ms_50_diverse_real_seeds.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'prompts': selected}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(selected)} diverse multi-step planning seeds to {output_file}")


if __name__ == "__main__":
    main()
