"""
独立质量评估程序
用于对种子池进行完整的质量评估、过滤、打分

使用方法:
    python scripts/quality_assessment.py --input data/seed_prompts.json --output data/filtered_seeds.json
"""
import json
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple


class QualityAssessmentPipeline:
    """完整的质量评估流水线"""

    def __init__(self, config: Dict = None):
        self.config = config or self.default_config()
        self.stats = {}

    @staticmethod
    def default_config() -> Dict:
        return {
            "min_prompt_length": 100,
            "min_quality_score": 0.70,
            "similarity_threshold": 0.70,
            "difficulty_targets": {"easy": 20, "medium": 20, "hard": 10},
            "min_code_length": 20,
            "require_code_for_code_debug": True,
        }

    def basic_check(self, seed: Dict) -> Tuple[bool, List[Tuple[bool, str]]]:
        """第一层：基础完整性检查"""
        checks = [
            (len(seed.get("prompt", "")) >= self.config["min_prompt_length"],
             f"prompt_length >= {self.config['min_prompt_length']} chars"),

            ("id" in seed and seed["id"], "has_valid_id"),
            ("domain" in seed and seed["domain"], "has_domain"),
            ("difficulty" in seed and seed["difficulty"], "has_difficulty"),
            ("quality_score" in seed, "has_quality_score"),
            ("source" in seed, "has_source"),
            ("test_cases" in seed and len(seed["test_cases"]) > 0, "has_test_cases"),
            ("validator_code" in seed and len(seed["validator_code"]) > 0, "has_validator"),
        ]

        # Code debug 特有检查：必须包含代码
        if self.config["require_code_for_code_debug"] and seed.get("domain") == "code_debug":
            has_code = (
                "```" in seed["prompt"]
                or "code" in seed["prompt"].lower()
                or "Code Context" in seed["prompt"]
            )
            checks.append((has_code, "has_code_block_for_code_debug"))

        all_passed = all(ok for ok, _ in checks)
        return all_passed, checks

    def heuristic_quality_score(self, seed: Dict) -> float:
        """第二层：启发式质量打分 0.70 ~ 0.95"""
        score = 0.70
        prompt = seed["prompt"]
        domain = seed.get("domain", "")

        # 1. 长度加分（详细的问题质量更高）
        prompt_len = len(prompt)
        if prompt_len >= 500:
            score += 0.08
        elif prompt_len >= 300:
            score += 0.05
        elif prompt_len >= 150:
            score += 0.03

        # 2. 代码相关加分（code_debug专用）
        if domain == "code_debug":
            code_chars = self._count_code_chars(prompt)
            if code_chars >= 200:
                score += 0.08
            elif code_chars >= 100:
                score += 0.05
            elif code_chars >= 50:
                score += 0.03

        # 3. 数学题步骤完整性加分
        if domain == "math_reasoning":
            step_markers = ["step", "therefore", "because", "thus", "first", "second", "then"]
            has_steps = any(m in prompt.lower() for m in step_markers)
            if has_steps:
                score += 0.05
            if "Solve" in prompt or "Calculate" in prompt:
                score += 0.02

        # 4. 明确的输出要求
        has_clear_requirements = any(kw in prompt.lower() for kw in [
            "provide", "explain", "show", "give", "return", "output",
            "root cause", "fixed code", "explanation", "steps"
        ])
        if has_clear_requirements:
            score += 0.04

        # 5. 结构化输出要求
        has_structure = any(kw in prompt for kw in ["1.", "2.", "3.", "4.", "---", "**"])
        if has_structure:
            score += 0.02

        # 6. 真实来源加分
        if seed.get("source") == "crawled":
            score += 0.02

        return min(0.95, round(score, 2))

    @staticmethod
    def _count_code_chars(prompt: str) -> int:
        """统计prompt中的代码字符数"""
        import re
        code_blocks = re.findall(r"```.*?```", prompt, re.DOTALL)
        return sum(len(block) for block in code_blocks)

    def classify_difficulty(self, seed: Dict) -> str:
        """第三层：难度分级校准"""
        # 已有difficulty的做校验和调整
        difficulty = seed.get("difficulty", "medium")
        prompt_len = len(seed["prompt"])
        code_chars = self._count_code_chars(seed["prompt"])

        # 根据实际内容校准难度
        if difficulty == "easy" and (prompt_len > 400 or code_chars > 150):
            return "medium"
        if difficulty == "hard" and prompt_len < 200 and code_chars < 50:
            return "medium"

        return difficulty

    @staticmethod
    def deduplicate(seeds: List[Dict], threshold: float = None) -> List[Dict]:
        """第四层：基于prompt指纹去重"""
        if threshold is None:
            threshold = self.config["similarity_threshold"]

        kept = []
        seen_fingerprints = set()

        # 按质量分降序处理，高质量优先保留
        sorted_seeds = sorted(seeds, key=lambda s: s.get("quality_score", 0.7), reverse=True)

        for seed in sorted_seeds:
            prompt = seed["prompt"]
            # 指纹：前100字符（小写）
            fingerprint = prompt[:100].lower()

            is_duplicate = False
            for existing_fp in seen_fingerprints:
                set_a = set(fingerprint)
                set_b = set(existing_fp)
                jaccard = len(set_a & set_b) / (len(set_a | set_b) + 1e-8)
                if jaccard > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(seed)
                seen_fingerprints.add(fingerprint)

        return kept

    def select_balanced(self, seeds: List[Dict], domain_targets: Dict[str, int] = None) -> List[Dict]:
        """第五层：均衡抽样，确保domain和难度分布合理"""
        if domain_targets is None:
            domain_targets = {"code_debug": 50, "math_reasoning": 50}

        selected = []

        for domain, target_count in domain_targets.items():
            domain_seeds = [s for s in seeds if s["domain"] == domain]

            # 按难度分层
            by_difficulty = {
                "easy": sorted([s for s in domain_seeds if s["difficulty"] == "easy"],
                              key=lambda x: -x["quality_score"]),
                "medium": sorted([s for s in domain_seeds if s["difficulty"] == "medium"],
                                key=lambda x: -x["quality_score"]),
                "hard": sorted([s for s in domain_seeds if s["difficulty"] == "hard"],
                              key=lambda x: -x["quality_score"]),
            }

            targets = self.config["difficulty_targets"]
            selected_domain = []

            for diff, target in targets.items():
                available = by_difficulty[diff]
                take = min(target, len(available))
                selected_domain.extend(available[:take])

            # 如果不够，用最高质量的其他难度补充
            remaining = target_count - len(selected_domain)
            if remaining > 0:
                all_sorted = sorted(domain_seeds, key=lambda x: -x["quality_score"])
                for s in all_sorted:
                    if s not in selected_domain and remaining > 0:
                        selected_domain.append(s)
                        remaining -= 1

            selected.extend(selected_domain[:target_count])

        return selected

    def run(self, seeds: List[Dict], return_details: bool = False) -> Dict:
        """运行完整的质量评估pipeline"""
        print("=" * 70)
        print("Quality Assessment Pipeline")
        print("=" * 70)
        print(f"\nInput seeds: {len(seeds)}")

        # Step 1: 基础检查过滤
        print("\n[Step 1] Basic integrity check...")
        passed_basic = []
        failed_basic = []
        for seed in seeds:
            ok, checks = self.basic_check(seed)
            if ok:
                passed_basic.append(seed)
            else:
                failed = [(name, ok) for ok, name in checks if not ok]
                failed_basic.append({"seed": seed["id"], "failed_checks": failed})

        print(f"  Passed: {len(passed_basic)}, Failed: {len(failed_basic)}")

        # Step 2: 启发式质量打分
        print("\n[Step 2] Heuristic quality scoring...")
        for seed in passed_basic:
            calculated_score = self.heuristic_quality_score(seed)
            # 如果已有分数，取较高的那个
            existing_score = seed.get("quality_score", 0.7)
            seed["quality_score"] = round(max(existing_score, calculated_score), 2)

        # Step 3: 难度校准
        print("\n[Step 3] Difficulty calibration...")
        for seed in passed_basic:
            seed["difficulty_original"] = seed.get("difficulty", "medium")
            seed["difficulty"] = self.classify_difficulty(seed)

        # Step 4: 去重
        print("\n[Step 4] Deduplication...")
        deduplicated = self.deduplicate(passed_basic)
        print(f"  After deduplication: {len(deduplicated)} (removed {len(passed_basic) - len(deduplicated)})")

        # Step 5: 质量阈值过滤
        print(f"\n[Step 5] Quality threshold filtering (>= {self.config['min_quality_score']})...")
        quality_filtered = [s for s in deduplicated
                           if s["quality_score"] >= self.config["min_quality_score"]]
        print(f"  Passed: {len(quality_filtered)}")

        # Step 6: 均衡抽样
        print("\n[Step 6] Balanced sampling by domain and difficulty...")
        final_seeds = self.select_balanced(quality_filtered)
        print(f"  Final selected: {len(final_seeds)}")

        # 统计报告
        self.stats = self.generate_stats(final_seeds, seeds, passed_basic, deduplicated, quality_filtered)

        result = {
            "final_seeds": final_seeds,
            "stats": self.stats,
            "failed_basic": failed_basic,
        }

        if return_details:
            result["deduplicated"] = deduplicated
            result["quality_filtered"] = quality_filtered

        return result

    @staticmethod
    def generate_stats(final_seeds: List[Dict], original: List[Dict],
                    passed_basic: List[Dict], deduplicated: List[Dict],
                    quality_filtered: List[Dict]) -> Dict:
        """生成完整的质量统计报告"""
        return {
            "pipeline": {
                "original_count": len(original),
                "passed_basic_check": len(passed_basic),
                "passed_deduplication": len(deduplicated),
                "passed_quality_filter": len(quality_filtered),
                "final_count": len(final_seeds),
                "overall_pass_rate": round(len(final_seeds) / len(original), 3) if original else 0,
            },
            "distribution": {
                "by_domain": dict(Counter(s["domain"] for s in final_seeds)),
                "by_source": dict(Counter(s["source"] for s in final_seeds)),
                "by_difficulty": dict(Counter(s["difficulty"] for s in final_seeds)),
            },
            "quality_scores": {
                "min": round(min(s["quality_score"] for s in final_seeds), 2) if final_seeds else 0,
                "max": round(max(s["quality_score"] for s in final_seeds), 2) if final_seeds else 0,
                "avg": round(sum(s["quality_score"] for s in final_seeds) / len(final_seeds), 2) if final_seeds else 0,
                "p90": round(sorted(s["quality_score"] for s in final_seeds)[int(len(final_seeds) * 0.9)], 2) if final_seeds else 0,
            }
        }

    def print_report(self):
        """打印质量评估报告"""
        print("\n" + "=" * 70)
        print("QUALITY ASSESSMENT REPORT")
        print("=" * 70)

        print("\n[Pipeline Summary]")
        for key, value in self.stats["pipeline"].items():
            print(f"  {key}: {value}")

        print("\n[Distribution]")
        for category, dist in self.stats["distribution"].items():
            print(f"  {category}: {dist}")

        print("\n[Quality Scores]")
        for key, value in self.stats["quality_scores"].items():
            print(f"  {key}: {value}")

        # 质量分分布直方图
        print("\n[Quality Score Distribution]")
        buckets = [(0.7, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.96)]
        final_seeds = self.stats.get("_seeds", [])
        scores = [s.get("quality_score", 0) for s in final_seeds]

        for low, high in buckets:
            count = sum(1 for s in scores if low <= s < high)
            bar = "█" * int(count / len(scores) * 50) if scores else ""
            print(f"  {low:.2f} - {high:.2f}: {count:3d} {bar}")

        print("\n" + "=" * 70)


def load_seeds(file_path: str) -> List[Dict]:
    """加载种子文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompts", data) if isinstance(data, dict) else data


def save_seeds(seeds: List[Dict], file_path: str):
    """保存种子文件"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"prompts": seeds}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(seeds)} seeds to: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Seed Pool Quality Assessment Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input seed pool JSON file")
    parser.add_argument("--output", "-o", help="Output filtered seed pool JSON file")
    parser.add_argument("--min-quality", type=float, default=0.70, help="Minimum quality score (default: 0.70)")
    parser.add_argument("--similarity-threshold", type=float, default=0.70, help="Deduplication Jaccard threshold (default: 0.70)")
    parser.add_argument("--stats-only", action="store_true", help="Only show stats, do not filter")

    args = parser.parse_args()

    # 加载种子
    seeds = load_seeds(args.input)
    print(f"Loaded {len(seeds)} seeds from {args.input}")

    # 配置pipeline
    config = QualityAssessmentPipeline.default_config()
    config["min_quality_score"] = args.min_quality
    config["similarity_threshold"] = args.similarity_threshold

    pipeline = QualityAssessmentPipeline(config)

    # 运行评估
    if args.stats_only:
        # 只做统计不做过滤
        pipeline.stats = pipeline.generate_stats(seeds, seeds, [], [], [])
        pipeline.stats["_seeds"] = seeds
        pipeline.print_report()
    else:
        result = pipeline.run(seeds, return_details=True)
        pipeline.stats["_seeds"] = result["final_seeds"]
        pipeline.print_report()

        if args.output:
            save_seeds(result["final_seeds"], args.output)


if __name__ == "__main__":
    main()
