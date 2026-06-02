#!/usr/bin/env python
"""
Complete SFT Data Formatter Test Script with configurable parameters.

Usage:
    # Basic test with built-in sample
    python scripts/sft_formatter_full_test.py

    # Test with real data directory
    python scripts/sft_formatter_full_test.py --data-dir data/sft_trajectories --limit 10

    # Test different truncation strategies
    python scripts/sft_formatter_full_test.py --strategy middle --max-tokens 4096

    # Full pipeline with export
    python scripts/sft_formatter_full_test.py --data-dir data/sft_trajectories --limit 50 \\
        --output-dir data/formatted_sft --export-format both

    # Custom loss mask settings
    python scripts/sft_formatter_full_test.py --thought-max-tokens 200 --no-mask-tool
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.dataset_builder import (
    SFTDataPipeline,
    SFTFormatterConfig,
    FormatConfig,
    TruncationConfig,
    LossMaskConfig,
    TokenCounter,
)


def load_raw_trajectories(data_dir: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load raw trajectory JSON files from a directory."""
    data_path = Path(data_dir)
    raw_files = sorted(data_path.glob("*_raw.json"))

    if limit:
        raw_files = raw_files[:limit]

    trajectories = []
    for f in raw_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                traj = json.load(fp)
                # Check if this is a valid raw trajectory (has 'steps' field)
                if "steps" in traj:
                    trajectories.append(traj)
                else:
                    print(f"  Skip {f.name}: no 'steps' field (already formatted?)")
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")

    return trajectories


def create_sample_trajectory() -> Dict[str, Any]:
    """Create a synthetic long trajectory for testing."""
    steps = []
    for i in range(20):  # 20 steps to ensure it's long enough for truncation
        is_final = i == 19
        step = {
            "state_snapshot": {
                "metadata": {
                    "task_prompt": "Solve this problem step by step: A train travels from A to B at 60 km/h..."
                } if i == 0 else {},
            },
            "thought": f"Step {i+1}: Let me think about this carefully. " * (5 + i),  # Get longer each step
            "action": {
                "action_type": "final_answer" if is_final else "tool_call",
                "name": "python" if not is_final else None,
                "kwargs": {"code": f"result = {i} * 60"} if not is_final else {},
                "answer": "Final answer: The train will arrive in 8 hours." if is_final else None,
            },
            "observation": {
                "success": True,
                "content": f"Output: {i * 60}\nExecution time: 0.001s" if not is_final else "",
            },
        }
        steps.append(step)

    return {
        "task_id": "synthetic_001",
        "domain": "math_reasoning",
        "difficulty": "medium",
        "steps": steps,
        "final_state": {"done": True},
        "termination_reason": "success",
        "final_score": 1.0,
        "success": True,
    }


def print_statistics(name: str, values: List[int]) -> None:
    """Print statistics for a list of numbers."""
    if not values:
        print(f"  {name}: No data")
        return
    print(f"  {name}:")
    print(f"    Count: {len(values)}")
    print(f"    Min: {min(values)}")
    print(f"    Max: {max(values)}")
    print(f"    Mean: {statistics.mean(values):.1f}")
    print(f"    Median: {statistics.median(values):.1f}")


def main():
    parser = argparse.ArgumentParser(description="SFT Data Formatter Test Script")

    # Data source
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing *_raw.json files (default: use synthetic data)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of trajectories to process (default: all)",
    )

    # Format config
    parser.add_argument(
        "--format",
        type=str,
        choices=["react", "function_json"],
        default="react",
        help="Trajectory format (default: react)",
    )
    parser.add_argument(
        "--merge-thought-action",
        action="store_true",
        default=True,
        help="Merge thought and action into one message (default: True)",
    )
    parser.add_argument(
        "--use-tool-role",
        action="store_true",
        default=True,
        help="Use 'tool' role for observations instead of 'user' (default: True)",
    )

    # Truncation config
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["middle", "head", "tail"],
        default="middle",
        help="Truncation strategy (default: middle)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3277,  # 4096 * 0.8
        help="Maximum token count (default: 3277 = 80%% of 4096)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=4096,
        help="Model context window for auto-scaling (default: 4096)",
    )
    parser.add_argument(
        "--no-auto-scale",
        action="store_true",
        help="Disable auto-scaling max_tokens to 80%% of context window",
    )
    parser.add_argument(
        "--head-keep-steps",
        type=int,
        default=2,
        help="Number of steps to keep from head (middle strategy, default: 2)",
    )
    parser.add_argument(
        "--tail-keep-steps",
        type=int,
        default=3,
        help="Number of steps to keep from tail (middle strategy, default: 3)",
    )

    # Loss mask config
    parser.add_argument(
        "--no-mask-system",
        action="store_true",
        help="Don't mask system messages (default: mask)",
    )
    parser.add_argument(
        "--no-mask-user",
        action="store_true",
        help="Don't mask user messages (default: mask)",
    )
    parser.add_argument(
        "--no-mask-tool",
        action="store_true",
        help="Don't mask tool messages (default: mask)",
    )
    parser.add_argument(
        "--no-mask-thought",
        action="store_true",
        help="Disable long thought masking (default: enabled)",
    )
    parser.add_argument(
        "--thought-max-tokens",
        type=int,
        default=500,
        help="Mask thoughts longer than this token count (default: 500)",
    )

    # Tokenizer config
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer model for counting (default: Qwen/Qwen2.5-7B-Instruct)",
    )

    # Export config
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for formatted data (default: no export)",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        choices=["parquet", "jsonl", "both"],
        default="both",
        help="Export format (default: both)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="formatted_sft",
        help="Name for exported dataset (default: formatted_sft)",
    )

    args = parser.parse_args()

    # Build config
    config = SFTFormatterConfig(
        format=FormatConfig(
            trajectory_format=args.format,
            merge_thought_action=args.merge_thought_action,
            use_tool_role=args.use_tool_role,
        ),
        truncation=TruncationConfig(
            strategy=args.strategy,
            max_tokens=args.max_tokens,
            context_window=args.context_window,
            auto_scale=not args.no_auto_scale,
            head_keep_steps=args.head_keep_steps,
            tail_keep_steps=args.tail_keep_steps,
        ),
        loss_mask=LossMaskConfig(
            mask_system=not args.no_mask_system,
            mask_user=not args.no_mask_user,
            mask_tool=not args.no_mask_tool,
            mask_thought=not args.no_mask_thought,
            thought_max_tokens=args.thought_max_tokens,
        ),
        tokenizer_model=args.tokenizer_model,
    )

    print("=" * 70)
    print("SFT Data Formatter - Full Test Script")
    print("=" * 70)

    # Print configuration
    print("\n[Configuration]")
    print(f"  Format: {args.format}")
    print(f"  Merge thought+action: {args.merge_thought_action}")
    print(f"  Use tool role: {args.use_tool_role}")
    print(f"  Truncation strategy: {args.strategy}")
    print(f"  Max tokens: {args.max_tokens} (context: {args.context_window})")
    print(f"  Head keep steps: {args.head_keep_steps}")
    print(f"  Tail keep steps: {args.tail_keep_steps}")
    print(f"  Mask system: {not args.no_mask_system}")
    print(f"  Mask user: {not args.no_mask_user}")
    print(f"  Mask tool: {not args.no_mask_tool}")
    print(f"  Mask long thought: {not args.no_mask_thought} (threshold: {args.thought_max_tokens} tokens)")
    print(f"  Tokenizer: {args.tokenizer_model}")

    # Load data
    print("\n[Loading Data]")
    if args.data_dir:
        print(f"  Loading from: {args.data_dir}")
        trajectories = load_raw_trajectories(args.data_dir, args.limit)
        print(f"  Loaded: {len(trajectories)} valid raw trajectories")
        if not trajectories:
            print("  ERROR: No valid raw trajectories found!")
            return 1
    else:
        print("  Using synthetic test data")
        trajectories = [create_sample_trajectory()]

    # Initialize pipeline
    print("\n[Initializing Pipeline]")
    pipeline = SFTDataPipeline(config)
    print("  Ready!")

    # Process trajectories
    print("\n[Processing]")
    print(f"  Processing {len(trajectories)} trajectories...")

    results = []
    token_counts_before = []
    token_counts_after = []
    messages_before = []
    messages_after = []
    truncated_count = 0

    for i, traj in enumerate(trajectories):
        # First count tokens without truncation for comparison
        # We use the formatter directly then count
        formatted = pipeline.formatter.format(traj)
        count_before = pipeline.token_counter.count_messages(formatted["messages"])
        token_counts_before.append(count_before)
        messages_before.append(len(formatted["messages"]))

        # Full pipeline (with truncation)
        result = pipeline.process_trajectory(traj)
        results.append(result)

        count_after = result["token_count"]
        token_counts_after.append(count_after)
        messages_after.append(len(result["messages"]))

        if count_after < count_before:
            truncated_count += 1

        if i < 3 or i == len(trajectories) - 1:  # Show first 3 and last
            status = "TRUNCATED" if count_after < count_before else "OK"
            print(f"  [{i+1}/{len(trajectories)}] Task: {traj.get('task_id', 'unknown')[:20]}... "
                  f"Tokens: {count_before} -> {count_after} ({status})")

    # Statistics
    print("\n[Statistics]")

    print("\nToken Counts Before Truncation:")
    print_statistics("Tokens", token_counts_before)

    print("\nToken Counts After Truncation:")
    print_statistics("Tokens", token_counts_after)

    print("\nTruncation Summary:")
    print(f"  Total processed: {len(trajectories)}")
    print(f"  Truncated: {truncated_count} ({truncated_count/len(trajectories)*100:.1f}%)")

    if truncated_count > 0:
        reductions = [
            (b - a) / b * 100 for b, a in zip(token_counts_before, token_counts_after) if b > a
        ]
        if reductions:
            print(f"  Average reduction: {statistics.mean(reductions):.1f}%")
            print(f"  Max reduction: {max(reductions):.1f}%")

    # Role distribution
    print("\nMessage Role Distribution:")
    role_counts: Dict[str, int] = {}
    for r in results:
        for msg in r["messages"]:
            role = msg["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
    total = sum(role_counts.values())
    for role, count in sorted(role_counts.items()):
        print(f"  {role:10s}: {count:5d} ({count/total*100:5.1f}%)")

    # Loss mask statistics
    print("\nLoss Mask Statistics:")
    mask_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        for mask_info in r["message_loss_mask"]:
            role = mask_info["role"]
            if role not in mask_stats:
                mask_stats[role] = {"train": 0, "masked": 0}
            key = "train" if mask_info["loss_mask"] else "masked"
            mask_stats[role][key] += 1

    for role, stats in sorted(mask_stats.items()):
        total_role = stats["train"] + stats["masked"]
        train_pct = stats["train"] / total_role * 100 if total_role > 0 else 0
        print(f"  {role:10s}: TRAIN={stats['train']:3d}, MASKED={stats['masked']:3d} ({train_pct:.1f}% train)")

    # Export
    if args.output_dir:
        print("\n[Exporting]")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        export_paths = pipeline.exporter.export(
            results, output_dir=args.output_dir, name=args.dataset_name
        )

        for fmt, path in export_paths.items():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {fmt.upper()}: {path} ({size_mb:.2f} MB)")

        # Verify with datasets
        print("\n[Verification]")
        try:
            from datasets import load_dataset
            ds = load_dataset("parquet", data_files=str(export_paths["parquet"]), split="train")
            print(f"  Dataset loaded successfully!")
            print(f"    Rows: {len(ds)}")
            print(f"    Columns: {list(ds.column_names)}")
        except ImportError:
            print("  datasets package not installed, skipping verification")
        except Exception as e:
            print(f"  Verification error: {e}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
