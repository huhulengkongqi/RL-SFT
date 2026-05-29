"""Check that QualityFilter embedding mode can encode texts locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.quality_filter.quality_filter import (  # noqa: E402
    cosine_similarity_matrix,
    encode_texts_with_transformers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local embedding encoder used by QualityFilter")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = args.text or [
        "Fix the Python function and explain the bug.",
        "Repair the Python function and describe the root cause.",
        "Compute the final numeric answer for the math problem.",
    ]
    embeddings = encode_texts_with_transformers(texts, args.model)
    similarities = cosine_similarity_matrix(embeddings)
    print(
        json.dumps(
            {
                "model": args.model,
                "texts": texts,
                "embedding_count": len(embeddings),
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
                "similarities": similarities,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
