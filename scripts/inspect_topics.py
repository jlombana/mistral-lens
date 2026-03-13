#!/usr/bin/env python3
"""Inspect topic distribution across repliqa splits.

Usage:
    python scripts/inspect_topics.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_topics_from_dataset(split_name: str) -> list[str]:
    """Load document topics from a HuggingFace dataset split.

    Returns one topic per unique document (deduplicates by document_id).
    """
    from datasets import load_dataset

    ds = load_dataset("ServiceNow/repliqa", split=split_name)
    seen = set()
    topics = []
    for record in ds:
        doc_id = record["document_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            topics.append(record["document_topic"])
    return topics


def print_distribution(name: str, topics: list[str]) -> Counter:
    """Print topic distribution stats and return the counter."""
    counter = Counter(topics)
    print(f"\n{'='*60}")
    print(f"Split: {name}")
    print(f"{'='*60}")
    print(f"  Total unique documents: {len(topics)}")
    print(f"  Unique categories: {len(counter)}")
    print(f"  Min examples per category: {min(counter.values())}")
    print(f"  Max examples per category: {max(counter.values())}")
    print(f"\n  Category distribution:")
    for topic, count in counter.most_common():
        print(f"    {count:4d}  {topic}")
    return counter


def main() -> None:
    """Inspect topic distribution in dev and holdout splits."""
    print("Loading repliqa_0 (dev)...")
    topics_0 = load_topics_from_dataset("repliqa_0")
    print("Loading repliqa_3 (eval)...")
    topics_3 = load_topics_from_dataset("repliqa_3")

    counter_0 = print_distribution("repliqa_0 (dev)", topics_0)
    counter_3 = print_distribution("repliqa_3 (eval)", topics_3)

    cats_0 = set(counter_0.keys())
    cats_3 = set(counter_3.keys())

    # Gap analysis
    in_eval_not_dev = cats_3 - cats_0
    in_dev_not_eval = cats_0 - cats_3

    print(f"\n{'='*60}")
    print("GAP ANALYSIS")
    print(f"{'='*60}")

    print(f"\n  Categories in repliqa_3 but NOT in repliqa_0 (CRITICAL GAPS): {len(in_eval_not_dev)}")
    if in_eval_not_dev:
        for cat in sorted(in_eval_not_dev):
            print(f"    !! {cat} ({counter_3[cat]} docs in eval)")
    else:
        print("    None — full coverage")

    print(f"\n  Categories in repliqa_0 but NOT in repliqa_3: {len(in_dev_not_eval)}")
    if in_dev_not_eval:
        for cat in sorted(in_dev_not_eval):
            print(f"    -- {cat} ({counter_0[cat]} docs in dev)")
    else:
        print("    None")

    print(f"\n  Shared categories: {len(cats_0 & cats_3)}")


if __name__ == "__main__":
    main()
