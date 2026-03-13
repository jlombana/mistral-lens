#!/usr/bin/env python3
"""Build fine-tuning dataset for topic classification.

Generates:
  - data/topic_finetune_train.jsonl (from repliqa_0 dev set)
  - data/topic_finetune_val.jsonl (from repliqa_3 eval set)
  - data/category_list.txt (unique categories)

Usage:
    python scripts/build_finetune_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data")
MAX_TEXT_CHARS = 2000


def load_unique_docs(jsonl_path: Path) -> list[dict]:
    """Load unique documents from JSONL, deduplicating by document_id."""
    seen = set()
    docs = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            doc_id = record["document_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                docs.append(record)
    return docs


def build_message(text: str, topic: str, category_list: str) -> dict:
    """Build a fine-tuning message in Mistral chat format."""
    user_content = (
        f"Classify the following document into exactly one of these categories:\n"
        f"{category_list}\n\n"
        f"Return only the exact category name, nothing else.\n\n"
        f"Document:\n{text[:MAX_TEXT_CHARS]}"
    )
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": topic},
        ]
    }


def main() -> None:
    """Build fine-tuning train and validation datasets."""
    # Load dev and eval docs
    train_docs = load_unique_docs(DATA_DIR / "repliqa_0.jsonl")
    val_docs = load_unique_docs(DATA_DIR / "repliqa_3.jsonl")

    print(f"Train docs (repliqa_0): {len(train_docs)}")
    print(f"Val docs (repliqa_3): {len(val_docs)}")

    # Extract all unique categories from training set
    all_categories = sorted(set(doc["document_topic"] for doc in train_docs))
    category_list_str = "\n".join(all_categories)

    # Save category list
    cat_file = DATA_DIR / "category_list.txt"
    cat_file.write_text("\n".join(all_categories) + "\n")
    print(f"\nCategories ({len(all_categories)}):")
    for cat in all_categories:
        print(f"  - {cat}")

    # Build training JSONL
    train_file = DATA_DIR / "topic_finetune_train.jsonl"
    train_counter = Counter()
    with open(train_file, "w") as f:
        for doc in train_docs:
            topic = doc["document_topic"]
            text = doc["document_extracted"]
            msg = build_message(text, topic, category_list_str)
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            train_counter[topic] += 1

    print(f"\nTraining set: {sum(train_counter.values())} examples")
    print("  Examples per category:")
    for topic, count in train_counter.most_common():
        print(f"    {count:4d}  {topic}")

    # Build validation JSONL
    val_file = DATA_DIR / "topic_finetune_val.jsonl"
    val_counter = Counter()
    with open(val_file, "w") as f:
        for doc in val_docs:
            topic = doc["document_topic"]
            text = doc["document_extracted"]
            msg = build_message(text, topic, category_list_str)
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            val_counter[topic] += 1

    print(f"\nValidation set: {sum(val_counter.values())} examples")
    print("  Examples per category:")
    for topic, count in val_counter.most_common():
        print(f"    {count:4d}  {topic}")

    # Verify outputs
    print(f"\nFiles created:")
    print(f"  {train_file} ({sum(train_counter.values())} lines)")
    print(f"  {val_file} ({sum(val_counter.values())} lines)")
    print(f"  {cat_file} ({len(all_categories)} categories)")


if __name__ == "__main__":
    main()
