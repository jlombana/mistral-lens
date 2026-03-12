#!/usr/bin/env python3
"""Run the Mistral-Lens evaluation pipeline on repliqa dataset.

Usage:
    python scripts/run_evaluation.py [--split repliqa_0] [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.utils import save_json, timestamp_now


def main() -> None:
    """Run evaluation on a repliqa split."""
    parser = argparse.ArgumentParser(description="Mistral-Lens evaluation pipeline")
    parser.add_argument(
        "--split",
        default="repliqa_0",
        help="Dataset split to evaluate (default: repliqa_0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max documents to evaluate (default: 50)",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Results directory (default: from .env)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    settings = get_settings()
    results_path = args.results_path or settings.results_dir

    print(f"Mistral-Lens Evaluation")
    print(f"Split: {args.split}")
    print(f"Limit: {args.limit} documents")
    print(f"OCR model: {settings.OCR_MODEL}")
    print(f"Chat model: {settings.CHAT_MODEL}")
    print()

    # Load dataset
    data_file = settings.dataset_dir / f"{args.split}.json"
    if not data_file.exists():
        print(f"Dataset not found at {data_file}")
        print("Run: python scripts/download_dataset.py first")
        sys.exit(1)

    with open(data_file) as f:
        records = [json.loads(line) for line in f][:args.limit]

    print(f"Loaded {len(records)} records from {args.split}")
    print("TODO: Implement batch extraction + evaluation once dataset schema is explored")
    print("See ML-T004 (explore dataset schema) in the project tracker")


if __name__ == "__main__":
    main()
