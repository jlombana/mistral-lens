#!/usr/bin/env python3
"""Download the repliqa dataset from HuggingFace.

Usage:
    python scripts/download_dataset.py [--output-path PATH] [--splits 0 1 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    """Download repliqa dataset splits."""
    parser = argparse.ArgumentParser(description="Download repliqa dataset")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/"),
        help="Output directory for dataset (default: data/)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["0", "1", "2"],
        help="Dataset splits to download (default: 0 1 2)",
    )
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    for split in args.splits:
        split_name = f"repliqa_{split}"
        print(f"Downloading {split_name}...")
        ds = load_dataset("ServiceNow/repliqa", split_name)
        output_file = args.output_path / f"{split_name}.json"
        ds.to_json(str(output_file))
        print(f"  Saved to {output_file} ({len(ds)} records)")

    print("Done.")


if __name__ == "__main__":
    main()
