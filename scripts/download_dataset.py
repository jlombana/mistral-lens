#!/usr/bin/env python3
"""Download and prepare the evaluation dataset.

Usage:
    python scripts/download_dataset.py [--output-path PATH]

TODO: Implement actual dataset download logic once dataset source is defined.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    """Download dataset to the specified directory."""
    parser = argparse.ArgumentParser(description="Download evaluation dataset")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/"),
        help="Output directory for dataset (default: data/)",
    )
    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    print(f"Dataset directory: {args.output_path}")
    print("TODO: Implement dataset download from source")
    print("For now, place images and ground_truth.json manually in the data/ directory.")


if __name__ == "__main__":
    main()
