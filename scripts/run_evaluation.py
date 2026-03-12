#!/usr/bin/env python3
"""Run the full Mistral-Lens evaluation pipeline.

Usage:
    python scripts/run_evaluation.py [--dataset-path PATH] [--results-path PATH]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import run_evaluation


def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Mistral-Lens evaluation pipeline")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to dataset directory (default: from .env)",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Path to results directory (default: from .env)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        run_evaluation(
            dataset_path=args.dataset_path,
            results_path=args.results_path,
        )
    except Exception as e:
        logging.error("Evaluation failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
