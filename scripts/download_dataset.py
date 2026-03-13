#!/usr/bin/env python3
"""Download the repliqa dataset from HuggingFace (metadata + PDFs).

Usage:
    python scripts/download_dataset.py [--output-path data/] [--splits 0 1 2] [--limit 50] [--no-pdfs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def download_pdfs(records: list[dict], output_path: Path, split_name: str) -> int:
    """Download PDFs for a list of records using huggingface_hub.

    Args:
        records: Dataset records with 'document_path' field.
        output_path: Base output directory.
        split_name: Split name (e.g., 'repliqa_0').

    Returns:
        Number of PDFs successfully downloaded.
    """
    from huggingface_hub import hf_hub_download
    import shutil

    pdf_dir = output_path / "pdfs" / split_name
    pdf_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    downloaded = 0

    for record in records:
        doc_path = record.get("document_path", "")
        doc_id = record.get("document_id", "")
        if not doc_path or doc_id in seen:
            continue
        seen.add(doc_id)

        target = pdf_dir / f"{doc_id}.pdf"
        if target.exists():
            downloaded += 1
            continue

        try:
            local_path = hf_hub_download(
                repo_id="ServiceNow/repliqa",
                filename=doc_path,
                repo_type="dataset",
            )
            shutil.copy2(local_path, target)
            downloaded += 1
            print(f"    Downloaded: {doc_id}.pdf")
        except Exception as e:
            print(f"    Failed: {doc_id}.pdf — {e}")

    return downloaded


def main() -> None:
    """Download repliqa dataset splits with optional PDF download."""
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
        default=["0"],
        help="Dataset splits to download (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max records per split (0 = all, default: 0)",
    )
    parser.add_argument(
        "--no-pdfs",
        action="store_true",
        help="Skip PDF download (metadata only)",
    )
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    for split in args.splits:
        split_name = f"repliqa_{split}"
        print(f"\nDownloading {split_name}...")

        ds = load_dataset("ServiceNow/repliqa", split=split_name)
        records = [r for r in ds]

        if args.limit > 0:
            records = records[:args.limit]

        # Save metadata as JSONL
        output_file = args.output_path / f"{split_name}.jsonl"
        with open(output_file, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Metadata: {output_file} ({len(records)} records)")

        # Download PDFs
        if not args.no_pdfs:
            # Deduplicate by document_id (multiple questions per doc)
            unique_docs = {r["document_id"]: r for r in records}
            print(f"  Downloading {len(unique_docs)} unique PDFs...")
            n = download_pdfs(records, args.output_path, split_name)
            print(f"  PDFs: {n}/{len(unique_docs)} downloaded")

    print("\nDone.")


if __name__ == "__main__":
    main()
