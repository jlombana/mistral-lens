#!/usr/bin/env python3
"""Run the Mistral-Lens evaluation pipeline on repliqa dataset.

Usage:
    python scripts/run_evaluation.py [--split repliqa_0] [--limit 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.extractor import extract_document
from app.metrics import compute_metrics
from app.utils import save_json, timestamp_now


# Mistral pricing (USD per 1M tokens, as of March 2026)
PRICING = {
    "mistral-large-latest": {"input": 2.0, "output": 6.0},
    "mistral-small-latest": {"input": 0.1, "output": 0.3},
    "mistral-ocr-latest": {"page": 0.01},  # per page estimate
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a chat completion call.

    Args:
        model: Model identifier.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    pricing = PRICING.get(model, PRICING["mistral-large-latest"])
    input_cost = (prompt_tokens / 1_000_000) * pricing.get("input", 2.0)
    output_cost = (completion_tokens / 1_000_000) * pricing.get("output", 6.0)
    return input_cost + output_cost


def load_records(data_dir: Path, split: str, limit: int) -> list[dict]:
    """Load records from a JSONL file.

    Args:
        data_dir: Directory containing dataset files.
        split: Split name (e.g., 'repliqa_0').
        limit: Max records (0 = all).

    Returns:
        List of record dicts.
    """
    jsonl_file = data_dir / f"{split}.jsonl"
    json_file = data_dir / f"{split}.json"

    if jsonl_file.exists():
        path = jsonl_file
    elif json_file.exists():
        path = json_file
    else:
        print(f"Dataset not found at {jsonl_file} or {json_file}")
        print("Run: python scripts/download_dataset.py first")
        sys.exit(1)

    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if limit > 0:
        records = records[:limit]
    return records


def main() -> None:
    """Run evaluation on a repliqa split."""
    parser = argparse.ArgumentParser(description="Mistral-Lens evaluation pipeline")
    parser.add_argument("--split", default="repliqa_0", help="Dataset split (default: repliqa_0)")
    parser.add_argument("--limit", type=int, default=3, help="Max documents (default: 3)")
    parser.add_argument("--results-path", type=Path, default=None, help="Results directory")
    parser.add_argument("--skip-ocr", action="store_true", help="Use ground truth text instead of OCR")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    settings = get_settings()
    results_path = args.results_path or settings.results_dir
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"Mistral-Lens Evaluation")
    print(f"  Split: {args.split}")
    print(f"  Limit: {args.limit} documents")
    print(f"  OCR model: {settings.OCR_MODEL}")
    print(f"  Chat model: {settings.CHAT_MODEL}")
    print(f"  Skip OCR: {args.skip_ocr}")
    print()

    # Load dataset records
    records = load_records(settings.dataset_dir, args.split, args.limit)
    print(f"Loaded {len(records)} records from {args.split}")

    # Deduplicate by document_id (multiple Q&A per doc)
    docs_by_id = {}
    for r in records:
        doc_id = r["document_id"]
        if doc_id not in docs_by_id:
            docs_by_id[doc_id] = r
    unique_docs = list(docs_by_id.values())

    pdf_dir = settings.dataset_dir / "pdfs" / args.split

    # Step 1+2+3: Extract from each document
    extractions = []
    ground_truth = []
    total_latencies = []
    total_tokens_all = 0
    total_cost_all = 0.0

    for i, record in enumerate(unique_docs):
        doc_id = record["document_id"]
        question = record.get("question", "")
        print(f"\n[{i+1}/{len(unique_docs)}] Processing {doc_id}...")

        if args.skip_ocr:
            from app.extractor import extract_topic, extract_answer

            extracted_text = record["document_extracted"]
            topic = extract_topic(extracted_text)
            answer = extract_answer(extracted_text, question) if question else ""

            extraction = {
                "document_id": doc_id,
                "extracted_text": extracted_text,
                "topic": topic,
                "answer": answer,
            }
        else:
            # Full pipeline with OCR
            pdf_path = pdf_dir / f"{doc_id}.pdf"
            if not pdf_path.exists():
                print(f"  PDF not found: {pdf_path}, skipping")
                continue

            result = extract_document(pdf_path, question=question, document_id=doc_id)

            # Track latency and cost
            total_latencies.append(result.latency_total_s)
            total_tokens_all += result.tokens_total
            doc_cost = estimate_cost(settings.CHAT_MODEL, result.tokens_prompt, result.tokens_completion)
            total_cost_all += doc_cost

            extraction = {
                "document_id": result.document_id,
                "extracted_text": result.extracted_text,
                "topic": result.topic,
                "answer": result.answer,
            }

            print(f"  Latency: {result.latency_total_s:.1f}s (OCR:{result.latency_ocr_s:.1f}s + Topic:{result.latency_topic_s:.1f}s + Q&A:{result.latency_answer_s:.1f}s)")
            print(f"  Tokens: {result.tokens_total:,} (prompt:{result.tokens_prompt:,} + completion:{result.tokens_completion:,})")
            print(f"  Est. cost: ${doc_cost:.5f}")

        extractions.append(extraction)
        print(f"  OCR text length: {len(extraction['extracted_text'])} chars")
        print(f"  Topic: {extraction['topic'][:100]}...")
        if extraction.get("answer"):
            print(f"  Answer: {extraction['answer'][:100]}...")

        # Build ground truth entry
        gt = {
            "document_id": doc_id,
            "text": record["document_extracted"],
            "topic": record["document_topic"],
            "question": question,
            "answer": record.get("long_answer", record.get("answer", "")),
        }
        ground_truth.append(gt)

    if not extractions:
        print("\nNo documents processed. Ensure PDFs are downloaded.")
        sys.exit(1)

    # Step 4: Compute metrics
    print(f"\n{'='*60}")
    print(f"Computing metrics for {len(extractions)} documents...")
    report = compute_metrics(extractions, ground_truth)

    # Inject latency/cost into report
    if total_latencies:
        report.avg_latency_s = round(sum(total_latencies) / len(total_latencies), 3)
    report.total_tokens = total_tokens_all
    report.total_cost_usd = round(total_cost_all, 6)
    if len(extractions) > 0:
        report.avg_cost_per_doc_usd = round(total_cost_all / len(extractions), 6)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Documents evaluated: {report.total_documents}")
    print(f"  OCR model: {report.ocr_model}")
    print(f"  Chat model: {report.chat_model}")
    print(f"  Avg WER: {report.avg_wer:.4f} (target < 0.15)")
    print(f"  Avg ROUGE-L: {report.avg_rouge_l:.4f} (target > 0.80)")
    print(f"  Avg Topic Score: {report.avg_topic_score:.1f}/5 (LLM-judge)")
    print(f"  Avg Topic Accuracy: {report.avg_topic_accuracy * 100:.1f}% (exact match, target > 80%)")
    print(f"  Avg Answer Score: {report.avg_answer_score:.1f}/5 (target > 4.0)")
    print(f"  ---")
    print(f"  Avg Latency: {report.avg_latency_s:.1f}s per document")
    print(f"  Total Tokens: {report.total_tokens:,}")
    print(f"  Total Cost: ${report.total_cost_usd:.4f}")
    print(f"  Avg Cost/Doc: ${report.avg_cost_per_doc_usd:.5f}")
    print(f"{'='*60}")

    # Save results
    output_file = results_path / f"eval_{args.split}_{timestamp_now().replace(':', '-').replace(' ', '_')}.json"
    save_json(report.model_dump(), output_file)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
