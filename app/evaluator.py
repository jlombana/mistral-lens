"""Batch evaluation engine — runs pipeline on repliqa_3 holdout set.

Processes N documents from the evaluation split, computes all metrics,
and yields per-document results for real-time UI updates.
"""

from __future__ import annotations

import json
import logging
import random
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from app.cache import load_from_cache, save_to_cache
from app.config import get_settings
from app.extractor import extract_document
from app.metrics import (
    compute_text_metrics,
    compute_topic_accuracy,
    judge_answer,
    judge_topic,
)
from app.utils import save_json, timestamp_now

logger = logging.getLogger(__name__)

# Mistral pricing (USD per 1M tokens)
PRICING = {
    "input": 2.0,
    "output": 6.0,
    "page": 0.01,
}


@dataclass
class EvalResult:
    """Result for a single evaluated document."""

    index: int = 0
    document_id: str = ""
    num_pages: int = 1
    topic_extracted: str = ""
    topic_gt: str = ""
    topic_match: bool = False
    wer: float = 1.0
    rouge_l: float = 0.0
    topic_score: int = 0
    topic_rationale: str = ""
    answer_score: int = 0
    answer_rationale: str = ""
    answer_extracted: str = ""
    answer_gt: str = ""
    question: str = ""
    extracted_text: str = ""
    reference_text: str = ""
    latency_s: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    status: str = "ok"  # ok, error, skipped
    error_msg: str = ""
    topic_criteria: dict = field(default_factory=dict)
    answer_criteria: dict = field(default_factory=dict)


def _load_records(data_dir: Path, split: str, limit: int) -> list[dict]:
    """Load unique docs from a split JSONL."""
    jsonl_path = data_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        return []

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
                if len(docs) >= limit:
                    break
    return docs


def _load_all_unique_docs(data_dir: Path, split: str) -> list[dict]:
    """Load ALL unique docs from a split JSONL (no limit)."""
    jsonl_path = data_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        return []

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


def preview_sample(sample_size: int, split: str = "repliqa_3") -> tuple[list[dict], int, int]:
    """Generate a random sample and compute category coverage.

    Args:
        sample_size: Number of documents to sample.
        split: Dataset split.

    Returns:
        Tuple of (sampled_records, unique_categories, total_available).
    """
    settings = get_settings()
    all_docs = _load_all_unique_docs(settings.dataset_dir, split)
    total_available = len(all_docs)

    if sample_size >= total_available:
        sample = all_docs
    else:
        sample = random.sample(all_docs, sample_size)

    unique_cats = len({r.get("document_topic", "") for r in sample})
    return sample, unique_cats, total_available


def _estimate_cost(prompt_tokens: int, completion_tokens: int, pages: int = 1) -> float:
    return (
        (prompt_tokens / 1_000_000) * PRICING["input"]
        + (completion_tokens / 1_000_000) * PRICING["output"]
        + pages * PRICING["page"]
    )


def run_evaluation(
    limit: int = 50,
    split: str = "repliqa_3",
    doc_ids: list[str] | None = None,
    use_cache: bool = True,
) -> Generator[tuple[EvalResult, int, int], None, None]:
    """Run evaluation on repliqa_3 holdout set.

    Args:
        limit: Max number of unique documents to evaluate.
        split: Dataset split to evaluate.
        doc_ids: Optional pre-selected document IDs (from sample slider).
        use_cache: Whether to use cached results. If False, re-runs API calls.

    Yields:
        Tuple of (EvalResult, current_index, total_docs).
    """
    settings = get_settings()
    data_dir = settings.dataset_dir

    if doc_ids is not None:
        all_docs = _load_all_unique_docs(data_dir, split)
        id_set = set(doc_ids)
        records = [r for r in all_docs if r["document_id"] in id_set]
    else:
        records = _load_records(data_dir, split, limit)
    total = len(records)
    logger.info("Evaluation: %d documents from %s", total, split)

    for idx, record in enumerate(records):
        doc_id = record["document_id"]
        question = record.get("question", "")
        result = EvalResult(
            index=idx + 1,
            document_id=doc_id,
            question=question,
            topic_gt=record.get("document_topic", ""),
            reference_text=record.get("document_extracted", ""),
            answer_gt=record.get("long_answer", record.get("answer", "")),
        )

        try:
            pdf_dir = data_dir / "pdfs" / split
            pdf_path = pdf_dir / f"{doc_id}.pdf"

            if not pdf_path.exists():
                result.status = "skipped"
                result.error_msg = f"PDF not found: {pdf_path}"
                yield result, idx + 1, total
                continue

            # Check cache
            cached = load_from_cache(pdf_path) if use_cache else None
            if cached:
                result.cached = True
                result.extracted_text = cached.get("extracted_text", "")
                result.topic_extracted = cached.get("topic", "")
                result.answer_extracted = cached.get("answer", "")
                result.wer = cached.get("wer", 1.0)
                result.rouge_l = cached.get("rouge_l", 0.0)
                result.topic_score = cached.get("topic_score", 0)
                result.topic_rationale = cached.get("topic_rationale", "")
                result.answer_score = cached.get("answer_score", 0)
                result.answer_rationale = cached.get("answer_rationale", "")
                result.latency_s = cached.get("latency_s", 0.0)
                result.cost_usd = cached.get("cost_usd", 0.0)
                result.num_pages = cached.get("num_pages", 1)
                result.topic_criteria = cached.get("topic_criteria", {})
                result.answer_criteria = cached.get("answer_criteria", {})
                result.topic_match = (
                    compute_topic_accuracy(result.topic_extracted, result.topic_gt) == 1.0
                )
                result.status = "ok"
                yield result, idx + 1, total
                continue

            # Run extraction pipeline (OCR → Topic → Q&A)
            extraction = extract_document(pdf_path, question=question, document_id=doc_id)

            result.extracted_text = extraction.extracted_text
            result.topic_extracted = extraction.topic
            result.answer_extracted = extraction.answer or ""
            result.latency_s = extraction.latency_total_s
            result.num_pages = max(1, extraction.extracted_text.count("\n\n") // 2 + 1)
            result.cost_usd = _estimate_cost(
                extraction.tokens_prompt, extraction.tokens_completion, result.num_pages
            )

            # Text metrics
            tm = compute_text_metrics(doc_id, result.reference_text, result.extracted_text)
            result.wer = tm.wer
            result.rouge_l = tm.rouge_l

            # Topic accuracy + judge
            result.topic_match = (
                compute_topic_accuracy(result.topic_extracted, result.topic_gt) == 1.0
            )
            if result.topic_gt and result.topic_extracted:
                js = judge_topic(doc_id, result.topic_gt, result.topic_extracted)
                result.topic_score = js.score
                result.topic_rationale = js.rationale
                result.topic_criteria = js.criteria

            # Answer judge
            if result.answer_gt and result.answer_extracted:
                ja = judge_answer(doc_id, question, result.answer_gt, result.answer_extracted)
                result.answer_score = ja.score
                result.answer_rationale = ja.rationale
                result.answer_criteria = ja.criteria

            result.status = "ok"

            # Save to cache
            cache_data = {
                "document_id": doc_id,
                "extracted_text": result.extracted_text,
                "topic": result.topic_extracted,
                "answer": result.answer_extracted,
                "wer": result.wer,
                "rouge_l": result.rouge_l,
                "topic_score": result.topic_score,
                "topic_rationale": result.topic_rationale,
                "answer_score": result.answer_score,
                "answer_rationale": result.answer_rationale,
                "latency_s": result.latency_s,
                "cost_usd": result.cost_usd,
                "num_pages": result.num_pages,
                "topic_criteria": result.topic_criteria,
                "answer_criteria": result.answer_criteria,
            }
            save_to_cache(pdf_path, cache_data)

        except Exception as exc:
            result.status = "error"
            result.error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Error processing %s: %s", doc_id, traceback.format_exc()
            )

        yield result, idx + 1, total


def save_eval_results(results: list[EvalResult]) -> Path:
    """Save evaluation results to a timestamped JSON file.

    Returns:
        Path to the saved file.
    """
    ts = timestamp_now().replace(":", "-").replace(" ", "_")
    output_path = Path("results") / f"eval_{ts}.json"

    ok_results = [r for r in results if r.status == "ok"]

    summary = {
        "timestamp": timestamp_now(),
        "split": "repliqa_3",
        "total_documents": len(results),
        "successful": len(ok_results),
        "errors": sum(1 for r in results if r.status == "error"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "cached": sum(1 for r in results if r.cached),
        "avg_wer": round(sum(r.wer for r in ok_results) / len(ok_results), 4)
        if ok_results
        else 0,
        "avg_rouge_l": round(sum(r.rouge_l for r in ok_results) / len(ok_results), 4)
        if ok_results
        else 0,
        "avg_topic_score": round(
            sum(r.topic_score for r in ok_results) / len(ok_results), 2
        )
        if ok_results
        else 0,
        "avg_answer_score": round(
            sum(r.answer_score for r in ok_results) / len(ok_results), 2
        )
        if ok_results
        else 0,
        "topic_accuracy": round(
            sum(1 for r in ok_results if r.topic_match) / len(ok_results), 4
        )
        if ok_results
        else 0,
        "avg_latency_s": round(
            sum(r.latency_s for r in ok_results) / len(ok_results), 2
        )
        if ok_results
        else 0,
        "total_cost_usd": round(sum(r.cost_usd for r in ok_results), 4),
    }

    details = []
    for r in results:
        details.append(
            {
                "index": r.index,
                "document_id": r.document_id,
                "num_pages": r.num_pages,
                "topic_extracted": r.topic_extracted,
                "topic_gt": r.topic_gt,
                "topic_match": r.topic_match,
                "wer": r.wer,
                "rouge_l": r.rouge_l,
                "topic_score": r.topic_score,
                "answer_score": r.answer_score,
                "latency_s": r.latency_s,
                "cost_usd": r.cost_usd,
                "cached": r.cached,
                "status": r.status,
                "error_msg": r.error_msg,
            }
        )

    save_json({"summary": summary, "details": details}, output_path)
    return output_path
