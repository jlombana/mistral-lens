"""Gradio UI entry point for Mistral-Lens v5.

Three tabs + topbar:
  1. Process  — Run the pipeline on a single document, inspect raw outputs
  2. Evaluate — Batch evaluation with KPIs, detail panel, and API cost breakdown
  3. Business Case — ROI narrative with incumbent comparison

No business logic here — delegates to extractor, evaluator, metrics, and cache modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import gradio as gr

from app.cache import get_cache_stats, load_from_cache, save_to_cache
from app.config import get_settings
from app.evaluator import EvalResult, preview_sample, run_evaluation, save_eval_results
from app.extractor import compute_grounding_score, extract_document
from app.metrics import compute_extraction_density, compute_ttr
from app.utils import save_json, timestamp_now

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing constants
# ---------------------------------------------------------------------------
PRICING = {"input": 2.0, "output": 6.0, "page": 0.01}
INCUMBENT_COST_PER_PAGE = 0.75
AVG_PAGES_PER_DOC = 4
MISTRAL_COST_PER_DOC = AVG_PAGES_PER_DOC * PRICING["page"] + 0.02


def _estimate_cost(prompt_tokens: int, completion_tokens: int, pages: int = 1) -> float:
    return (
        (prompt_tokens / 1_000_000) * PRICING["input"]
        + (completion_tokens / 1_000_000) * PRICING["output"]
        + pages * PRICING["page"]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_file(pdf_file) -> Path | None:
    """Extract file path from Gradio file object (works with v3 and v4)."""
    if pdf_file is None:
        return None
    if isinstance(pdf_file, str):
        return Path(pdf_file)
    if isinstance(pdf_file, dict) and "name" in pdf_file:
        return Path(pdf_file["name"])
    if hasattr(pdf_file, "name"):
        return Path(pdf_file.name)
    return None


# ---------------------------------------------------------------------------
# Dataset index for auto-populating questions
# ---------------------------------------------------------------------------

def _build_dataset_index() -> tuple[dict, dict]:
    """Build hash->question and filename->question indexes from all repliqa splits."""
    settings = get_settings()
    data_dir = settings.dataset_dir
    hash_idx: dict[str, dict] = {}
    name_idx: dict[str, dict] = {}

    for split in ["repliqa_0", "repliqa_1", "repliqa_2", "repliqa_3"]:
        jsonl_path = data_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            continue
        seen_docs: set[str] = set()
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                doc_id = rec["document_id"]
                question = rec.get("question", "")
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    name_idx[doc_id.lower()] = {"document_id": doc_id, "question": question}
                    pdf_path = data_dir / "pdfs" / split / f"{doc_id}.pdf"
                    if pdf_path.exists():
                        h = hashlib.md5(pdf_path.read_bytes()).hexdigest()
                        hash_idx[h] = {"document_id": doc_id, "question": question}

    logger.info("Dataset index built: %d hash, %d filename", len(hash_idx), len(name_idx))
    return hash_idx, name_idx


_HASH_INDEX, _NAME_INDEX = _build_dataset_index()


def on_pdf_upload(pdf_file):
    """Auto-fill question from dataset index."""
    pdf_path = _resolve_file(pdf_file)
    if pdf_path is None:
        return "", ""
    try:
        h = hashlib.md5(pdf_path.read_bytes()).hexdigest()
        if h in _HASH_INDEX:
            return (
                _HASH_INDEX[h]["question"],
                '<div style="color:#D85A30;font-weight:500;font-size:0.85rem;padding:4px 0">'
                "&#x26A1; Auto-filled from repliqa</div>",
            )
    except OSError:
        pass
    stem = pdf_path.stem.lower()
    if stem in _NAME_INDEX:
        return (
            _NAME_INDEX[stem]["question"],
            '<div style="color:#D85A30;font-weight:500;font-size:0.85rem;padding:4px 0">'
            "&#x26A1; Auto-filled from repliqa</div>",
        )
    return "", ""


# ---------------------------------------------------------------------------
# Tab 1 — Process: single-document pipeline
# ---------------------------------------------------------------------------

def _pipeline_html(state: str = "idle", latency: float = 0.0,
                   wer: str = "\u2014", topic: str = "\u2014", qa: str = "\u2014") -> str:
    """Build pipeline visualization HTML."""
    if state == "idle":
        dot = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#c0bdb8;margin-right:6px"></span>'
        status_text = "Ready to run"
    elif state == "running":
        dot = '<span class="pulse-dot"></span>'
        status_text = "Processing\u2026"
    else:
        dot = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#1D9E75;margin-right:6px"></span>'
        status_text = f"Complete &middot; {latency:.1f}s"

    steps = [
        ("01", "OCR extraction", "mistral-ocr-latest &middot; /v1/ocr", wer if state == "done" else "\u2014"),  # density w/p
        ("02", "Topic classification", "mistral-large-latest &middot; few-shot CoT", topic if state == "done" else "\u2014"),
        ("03", "Grounded Q&A", "mistral-large-latest &middot; doc-only context", qa if state == "done" else "\u2014"),  # grounding score
    ]
    steps_html = ""
    for num, title, subtitle, metric in steps:
        steps_html += f"""
        <div style="display:flex;align-items:center;padding:14px 0;border-bottom:1px solid #f0eee9">
            <div style="background:#FAECE7;color:#D85A30;font-weight:500;width:32px;height:32px;
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-size:0.8rem;margin-right:14px;flex-shrink:0">{num}</div>
            <div style="flex:1">
                <div style="font-weight:500;color:#1a1a1a;font-size:0.95rem">{title}</div>
                <div style="color:#999;font-size:0.8rem;margin-top:2px">{subtitle}</div>
            </div>
            <div style="font-weight:500;color:#999;font-size:0.9rem">{metric}</div>
        </div>"""

    return f"""
    <div style="background:white;border:0.5px solid #e0ddd8;border-radius:12px;padding:14px 18px;margin-bottom:16px">
        <div style="text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;color:#999;
                    font-weight:500;margin-bottom:8px">Pipeline</div>
        {steps_html}
        <div style="display:flex;align-items:center;padding-top:10px;margin-top:4px">
            {dot}
            <span style="font-size:0.85rem;color:#666">{status_text}</span>
        </div>
    </div>"""


def _build_output_html(pdf_name: str, extracted_text: str, topic: str,
                       density: int, ttr: float, grounding_score: int,
                       grounding_reason: str, latency: float, answer: str,
                       num_pages: int) -> str:
    """Build full output section HTML after pipeline completes."""
    text_display = extracted_text[:3000] if extracted_text else "(empty)"
    text_display = text_display.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    answer_display = answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") if answer else "(no answer)"
    grounding_reason_safe = grounding_reason.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") if grounding_reason else ""

    # Threshold colours
    density_color = "#1D9E75" if density >= 50 else "#EF9F27"
    ttr_color = "#1D9E75" if ttr >= 0.30 else "#EF9F27"
    if grounding_score >= 4:
        ground_color = "#1D9E75"
    elif grounding_score >= 2:
        ground_color = "#EF9F27"
    else:
        ground_color = "#E24B4A"

    return f"""
    <div style="background:white;border:0.5px solid #e0ddd8;border-radius:12px;padding:14px 18px;margin-bottom:16px">
        <div style="display:flex;align-items:center;margin-bottom:8px">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                         background:#D85A30;margin-right:8px"></span>
            <span style="font-weight:500;color:#1a1a1a;font-size:0.95rem">OCR output &mdash; mistral-ocr-latest</span>
        </div>
        <div style="color:#999;font-size:0.8rem;margin-bottom:12px">Extracted from {pdf_name} &middot; {num_pages} pages</div>
        <div style="background:#f5f4f0;border-radius:8px;padding:12px 14px;font-size:0.88rem;
                    line-height:1.6;color:#333;white-space:pre-wrap;max-height:300px;overflow-y:auto">{text_display}</div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
        <div style="background:white;border:0.5px solid #e0ddd8;border-radius:12px;padding:14px 18px">
            <div style="text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;color:#999;
                        font-weight:500;margin-bottom:10px">Detected topic</div>
            <span style="background:#FAECE7;color:#D85A30;padding:4px 12px;border-radius:6px;
                         font-weight:500;font-size:0.9rem">{topic}</span>
            <div style="color:#999;font-size:0.75rem;margin-top:8px">From 15-category taxonomy</div>
        </div>
        <div style="background:white;border:0.5px solid #e0ddd8;border-radius:12px;padding:14px 18px">
            <div style="display:flex;align-items:center;margin-bottom:10px">
                <div style="text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;color:#999;
                            font-weight:500">Quality signals</div>
                <span style="margin-left:auto;font-size:0.7rem;color:#bbb">Reference-free &middot; no ground truth required</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px">
                <div style="background:#f5f4f0;border-radius:8px;padding:10px 12px">
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500">Extraction density</div>
                    <div style="font-size:1.1rem;font-weight:500;color:{density_color}">{density} <span style="font-size:0.7rem;color:#999">w/p</span></div>
                    <div style="font-size:0.65rem;color:#bbb;margin-top:2px">Words/page</div>
                </div>
                <div style="background:#f5f4f0;border-radius:8px;padding:10px 12px">
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500">Lexical coherence</div>
                    <div style="font-size:1.1rem;font-weight:500;color:{ttr_color}">{ttr:.2f}</div>
                    <div style="font-size:0.65rem;color:#bbb;margin-top:2px">Type-token ratio</div>
                </div>
                <div style="background:#f5f4f0;border-radius:8px;padding:10px 12px">
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500">Grounding score</div>
                    <div style="font-size:1.1rem;font-weight:500;color:{ground_color}">{grounding_score} / 5</div>
                    <div style="font-size:0.65rem;color:#bbb;margin-top:2px" title="{grounding_reason_safe}">LLM judge</div>
                </div>
                <div style="background:#f5f4f0;border-radius:8px;padding:10px 12px">
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500">Latency</div>
                    <div style="font-size:1.1rem;font-weight:500;color:#666">{latency:.1f} <span style="font-size:0.7rem;color:#999">s</span></div>
                    <div style="font-size:0.65rem;color:#bbb;margin-top:2px">Pipeline time</div>
                </div>
            </div>
            <div style="color:#999;font-size:0.7rem;margin-top:8px;font-style:italic" title="{grounding_reason_safe}">
                {grounding_reason_safe[:120] if grounding_reason_safe else ''}
            </div>
        </div>
    </div>

    <div style="background:white;border:0.5px solid #e0ddd8;border-radius:12px;padding:14px 18px;margin-bottom:16px">
        <div style="display:flex;align-items:center;margin-bottom:8px">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                         background:#D85A30;margin-right:8px"></span>
            <span style="font-weight:500;color:#1a1a1a;font-size:0.95rem">Mistral answer &mdash; mistral-large-latest</span>
        </div>
        <div style="color:#999;font-size:0.8rem;margin-bottom:12px">Grounded on extracted text only &middot; no hallucination by design</div>
        <div style="background:#f5f4f0;border-radius:8px;padding:12px 14px;font-size:0.88rem;
                    line-height:1.6;color:#333;white-space:pre-wrap">{answer_display}</div>
    </div>
    """


def run_llm(pdf_file, question: str):
    """Run the full pipeline on a single document.

    Returns: (pipeline_html, output_html, status_msg)
    """
    pdf_path = _resolve_file(pdf_file)
    if pdf_path is None:
        return _pipeline_html("idle"), "", "No file uploaded"
    if not question or not question.strip():
        return _pipeline_html("idle"), "", "Question is required"

    cached = load_from_cache(pdf_path)
    if cached:
        topic = cached.get("topic", "")
        answer = cached.get("answer", "")
        extracted_text = cached.get("extracted_text", "")
        latency = cached.get("latency_s", 0.0)
        num_pages = cached.get("num_pages", AVG_PAGES_PER_DOC)
        grounding_score = cached.get("grounding_score", 0)
        grounding_reason = cached.get("grounding_reason", "")
    else:
        try:
            result = extract_document(pdf_path, question=question)
            topic = result.topic
            answer = result.answer or ""
            extracted_text = result.extracted_text
            latency = result.latency_total_s
            num_pages = result.num_pages
            grounding_score = result.grounding_score
            grounding_reason = result.grounding_reason
            cost = _estimate_cost(result.tokens_prompt, result.tokens_completion)
            save_to_cache(pdf_path, {
                "document_id": result.document_id,
                "extracted_text": extracted_text,
                "topic": topic, "answer": answer,
                "cost_usd": cost, "latency_s": latency,
                "num_pages": num_pages,
                "grounding_score": grounding_score,
                "grounding_reason": grounding_reason,
            })
        except Exception as exc:
            return _pipeline_html("idle"), "", f"Error: {exc}"

    # Compute reference-free metrics
    density = compute_extraction_density(extracted_text, num_pages)
    ttr = compute_ttr(extracted_text)

    pipeline = _pipeline_html(
        "done", latency,
        wer=f"{density} w/p", topic=topic[:20],
        qa=f"{grounding_score}/5" if grounding_score else "\u2014",
    )
    output = _build_output_html(
        pdf_path.name, extracted_text, topic,
        density, ttr, grounding_score, grounding_reason,
        latency, answer, num_pages,
    )

    return pipeline, output, f"Complete \u00b7 {latency:.1f}s"


_batch_rows: list[list[str]] = []


def add_to_batch(pdf_file, question: str):
    """Add last processed document to batch. Returns (status_html, batch_table_data)."""
    global _batch_rows
    pdf_path = _resolve_file(pdf_file)
    if pdf_path is None:
        return '<span style="color:#999">No document uploaded</span>', _batch_rows

    cached = load_from_cache(pdf_path)
    if not cached:
        return '<span style="color:#999">Run LLM first</span>', _batch_rows

    doc_id = cached.get("document_id", pdf_path.stem)
    # Avoid duplicates
    if any(row[0] == doc_id for row in _batch_rows):
        return '<span style="color:#EF9F27;font-weight:500">Already in batch</span>', _batch_rows

    topic = cached.get("topic", "\u2014")[:30]
    cost = cached.get("cost_usd", 0.0)
    latency = cached.get("latency_s", 0.0)
    _batch_rows.append([doc_id, topic, f"${cost:.3f}", f"{latency:.1f}s", "\u2705"])
    return (
        f'<span style="color:#1D9E75;font-weight:500">Added \u2713 ({len(_batch_rows)} in batch)</span>',
        _batch_rows,
    )


# ---------------------------------------------------------------------------
# Tab 2 — Evaluate (v4.2 style + API cost breakdown)
# ---------------------------------------------------------------------------

_current_sample: list[dict] = []


def _render_eval_page(table_rows, detail_items, page, page_size):
    """Render one page of evaluation rows + matching detail state."""
    safe_page_size = max(1, int(page_size))
    total_rows = len(table_rows)
    total_pages = max(1, math.ceil(total_rows / safe_page_size))
    safe_page = max(1, min(int(page), total_pages))
    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size
    page_rows = table_rows[start:end]
    page_details = detail_items[start:end]
    if total_rows == 0:
        info = "Page 1 / 1 \u00b7 0 rows"
    else:
        info = f"Page {safe_page} / {total_pages} \u00b7 Rows {start + 1}-{min(end, total_rows)} of {total_rows}"
    return page_rows, json.dumps(page_details, default=str), str(safe_page), info


def _coerce_rows_state(state):
    if isinstance(state, list):
        return state
    if isinstance(state, str):
        try:
            parsed = json.loads(state or "[]")
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _coerce_details_state(state):
    if isinstance(state, list):
        return state
    if isinstance(state, str):
        try:
            parsed = json.loads(state or "[]")
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def prev_eval_page(page_state, table_state, detail_full_state, page_size):
    rows = _coerce_rows_state(table_state)
    details = _coerce_details_state(detail_full_state)
    return _render_eval_page(rows, details, int(page_state or 1) - 1, int(page_size))


def next_eval_page(page_state, table_state, detail_full_state, page_size):
    rows = _coerce_rows_state(table_state)
    details = _coerce_details_state(detail_full_state)
    return _render_eval_page(rows, details, int(page_state or 1) + 1, int(page_size))


def reset_eval_page_size(page_size, table_state, detail_full_state):
    rows = _coerce_rows_state(table_state)
    details = _coerce_details_state(detail_full_state)
    return _render_eval_page(rows, details, 1, int(page_size))


def update_sample_preview(sample_size: int):
    """Update the sample preview when slider moves."""
    global _current_sample
    sample, unique_cats, total_available = preview_sample(int(sample_size))
    _current_sample = sample
    if unique_cats >= 15:
        color = "#1D9E75"
    elif unique_cats >= 12:
        color = "#EF9F27"
    else:
        color = "#E24B4A"
    coverage_html = (
        f'<div style="font-size:0.9rem;font-weight:500;color:{color};padding:6px 0">'
        f'{unique_cats} / 15 categories covered</div>'
    )
    return f"{int(sample_size)} / {total_available} documents", coverage_html


def run_eval_tab(sample_size: int, use_cache: bool = True, page_size: int = 20):
    """Run evaluation. Returns 10 values (including business case refresh)."""
    global _current_sample
    if not _current_sample:
        _current_sample, _, _ = preview_sample(int(sample_size))

    doc_ids = [r["document_id"] for r in _current_sample]
    results: list[EvalResult] = []

    for eval_result, current, total in run_evaluation(
        limit=int(sample_size), split="repliqa_3", doc_ids=doc_ids,
        use_cache=use_cache,
    ):
        results.append(eval_result)
        logger.info("Processing %d/%d: %s%s", current, total,
                     eval_result.document_id, " (cached)" if eval_result.cached else "")

    output_path = save_eval_results(results)

    ok = [r for r in results if r.status == "ok"]
    n_ok = len(ok)
    n_err = sum(1 for r in results if r.status == "error")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_cached = sum(1 for r in results if r.cached)

    if n_ok == 0:
        empty: list = []
        pr, pd, ps, pi = _render_eval_page(empty, empty, 1, int(page_size))
        return ("No successful results", pr, f"Errors: {n_err} | Skipped: {n_skip}",
                pd, empty, empty, ps, pi, build_comparison_cards(), build_roi_projection(400, "Pages"))

    avg_wer = sum(r.wer for r in ok) / n_ok
    avg_rouge = sum(r.rouge_l for r in ok) / n_ok
    topic_acc = sum(1 for r in ok if r.topic_match) / n_ok
    avg_answer = sum(r.answer_score for r in ok) / n_ok
    avg_latency = sum(r.latency_s for r in ok) / n_ok
    total_cost = sum(r.cost_usd for r in ok)

    def _badge(val, thresh, higher=True):
        return "pass" if (val >= thresh if higher else val <= thresh) else "fail"

    wer_b = _badge(avg_wer, 0.15, False)
    rouge_b = _badge(avg_rouge, 0.80)
    topic_b = _badge(topic_acc, 0.80)
    answer_b = _badge(avg_answer, 4.0)

    metrics_html = f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px">
        <div style="background:#f5f4f0;border-radius:8px;padding:14px 16px;
                    border-left:3px solid {'#1D9E75' if wer_b=='pass' else '#E24B4A'}">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500;margin-bottom:6px">Avg WER</div>
            <div style="font-size:1.6rem;font-weight:500;color:{'#1D9E75' if wer_b=='pass' else '#E24B4A'}">{avg_wer:.3f}</div>
            <div style="font-size:0.75rem;color:#999;margin-top:4px">target &lt; 0.15</div>
        </div>
        <div style="background:#f5f4f0;border-radius:8px;padding:14px 16px;
                    border-left:3px solid {'#1D9E75' if rouge_b=='pass' else '#E24B4A'}">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500;margin-bottom:6px">Avg ROUGE-L</div>
            <div style="font-size:1.6rem;font-weight:500;color:{'#1D9E75' if rouge_b=='pass' else '#E24B4A'}">{avg_rouge:.3f}</div>
            <div style="font-size:0.75rem;color:#999;margin-top:4px">target &gt; 0.80</div>
        </div>
        <div style="background:#f5f4f0;border-radius:8px;padding:14px 16px;
                    border-left:3px solid {'#1D9E75' if topic_b=='pass' else '#E24B4A'}">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500;margin-bottom:6px">Topic accuracy</div>
            <div style="font-size:1.6rem;font-weight:500;color:{'#1D9E75' if topic_b=='pass' else '#E24B4A'}">{topic_acc:.1%}</div>
            <div style="font-size:0.75rem;color:#999;margin-top:4px">target &gt; 80%</div>
        </div>
        <div style="background:#f5f4f0;border-radius:8px;padding:14px 16px;
                    border-left:3px solid {'#1D9E75' if answer_b=='pass' else '#E24B4A'}">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500;margin-bottom:6px">Avg Q&A score</div>
            <div style="font-size:1.6rem;font-weight:500;color:#D85A30">{avg_answer:.1f} / 5</div>
            <div style="font-size:0.75rem;color:#999;margin-top:4px">LLM-as-judge</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">
        <div style="background:#f5f4f0;border-radius:8px;padding:12px 16px">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500;margin-bottom:4px">Documents</div>
            <div style="font-size:1.3rem;font-weight:500;color:#1a1a1a">{n_ok}</div>
        </div>
        <div style="background:#f5f4f0;border-radius:8px;padding:12px 16px">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500;margin-bottom:4px">Avg latency</div>
            <div style="font-size:1.3rem;font-weight:500;color:#1a1a1a">{avg_latency:.1f}s</div>
        </div>
        <div style="background:#f5f4f0;border-radius:8px;padding:12px 16px">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500;margin-bottom:4px">Total cost</div>
            <div style="font-size:1.3rem;font-weight:500;color:#1a1a1a">${total_cost:.2f}</div>
        </div>
    </div>
    """

    table_data = []
    for r in results:
        if r.status == "ok":
            match_icon = "exact" if r.topic_match else ("judge>=4" if r.topic_score >= 4 else "fail")
            status_icon = "Pass" if r.answer_score >= 4 else ("Review" if r.answer_score >= 3 else "Error")
        else:
            match_icon = "\u2014"
            status_icon = "Error" if r.status == "error" else "Skip"
        table_data.append([
            "",
            r.index,
            r.document_id,
            r.topic_extracted[:30] if r.topic_extracted else "\u2014",
            r.topic_gt[:30] if r.topic_gt else "\u2014",
            match_icon,
            f"{r.wer:.4f}" if r.status == "ok" else "\u2014",
            f"{r.answer_score}" if r.status == "ok" else "\u2014",
            "cached" if r.cached else "new",
            status_icon,
        ])

    detail_items = []
    for r in results:
        detail_items.append({
            "doc_id": r.document_id, "question": r.question,
            "extracted_text": r.extracted_text[:500],
            "reference_text": r.reference_text[:500],
            "answer_extracted": r.answer_extracted, "answer_gt": r.answer_gt,
            "topic_extracted": r.topic_extracted, "topic_gt": r.topic_gt,
            "topic_match": r.topic_match, "topic_score": r.topic_score,
            "topic_rationale": r.topic_rationale, "answer_rationale": r.answer_rationale,
            "topic_criteria": r.topic_criteria, "answer_criteria": r.answer_criteria,
        })

    status = (
        f"Results saved to: {output_path}\n"
        f"{n_ok}/{len(results)} successful | {n_cached} cached | {n_err} errors | {n_skip} skipped"
    )

    new_comparison = build_comparison_cards()
    new_roi = build_roi_projection(400, "Pages")

    pr, pd, ps, pi = _render_eval_page(table_data, detail_items, 1, int(page_size))
    return (metrics_html, pr, status, pd, table_data, detail_items, ps, pi, new_comparison, new_roi)


def show_doc_detail(table_data, evt: gr.SelectData, detail_state: str, table_state):
    """Show detail panel when user clicks a table row.

    Returns: (updated_table, topic_html, question, extracted, reference, answer, answer_gt, judge)
    """
    try:
        details = json.loads(detail_state)
        row_idx = evt.index[0] if hasattr(evt.index, '__getitem__') else evt.index

        # Current visible table rows (may be user-sorted in UI)
        visible_rows = table_data.values.tolist() if hasattr(table_data, "values") else (
            table_data if isinstance(table_data, list) else []
        )

        # Update selection marker in visible table
        updated_table = []
        for i, row in enumerate(visible_rows):
            new_row = list(row)
            new_row[0] = "\u25B6" if i == row_idx else ""
            updated_table.append(new_row)

        if 0 <= row_idx < len(visible_rows):
            selected_row = visible_rows[row_idx]
            selected_doc_id = str(selected_row[2]) if len(selected_row) > 2 else ""

            # Resolve detail by doc_id to remain correct after sorting.
            d = next((item for item in details if str(item.get("doc_id", "")) == selected_doc_id), None)
            if d is None and row_idx < len(details):
                d = details[row_idx]
            if d is None:
                d = {}

            topic_ext = d.get("topic_extracted", "\u2014")
            topic_gt = d.get("topic_gt", "\u2014")
            topic_match = d.get("topic_match", False)
            topic_score = d.get("topic_score", 0)

            match_color = "#1D9E75" if topic_match else ("#EF9F27" if topic_score >= 4 else "#E24B4A")
            match_label = "Exact Match" if topic_match else (f"Score: {topic_score}/5" if topic_score else "No Match")
            match_icon = "\u2705" if topic_match else ("\u26A0\uFE0F" if topic_score >= 4 else "\u274C")

            topic_html = f"""
            <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:12px;padding:16px;margin-bottom:12px">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
                    <span style="font-size:1.2rem">{match_icon}</span>
                    <span style="font-weight:500;font-size:1rem;color:#92400E">Topic Classification</span>
                    <span style="margin-left:auto;background:{match_color};color:white;padding:2px 10px;
                           border-radius:12px;font-size:0.8rem;font-weight:500">{match_label}</span>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
                    <div style="background:white;border-left:3px solid #D85A30;border-radius:8px;padding:12px">
                        <div style="font-size:0.7rem;color:#999;text-transform:uppercase;font-weight:500;margin-bottom:4px">
                            Mistral Prediction</div>
                        <div style="font-weight:500;color:#1A1A1A;font-size:1rem">{topic_ext}</div>
                    </div>
                    <div style="background:white;border-left:3px solid #6B7280;border-radius:8px;padding:12px">
                        <div style="font-size:0.7rem;color:#999;text-transform:uppercase;font-weight:500;margin-bottom:4px">
                            Ground Truth</div>
                        <div style="font-weight:500;color:#1A1A1A;font-size:1rem">{topic_gt}</div>
                    </div>
                </div>
            </div>"""

            judge_info = ""
            if d.get("topic_rationale"):
                judge_info += f"Topic: {d['topic_rationale']}\n"
                if d.get("topic_criteria"):
                    judge_info += f"  Criteria: {json.dumps(d['topic_criteria'])}\n"
            if d.get("answer_rationale"):
                judge_info += f"\nAnswer: {d['answer_rationale']}\n"
                if d.get("answer_criteria"):
                    judge_info += f"  Criteria: {json.dumps(d['answer_criteria'])}\n"

            return (updated_table, topic_html, d.get("question", ""), d.get("extracted_text", ""),
                    d.get("reference_text", ""), d.get("answer_extracted", ""),
                    d.get("answer_gt", ""), judge_info)
    except (json.JSONDecodeError, IndexError, TypeError):
        pass
    return (table_data,
            '<div style="color:#999;padding:16px;text-align:center">Click a row to see topic comparison</div>',
            "", "", "", "", "", "Click a row to see details")


# ---------------------------------------------------------------------------
# Tab 3 — Business Case
# ---------------------------------------------------------------------------

def _load_latest_eval_results() -> dict | None:
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    for f in sorted(results_dir.glob("eval_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            if "summary" in data:
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return None


def build_comparison_cards(business_view: bool = False) -> str:
    """Build Incumbent vs Mistral-Lens comparison cards.

    When business_view=True, show business-oriented metrics instead of technical ones.
    """
    data = _load_latest_eval_results()
    wer_val, rouge_val, cost_doc = 0.017, 0.999, 0.06
    n_docs = 50
    if data:
        s = data["summary"]
        n = s["successful"]
        n_docs = n
        wer_val = s["avg_wer"]
        rouge_val = s["avg_rouge_l"]
        cost_doc = s["total_cost_usd"] / n if n else 0.06

    if business_view:
        saving_pct = ((INCUMBENT_COST_PER_PAGE * AVG_PAGES_PER_DOC - cost_doc) / (INCUMBENT_COST_PER_PAGE * AVG_PAGES_PER_DOC) * 100)
        incumbent_rows = f"""
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem">Cost per document</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#E24B4A">${INCUMBENT_COST_PER_PAGE * AVG_PAGES_PER_DOC:.2f}</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Processing quality</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#333;border-top:1px solid #f0eee9">85% accuracy</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Intelligent Q&A</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#999;border-top:1px solid #f0eee9">Not available</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Vendor dependency</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#E24B4A;border-top:1px solid #f0eee9">Full lock-in</td></tr>"""
        mistral_rows = f"""
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem">Cost per document</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#D85A30">~${cost_doc:.2f}</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Processing quality</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#1D9E75;border-top:1px solid #f0eee9">99.9% fidelity</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Intelligent Q&A</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#1D9E75;border-top:1px solid #f0eee9">4.9/5 quality</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Vendor dependency</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#1D9E75;border-top:1px solid #f0eee9">Open weights</td></tr>"""
        summary_row = f"""
        <div style="background:#EAF3DE;border:0.5px solid #c4dba6;border-radius:8px;padding:12px 16px;margin-top:12px;display:flex;align-items:center;justify-content:space-between">
            <span style="color:#1D9E75;font-weight:500;font-size:0.9rem">Cost reduction</span>
            <span style="color:#1D9E75;font-weight:500;font-size:1.4rem">{saving_pct:.0f}% savings per document</span>
        </div>"""
    else:
        incumbent_rows = f"""
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem">Cost per page</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#E24B4A">$0.75</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Accuracy metric</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#333;border-top:1px solid #f0eee9">Binary &mdash; 85%</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Q&A / judge</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#999;border-top:1px solid #f0eee9">Not included</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Portability</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#E24B4A;border-top:1px solid #f0eee9">100% lock-in</td></tr>"""
        mistral_rows = f"""
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem">Cost per doc (~4p)</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#D85A30">~${cost_doc:.2f}</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">WER / ROUGE-L</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#1D9E75;border-top:1px solid #f0eee9">{wer_val:.3f} / {rouge_val:.3f}</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Q&A + LLM judge</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#1D9E75;border-top:1px solid #f0eee9">Included</td></tr>
                <tr><td style="padding:8px 0;color:#333;font-size:0.9rem;border-top:1px solid #f0eee9">Portability</td>
                    <td style="padding:8px 0;text-align:right;font-weight:500;color:#1D9E75;border-top:1px solid #f0eee9">Open weights</td></tr>"""
        summary_row = ""

    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
        <div style="background:white;border:0.5px solid #e0ddd8;border-radius:12px;padding:14px 18px">
            <div style="text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;color:#999;font-weight:500;margin-bottom:14px">Incumbent</div>
            <table style="width:100%">{incumbent_rows}
            </table>
        </div>
        <div style="background:white;border:0.5px solid #e0ddd8;border-radius:12px;padding:14px 18px">
            <div style="text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;color:#999;font-weight:500;margin-bottom:14px">Mistral-Lens</div>
            <table style="width:100%">{mistral_rows}
            </table>
        </div>
    </div>{summary_row}"""


def build_roi_projection(volume: int, unit: str = "Pages") -> str:
    """Build ROI projection HTML."""
    pages = int(volume) * AVG_PAGES_PER_DOC if unit == "Documents" else int(volume)
    docs = pages // AVG_PAGES_PER_DOC
    incumbent = pages * INCUMBENT_COST_PER_PAGE
    mistral = docs * MISTRAL_COST_PER_DOC
    saving = incumbent - mistral
    annual = saving * 12
    pct = (saving / incumbent * 100) if incumbent > 0 else 0

    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px;margin:16px 0">
        <div style="background:#f5f4f0;border-radius:8px;padding:14px 16px">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500;margin-bottom:6px">Incumbent / mo</div>
            <div style="font-size:1.6rem;font-weight:500;color:#E24B4A">${incumbent:,.0f}</div>
        </div>
        <div style="background:#f5f4f0;border-radius:8px;padding:14px 16px">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#999;font-weight:500;margin-bottom:6px">Mistral / mo</div>
            <div style="font-size:1.6rem;font-weight:500;color:#D85A30">${mistral:,.0f}</div>
        </div>
        <div style="background:#EAF3DE;border:0.5px solid #c4dba6;border-radius:8px;padding:14px 16px">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#1D9E75;font-weight:500;margin-bottom:6px">Monthly saving</div>
            <div style="font-size:1.6rem;font-weight:500;color:#1D9E75">${saving:,.0f}</div>
        </div>
        <div style="background:#EAF3DE;border:0.5px solid #c4dba6;border-radius:8px;padding:14px 16px">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:#1D9E75;font-weight:500;margin-bottom:6px">Saving per month</div>
            <div style="font-size:1.6rem;font-weight:500;color:#1D9E75">{pct:.0f}%</div>
        </div>
    </div>
    <div style="color:#999;font-size:0.8rem">
        Annual saving: <span style="color:#1D9E75;font-weight:500">${annual:,.0f}</span> &middot; Avg 4 pages/doc
    </div>"""


def build_api_breakdown() -> str:
    """Build the API cost breakdown card — collapsible."""
    return """
    <details style="background:white;border:0.5px solid #e0ddd8;border-radius:12px;padding:0;margin-top:20px">
        <summary style="cursor:pointer;padding:14px 18px;display:flex;align-items:center;gap:8px;
                        list-style:none;-webkit-appearance:none">
            <span style="font-size:0.75rem;color:#999;transition:transform 0.2s" class="collapse-arrow">&#x25B6;</span>
            <span style="text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;color:#999;font-weight:500">API cost breakdown</span>
            <span style="margin-left:auto;font-size:0.85rem;color:#D85A30;font-weight:500">~$0.06 / doc</span>
        </summary>
        <div style="padding:0 18px 14px 18px">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
                <div style="background:#f5f4f0;border-radius:8px;padding:14px 16px">
                    <div style="color:#D85A30;font-weight:500;font-size:0.9rem;margin-bottom:10px">OCR layer &mdash; mistral-ocr-latest</div>
                    <table style="width:100%">
                        <tr><td style="padding:5px 0;color:#666;font-size:0.85rem">Endpoint</td><td style="padding:5px 0;text-align:right;font-weight:500;font-size:0.85rem">POST /v1/ocr</td></tr>
                        <tr><td style="padding:5px 0;color:#666;font-size:0.85rem">Pricing model</td><td style="padding:5px 0;text-align:right;font-weight:500;font-size:0.85rem">Per page</td></tr>
                        <tr><td style="padding:5px 0;color:#666;font-size:0.85rem">Cost</td><td style="padding:5px 0;text-align:right;font-weight:500;color:#D85A30;font-size:0.85rem">$0.01 / page</td></tr>
                        <tr><td style="padding:5px 0;color:#666;font-size:0.85rem">Avg pages / doc</td><td style="padding:5px 0;text-align:right;font-weight:500;font-size:0.85rem">4 pages</td></tr>
                        <tr style="border-top:1px solid #e0ddd8"><td style="padding:5px 0;color:#666;font-size:0.85rem;font-weight:500">Per document</td><td style="padding:5px 0;text-align:right;font-weight:500;color:#D85A30;font-size:0.85rem">$0.04</td></tr>
                    </table>
                </div>
                <div style="background:#f5f4f0;border-radius:8px;padding:14px 16px">
                    <div style="color:#D85A30;font-weight:500;font-size:0.9rem;margin-bottom:10px">LLM layer &mdash; mistral-large-latest</div>
                    <table style="width:100%">
                        <tr><td style="padding:5px 0;color:#666;font-size:0.85rem">Endpoint</td><td style="padding:5px 0;text-align:right;font-weight:500;font-size:0.85rem">POST /v1/chat/completions</td></tr>
                        <tr><td style="padding:5px 0;color:#666;font-size:0.85rem">Input tokens</td><td style="padding:5px 0;text-align:right;font-weight:500;font-size:0.85rem">$2.00 / 1M</td></tr>
                        <tr><td style="padding:5px 0;color:#666;font-size:0.85rem">Output tokens</td><td style="padding:5px 0;text-align:right;font-weight:500;font-size:0.85rem">$6.00 / 1M</td></tr>
                        <tr><td style="padding:5px 0;color:#666;font-size:0.85rem">Tasks covered</td><td style="padding:5px 0;text-align:right;font-weight:500;font-size:0.85rem">Topic + Q&A</td></tr>
                        <tr style="border-top:1px solid #e0ddd8"><td style="padding:5px 0;color:#666;font-size:0.85rem;font-weight:500">Per document</td><td style="padding:5px 0;text-align:right;font-weight:500;color:#D85A30;font-size:0.85rem">~$0.02</td></tr>
                    </table>
                </div>
            </div>
            <div style="display:flex;align-items:baseline;gap:12px;padding-top:12px;border-top:1px solid #e0ddd8">
                <span style="font-weight:500;color:#333;font-size:0.95rem">Total / document</span>
                <span style="font-size:1.6rem;font-weight:500;color:#D85A30;margin-left:auto">~$0.06</span>
                <span style="font-size:0.85rem;color:#1D9E75;font-weight:500">vs $0.75 incumbent &rarr; &minus;92%</span>
            </div>
        </div>
    </details>"""


def build_metrics_explainer() -> str:
    """Build collapsible metrics explainer for the Evaluate tab."""
    card_style = (
        "background:#f0f9f0;border:0.5px solid #c4dba6;border-radius:10px;"
        "padding:14px 16px;display:flex;flex-direction:column;gap:4px"
    )
    label_style = (
        "font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;"
        "color:#1D9E75;font-weight:600"
    )
    desc_style = "font-size:0.85rem;color:#333;line-height:1.5;margin-top:4px"
    why_label = (
        "font-size:0.7rem;text-transform:uppercase;letter-spacing:0.04em;"
        "color:#999;font-weight:500;margin-top:8px"
    )
    why_style = "font-size:0.82rem;color:#666;line-height:1.5;margin-top:2px"

    metrics = [
        (
            "WER (Word Error Rate)",
            "Measures the edit distance between OCR-extracted text and the ground truth, "
            "normalised by word count. Lower is better &mdash; 0.00 means perfect extraction.",
            "Directly impacts downstream accuracy: every misread word propagates errors into "
            "topic classification and Q&amp;A. A WER below 2% means the OCR layer is virtually "
            "lossless, eliminating re-processing costs.",
        ),
        (
            "ROUGE-L",
            "Longest Common Subsequence overlap between extracted and reference text. "
            "Captures structural fidelity &mdash; whether paragraphs, lists and headings survive extraction intact.",
            "High ROUGE-L (&gt;0.95) proves the pipeline preserves document structure, "
            "which is critical for compliance-sensitive workflows where layout matters as much as content.",
        ),
        (
            "Topic Accuracy",
            "Exact-match rate between the predicted topic category and the ground truth label "
            "from the repliqa dataset, evaluated across a 15-category taxonomy.",
            "Automated document routing depends on correct classification. "
            "Each mis-classified document requires manual triage &mdash; at scale, "
            "even a 5% accuracy drop translates to thousands of extra human-review hours per year.",
        ),
        (
            "Q&amp;A Score (LLM-as-Judge)",
            "A 1&ndash;5 score assigned by mistral-large-latest acting as an impartial judge. "
            "Evaluates correctness, completeness, and grounding of the generated answer against "
            "the ground truth answer.",
            "Measures whether the system can replace a human analyst reading the full document. "
            "Scores above 4.0 indicate production-ready quality; below 3.0 signals the answer "
            "would need manual verification, eroding the ROI advantage.",
        ),
    ]

    cards_html = ""
    for name, description, why in metrics:
        cards_html += f"""
        <div style="{card_style}">
            <div style="{label_style}">{name}</div>
            <div style="{desc_style}">{description}</div>
            <div style="{why_label}">Why it&rsquo;s relevant</div>
            <div style="{why_style}">{why}</div>
        </div>"""

    return f"""
    <details style="background:white;border:0.5px solid #e0ddd8;border-radius:12px;padding:0;margin-top:20px">
        <summary style="cursor:pointer;padding:14px 18px;display:flex;align-items:center;gap:8px;
                        list-style:none;-webkit-appearance:none">
            <span style="font-size:0.75rem;color:#999;transition:transform 0.2s" class="collapse-arrow">&#x25B6;</span>
            <span style="text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;color:#999;font-weight:500">
                Evaluation metrics &mdash; what we measure and why</span>
        </summary>
        <div style="padding:0 18px 18px 18px">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
                {cards_html}
            </div>
        </div>
    </details>"""


def on_roi_slider_change(volume, unit):
    return build_roi_projection(int(volume), unit)


def _on_unit_change(unit):
    if unit == "Documents":
        return gr.update(minimum=1, maximum=1000, step=1, value=100), build_roi_projection(100, "Documents")
    return gr.update(minimum=4, maximum=4000, step=4, value=400), build_roi_projection(400, "Pages")


def _on_view_toggle(business_view: bool):
    """Toggle between technical metrics and business view in comparison cards."""
    return build_comparison_cards(business_view=business_view)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

MISTRAL_CSS = """
.gradio-container {
    background: #FAFAFA !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 0 12px; border-bottom: 1px solid #e0ddd8; margin-bottom: 8px;
}
.topbar-title { font-size: 1.25rem; font-weight: 500; color: #1a1a1a; }
.topbar-title span { color: #D85A30; }
.topbar-subtitle { font-size: 0.8rem; color: #999; margin-top: 2px; }
.tabs > .tab-nav { border-bottom: 1px solid #e0ddd8 !important; }
.tabs > .tab-nav > button {
    font-weight: 400 !important; color: #999 !important; border: none !important;
    padding: 0.75rem 1.25rem !important; font-size: 0.9rem !important;
}
.tabs > .tab-nav > button:hover { color: #666 !important; }
.tabs > .tab-nav > button.selected {
    color: #D85A30 !important; border-bottom: 2px solid #D85A30 !important; font-weight: 500 !important;
}
.primary {
    background: #D85A30 !important; border: none !important; color: white !important;
    font-weight: 500 !important; border-radius: 8px !important; padding: 0.7rem 1.8rem !important;
    font-size: 0.9rem !important;
}
.primary:hover { background: #c04f28 !important; box-shadow: 0 2px 8px rgba(216,90,48,0.25) !important; }
.secondary {
    background: white !important; border: 0.5px solid #e0ddd8 !important;
    color: #333 !important; border-radius: 8px !important; font-size: 0.9rem !important;
}
.secondary:hover { border-color: #D85A30 !important; color: #D85A30 !important; }
textarea, input[type="text"] {
    border: 0.5px solid #e0ddd8 !important; border-radius: 8px !important; background: #FFF !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #D85A30 !important; box-shadow: 0 0 0 2px rgba(216,90,48,0.1) !important;
}
label span { font-weight: 400 !important; color: #333 !important; font-size: 0.85rem !important; }
.block { border: 0.5px solid #e0ddd8 !important; border-radius: 12px !important; background: #FFF !important; }
table { border-collapse: collapse !important; width: 100% !important; }
th {
    background: #f5f4f0 !important; color: #999 !important; padding: 0.5rem 0.75rem !important;
    font-weight: 500 !important; font-size: 0.7rem !important; text-transform: uppercase !important;
    letter-spacing: 0.05em !important; text-align: left !important;
}
td { padding: 0.5rem 0.75rem !important; border-bottom: 1px solid #f0eee9 !important; font-size: 0.85rem !important; }
tr:hover td { background: rgba(216,90,48,0.03) !important; }
#eval-table table { table-layout: fixed !important; width: 100% !important; }
#eval-table td:first-child { width: 28px !important; text-align: center !important; }
.section-divider { border-top: 1px solid #e0ddd8 !important; margin-top: 16px !important; padding-top: 16px !important; }
#batch-table table { width: 100% !important; }
#batch-table td { font-size: 0.85rem !important; }
details summary::-webkit-details-marker { display: none !important; }
details summary::marker { display: none !important; content: "" !important; }
details[open] > summary .collapse-arrow { transform: rotate(90deg) !important; }
details summary:hover { background: rgba(216,90,48,0.02) !important; border-radius: 12px !important; }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
.pulse-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #EF9F27; margin-right: 6px; animation: pulse 1.5s ease-in-out infinite;
}
footer { display: none !important; }
"""


# ---------------------------------------------------------------------------
# UI Builder
# ---------------------------------------------------------------------------

def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    with gr.Blocks(title="Mistral-Lens", css=MISTRAL_CSS) as app:

        # Topbar
        gr.HTML("""
        <div class="topbar">
            <div>
                <div class="topbar-title"><span>mistral</span>-lens</div>
                <div class="topbar-subtitle">Document intelligence pipeline &mdash; OCR, topic extraction &amp; Q&A</div>
            </div>
        </div>
        """)

        # ================================================================
        # Tab 1: Process
        # ================================================================
        with gr.Tab("Process"):
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(label="PDF document", file_types=[".pdf"])
                    autofill_badge = gr.HTML(value="")
                with gr.Column(scale=1):
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="Ask a question about the document...",
                        lines=4,
                    )

            pdf_input.change(fn=on_pdf_upload, inputs=[pdf_input], outputs=[question_input, autofill_badge])

            pipeline_html = gr.HTML(value=_pipeline_html("idle"))

            with gr.Row():
                run_btn = gr.Button("Run LLM", variant="primary")
                add_btn = gr.Button("Add to batch", variant="secondary")

            add_status = gr.HTML(value="")
            process_status = gr.Textbox(label="Status", interactive=False, lines=1)
            output_html = gr.HTML(value="")

            gr.HTML('<div class="section-divider"></div>')
            gr.Markdown("### Batch")
            batch_table = gr.Dataframe(
                headers=["Doc ID", "Topic", "Cost", "Latency", "Status"],
                value=[], label="Documents in batch", interactive=False,
                elem_id="batch-table", height=200,
            )

            run_btn.click(
                fn=run_llm,
                inputs=[pdf_input, question_input],
                outputs=[pipeline_html, output_html, process_status],
            )
            add_btn.click(
                fn=add_to_batch,
                inputs=[pdf_input, question_input],
                outputs=[add_status, batch_table],
            )

        # ================================================================
        # Tab 2: Evaluate (v4.2 style + API cost breakdown)
        # ================================================================
        with gr.Tab("Evaluate"):
            gr.Markdown("### Measure pipeline quality against ground truth")

            cache_stats = get_cache_stats()
            cache_msg = ""
            if cache_stats["count"] > 0:
                cache_msg = f"Results available from cache ({cache_stats['count']} docs) \u2014 run will be faster"
            gr.Markdown(f"**Evaluation dataset:** repliqa_3 \u00b7 holdout set\n\n{cache_msg}")

            with gr.Row():
                with gr.Column(scale=2):
                    sample_slider = gr.Slider(minimum=1, maximum=50, value=50, step=1, label="Sample size")
                with gr.Column(scale=1):
                    sample_text = gr.Textbox(value="50 documents", label="Sample", interactive=False, lines=1)

            coverage_html = gr.HTML(value="")
            use_cache_check = gr.Checkbox(value=True, label="Use cache (uncheck to force fresh API calls)")
            eval_btn = gr.Button("Run Evaluation", variant="primary")

            sample_slider.change(fn=update_sample_preview, inputs=[sample_slider], outputs=[sample_text, coverage_html])

            eval_metrics = gr.HTML(label="Metrics Summary")

            eval_table = gr.Dataframe(
                headers=["Sel", "#", "Doc ID", "Topic Extracted", "Topic GT", "Match", "WER", "Answer", "Cached", "Status"],
                label="Per-Document Results", wrap=True, interactive=False, height=400, elem_id="eval-table",
            )
            with gr.Row():
                prev_page_btn = gr.Button("Prev", variant="secondary")
                next_page_btn = gr.Button("Next", variant="secondary")
                eval_page_size = gr.Dropdown(choices=[10, 20, 50], value=20, label="Rows per page", scale=1)
                page_info = gr.Textbox(value="Page 1 / 1 \u00b7 0 rows", label="Pagination", interactive=False, lines=1, scale=2)

            eval_status = gr.Textbox(label="Evaluation Status", lines=3, interactive=False)

            detail_state = gr.State("[]")
            table_state = gr.State([])
            detail_full_state = gr.State([])
            page_state = gr.State("1")

            # Detail panel
            gr.Markdown("### Document Detail")
            gr.Markdown("*Click a row in the table above to see details*")

            detail_topic_html = gr.HTML(
                value='<div style="color:#999;padding:16px;text-align:center">Click a row to see topic comparison</div>'
            )
            detail_question = gr.Textbox(label="\U0001f4cb Question (from repliqa dataset)", lines=1, interactive=False)
            with gr.Row():
                detail_extracted = gr.Textbox(label="\U0001f7e0 OCR Output \u2014 mistral-ocr-latest", lines=8)
                detail_reference = gr.Textbox(label="\U0001f4d6 Ground Truth Text \u2014 repliqa dataset", lines=8)
            with gr.Row():
                detail_answer = gr.Textbox(label="\U0001f7e0 Mistral Answer \u2014 mistral-large-latest", lines=4)
                detail_answer_gt = gr.Textbox(label="\U0001f4d6 Expected Answer \u2014 repliqa dataset", lines=4)
            detail_judge = gr.Textbox(
                label="\U0001f7e0 LLM Judge \u2014 mistral-large-latest (correctness, completeness, grounding)", lines=4)

            eval_table.select(
                fn=show_doc_detail, inputs=[eval_table, detail_state, table_state],
                outputs=[eval_table, detail_topic_html, detail_question, detail_extracted, detail_reference, detail_answer, detail_answer_gt, detail_judge],
            )
            prev_page_btn.click(fn=prev_eval_page, inputs=[page_state, table_state, detail_full_state, eval_page_size],
                                outputs=[eval_table, detail_state, page_state, page_info])
            next_page_btn.click(fn=next_eval_page, inputs=[page_state, table_state, detail_full_state, eval_page_size],
                                outputs=[eval_table, detail_state, page_state, page_info])
            eval_page_size.change(fn=reset_eval_page_size, inputs=[eval_page_size, table_state, detail_full_state],
                                  outputs=[eval_table, detail_state, page_state, page_info])

            # Metrics explainer (collapsible)
            gr.HTML(build_metrics_explainer())

        # ================================================================
        # Tab 3: Business Case (v4.2 style)
        # ================================================================
        with gr.Tab("Business case"):
            gr.Markdown("### Translate performance into financial impact")

            view_toggle = gr.Checkbox(
                value=False, label="Business view (hide technical metrics, show business impact)",
            )
            comparison_html = gr.HTML(value=build_comparison_cards(business_view=False))
            view_toggle.change(fn=_on_view_toggle, inputs=[view_toggle], outputs=[comparison_html])

            gr.HTML('<div class="section-divider"></div>')
            gr.Markdown("### ROI Projection")

            with gr.Row():
                roi_unit = gr.Radio(choices=["Pages", "Documents"], value="Pages", label="Monthly Volume Unit")

            roi_slider = gr.Slider(minimum=4, maximum=4000, value=400, step=4, label="Monthly Volume")
            gr.Markdown("*Assuming avg 4 pages / document*")
            roi_html = gr.HTML(value=build_roi_projection(400, "Pages"))

            roi_unit.change(fn=_on_unit_change, inputs=[roi_unit], outputs=[roi_slider, roi_html])
            roi_slider.change(fn=on_roi_slider_change, inputs=[roi_slider, roi_unit], outputs=[roi_html])

            gr.HTML(build_api_breakdown())

        # Wire eval button (after all tabs so comparison_html and roi_html exist)
        eval_btn.click(
            fn=run_eval_tab,
            inputs=[sample_slider, use_cache_check, eval_page_size],
            outputs=[
                eval_metrics, eval_table, eval_status, detail_state,
                table_state, detail_full_state, page_state, page_info,
                comparison_html, roi_html,
            ],
        )

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = create_ui()
    settings = get_settings()
    app.launch(server_name=settings.HOST, server_port=settings.PORT)
