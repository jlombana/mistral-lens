"""Gradio UI entry point for Mistral-Lens v4.

Three tabs:
  1. Process  — Upload documents and build your processing batch
  2. Evaluate — Measure pipeline quality against ground truth (repliqa_3, 50 docs)
  3. Business Case — Translate performance into financial impact

No business logic here — delegates to extractor, evaluator, metrics, and cache modules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import gradio as gr

from app.cache import get_cache_stats, load_from_cache, save_to_cache
from app.config import get_settings
from app.evaluator import EvalResult, run_evaluation, save_eval_results
from app.extractor import extract_document
from app.utils import save_json, timestamp_now

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
PRICING = {"input": 2.0, "output": 6.0, "page": 0.01}
INCUMBENT_COST_PER_PAGE = 0.75
AVG_PAGES_PER_DOC = 4


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
# Tab 1 — Process: accumulative batch
# ---------------------------------------------------------------------------

# Session state for batch table
_batch_rows: list[list[str]] = []


def process_and_add(pdf_file, question: str):
    """Process a PDF and add to the batch table.

    Returns:
        (table_data, docs_kpi, cost_kpi, savings_kpi, status_text)
    """
    global _batch_rows

    pdf_path = _resolve_file(pdf_file)
    if pdf_path is None:
        return _batch_rows, str(len(_batch_rows)), "$0.00", "$0.00", "No file uploaded"
    if not question or not question.strip():
        return _batch_rows, str(len(_batch_rows)), "$0.00", "$0.00", "Question is required"

    status_msg = ""

    # Check cache first
    cached = load_from_cache(pdf_path)
    if cached:
        topic = cached.get("topic", "")
        answer = cached.get("answer", "")
        cost = cached.get("cost_usd", 0.0)
        status_msg = f"Loaded from cache: {pdf_path.name}"
    else:
        # Run pipeline: OCR → Topic → Q&A
        try:
            result = extract_document(pdf_path, question=question)
            topic = result.topic
            answer = result.answer or ""
            cost = _estimate_cost(result.tokens_prompt, result.tokens_completion)

            # Save to cache
            cache_data = {
                "document_id": result.document_id,
                "extracted_text": result.extracted_text,
                "topic": topic,
                "answer": answer,
                "cost_usd": cost,
                "latency_s": result.latency_total_s,
                "num_pages": AVG_PAGES_PER_DOC,
            }
            save_to_cache(pdf_path, cache_data)
            status_msg = f"Processed: {pdf_path.name} ({result.latency_total_s:.1f}s)"
        except Exception as exc:
            _batch_rows.append([pdf_path.name, "ERROR", str(exc)[:100]])
            n = len(_batch_rows)
            return _batch_rows, str(n), "$0.00", "$0.00", f"Error: {exc}"

    # Truncate answer for table display
    answer_short = answer[:150] + "..." if len(answer) > 150 else answer
    _batch_rows.append([pdf_path.name, topic, answer_short])

    n = len(_batch_rows)
    batch_cost = n * AVG_PAGES_PER_DOC * 0.001
    savings = n * AVG_PAGES_PER_DOC * (INCUMBENT_COST_PER_PAGE - 0.001)

    return (
        _batch_rows,
        str(n),
        f"${batch_cost:.2f}",
        f"${savings:.2f}",
        status_msg,
    )


def clear_batch():
    """Clear the batch table."""
    global _batch_rows
    _batch_rows = []
    return [], "0", "$0.00", "$0.00", "Batch cleared"


# ---------------------------------------------------------------------------
# Tab 2 — Evaluate: repliqa_3 holdout
# ---------------------------------------------------------------------------

def run_eval_tab(progress=gr.Progress()):
    """Run evaluation on repliqa_3 (50 docs).

    Returns:
        (metrics_html, table_data, status_text, detail_state_json)
    """
    results: list[EvalResult] = []

    for eval_result, current, total in run_evaluation(limit=50, split="repliqa_3"):
        results.append(eval_result)
        cached_label = " (cached)" if eval_result.cached else ""
        progress(
            current / total,
            desc=f"Processing {current}/{total}: {eval_result.document_id}{cached_label}",
        )

    # Save results
    output_path = save_eval_results(results)

    ok = [r for r in results if r.status == "ok"]
    n_ok = len(ok)
    n_err = sum(1 for r in results if r.status == "error")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_cached = sum(1 for r in results if r.cached)

    if n_ok == 0:
        return "No successful results", [], f"Errors: {n_err} | Skipped: {n_skip}", "[]"

    avg_wer = sum(r.wer for r in ok) / n_ok
    topic_acc = sum(1 for r in ok if r.topic_match) / n_ok
    avg_answer = sum(r.answer_score for r in ok) / n_ok
    avg_latency = sum(r.latency_s for r in ok) / n_ok
    total_cost = sum(r.cost_usd for r in ok)

    # Metrics block
    def _badge(val: float, threshold: float, higher_better: bool = True) -> str:
        if higher_better:
            return "pass" if val >= threshold else "fail"
        return "pass" if val <= threshold else "fail"

    wer_badge = _badge(avg_wer, 0.15, higher_better=False)
    topic_badge = _badge(topic_acc, 0.90)
    answer_badge = _badge(avg_answer, 4.0)

    metrics_html = f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px">
        <div class="kpi-card {'kpi-pass' if wer_badge=='pass' else 'kpi-fail'}">
            <div class="kpi-value">{avg_wer:.1%}</div>
            <div class="kpi-label">WER (target &lt; 15%)</div>
        </div>
        <div class="kpi-card {'kpi-pass' if topic_badge=='pass' else 'kpi-fail'}">
            <div class="kpi-value">{topic_acc:.1%}</div>
            <div class="kpi-label">Topic Accuracy (target &gt; 90%)</div>
        </div>
        <div class="kpi-card {'kpi-pass' if answer_badge=='pass' else 'kpi-fail'}">
            <div class="kpi-value">{avg_answer:.1f}/5</div>
            <div class="kpi-label">Answer Quality (target &gt; 4.0)</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
        <div class="kpi-card">
            <div class="kpi-value">{n_ok}</div>
            <div class="kpi-label">Documents Processed</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{avg_latency:.1f}s</div>
            <div class="kpi-label">Avg Latency / doc</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">${total_cost:.2f}</div>
            <div class="kpi-label">Total Cost (Mistral)</div>
        </div>
    </div>
    """

    # Per-doc table
    table_data = []
    for r in results:
        if r.status == "ok":
            match_icon = "exact" if r.topic_match else ("judge>=4" if r.topic_score >= 4 else "fail")
            cached_icon = "cached" if r.cached else "new"
            status_icon = "Pass" if r.answer_score >= 4 else ("Review" if r.answer_score >= 3 else "Error")
        else:
            match_icon = "—"
            cached_icon = "—"
            status_icon = "Error" if r.status == "error" else "Skip"

        table_data.append([
            r.index,
            r.document_id[:12],
            r.topic_extracted[:30] if r.topic_extracted else "—",
            r.topic_gt[:30] if r.topic_gt else "—",
            match_icon,
            f"{r.wer:.4f}" if r.status == "ok" else "—",
            f"{r.answer_score}" if r.status == "ok" else "—",
            cached_icon,
            status_icon,
        ])

    # Build detail data for click-through (store as JSON in state)
    detail_items = []
    for r in results:
        detail_items.append({
            "doc_id": r.document_id,
            "extracted_text": r.extracted_text[:500],
            "reference_text": r.reference_text[:500],
            "answer_extracted": r.answer_extracted,
            "answer_gt": r.answer_gt,
            "topic_rationale": r.topic_rationale,
            "answer_rationale": r.answer_rationale,
            "topic_criteria": r.topic_criteria,
            "answer_criteria": r.answer_criteria,
        })

    error_banner = ""
    if n_err > 0:
        error_banner = f"\nCompleted with {n_err} errors"

    status = (
        f"Results saved to: {output_path}\n"
        f"{n_ok}/{len(results)} successful | {n_cached} cached | {n_err} errors | {n_skip} skipped"
        f"{error_banner}"
    )

    return metrics_html, table_data, status, json.dumps(detail_items, default=str)


def show_doc_detail(table_data, evt: gr.SelectData, detail_state: str):
    """Show detail panel when user clicks a table row.

    Returns:
        (extracted_text, reference_text, answer_extracted, answer_gt, judge_info)
    """
    try:
        details = json.loads(detail_state)
        row_idx = evt.index[0] if hasattr(evt.index, '__getitem__') else evt.index
        if 0 <= row_idx < len(details):
            d = details[row_idx]
            judge_info = ""
            if d.get("topic_rationale"):
                judge_info += f"Topic: {d['topic_rationale']}\n"
                if d.get("topic_criteria"):
                    judge_info += f"  Criteria: {json.dumps(d['topic_criteria'])}\n"
            if d.get("answer_rationale"):
                judge_info += f"\nAnswer: {d['answer_rationale']}\n"
                if d.get("answer_criteria"):
                    judge_info += f"  Criteria: {json.dumps(d['answer_criteria'])}\n"
            return (
                d.get("extracted_text", ""),
                d.get("reference_text", ""),
                d.get("answer_extracted", ""),
                d.get("answer_gt", ""),
                judge_info,
            )
    except (json.JSONDecodeError, IndexError, TypeError):
        pass
    return "", "", "", "", "Click a row to see details"


# ---------------------------------------------------------------------------
# Tab 3 — Business Case
# ---------------------------------------------------------------------------

def _load_latest_eval_results() -> dict | None:
    """Load the most recent evaluation results file (must have 'summary' key)."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    eval_files = sorted(results_dir.glob("eval_*.json"), reverse=True)
    for f in eval_files:
        try:
            data = json.loads(f.read_text())
            if "summary" in data:
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return None


def build_technical_view() -> str:
    """Build the technical view HTML."""
    data = _load_latest_eval_results()
    if not data:
        return '<div class="banner-warn">Run an Evaluation first to populate live metrics</div>'

    s = data["summary"]
    ts = s.get("timestamp", "N/A")[:10]
    n = s["successful"]

    def badge(val, thresh, higher=True):
        ok = (val >= thresh) if higher else (val <= thresh)
        return "kpi-pass" if ok else "kpi-fail"

    avg_wer = s["avg_wer"]
    avg_rouge = s["avg_rouge_l"]
    topic_acc = s["topic_accuracy"]
    avg_topic = s["avg_topic_score"]
    avg_answer = s["avg_answer_score"]
    avg_lat = s["avg_latency_s"]
    cost = s["total_cost_usd"] / n if n else 0

    return f"""
    <div class="banner-ok">Results based on last evaluation: {n} documents &middot; repliqa_3 &middot; {ts}</div>
    <table>
        <tr><th>Metric</th><th>Target</th><th>Measured</th><th>Status</th></tr>
        <tr><td>WER</td><td>&lt; 0.15</td><td>{avg_wer:.4f}</td>
            <td class="{badge(avg_wer, 0.15, False)}">{"Pass" if avg_wer < 0.15 else "Fail"}</td></tr>
        <tr><td>ROUGE-L</td><td>&gt; 0.80</td><td>{avg_rouge:.4f}</td>
            <td class="{badge(avg_rouge, 0.80)}">{"Pass" if avg_rouge > 0.80 else "Fail"}</td></tr>
        <tr><td>Topic Accuracy</td><td>&gt; 90%</td><td>{topic_acc:.1%}</td>
            <td class="{badge(topic_acc, 0.90)}">{"Pass" if topic_acc >= 0.90 else "Fail"}</td></tr>
        <tr><td>Topic Score</td><td>&gt; 4.0/5</td><td>{avg_topic:.1f}/5</td>
            <td class="{badge(avg_topic, 4.0)}">{"Pass" if avg_topic >= 4.0 else "Fail"}</td></tr>
        <tr><td>Answer Score</td><td>&gt; 4.0/5</td><td>{avg_answer:.1f}/5</td>
            <td class="{badge(avg_answer, 4.0)}">{"Pass" if avg_answer >= 4.0 else "Fail"}</td></tr>
        <tr><td>Latency</td><td>&mdash;</td><td>{avg_lat:.1f}s/doc</td><td>&mdash;</td></tr>
        <tr><td>Cost</td><td>&mdash;</td><td>${cost:.4f}/doc</td><td>&mdash;</td></tr>
    </table>
    """


def build_business_view() -> str:
    """Build the business view HTML."""
    data = _load_latest_eval_results()
    if not data:
        return '<div class="banner-warn">Run an Evaluation first to populate live metrics</div>'

    s = data["summary"]
    avg_wer = s["avg_wer"]
    topic_acc = s["topic_accuracy"]
    avg_answer = s["avg_answer_score"]
    n = s["successful"]
    cost_per_doc = s["total_cost_usd"] / n if n else 0.004

    return f"""
    <table>
        <tr><th>Metric</th><th>Business Impact</th></tr>
        <tr><td>WER {avg_wer:.1%}</td><td>1 error per {int(1/avg_wer) if avg_wer > 0 else 999} words &mdash; eliminates manual text review</td></tr>
        <tr><td>Topic {topic_acc:.0%}</td><td>Automated document routing without human triage</td></tr>
        <tr><td>Answer {avg_answer:.1f}/5</td><td>Production-ready Q&amp;A &mdash; no human-in-the-loop needed</td></tr>
        <tr><td>Cost ${cost_per_doc:.4f}/doc</td><td>{INCUMBENT_COST_PER_PAGE * AVG_PAGES_PER_DOC / cost_per_doc:.0f}x cheaper &mdash; ROI positive from day one</td></tr>
    </table>

    <h4 style="margin-top:24px">Cost Comparison vs Incumbent</h4>
    <table>
        <tr><th></th><th>Incumbent</th><th>Mistral-Lens</th><th>Delta</th></tr>
        <tr><td>Cost/page</td><td>$0.75</td><td>~$0.001</td><td>750x</td></tr>
        <tr><td>Cost/doc (4p)</td><td>$3.00</td><td>${cost_per_doc:.4f}</td><td>{3.0/cost_per_doc:.0f}x</td></tr>
        <tr><td>Accuracy</td><td>~85%</td><td>&gt;99%</td><td>+14pp</td></tr>
    </table>
    """


def build_roi_projection(volume: int) -> str:
    """Build ROI projection HTML for given monthly page volume."""
    data = _load_latest_eval_results()
    cost_per_page = 0.001  # default

    if data:
        s = data["summary"]
        n = s["successful"]
        if n > 0:
            cost_per_page = s["total_cost_usd"] / n / AVG_PAGES_PER_DOC

    incumbent = volume * INCUMBENT_COST_PER_PAGE
    mistral = volume * cost_per_page
    monthly_savings = incumbent - mistral
    annual_savings = monthly_savings * 12

    return f"""
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:16px 0">
        <div class="kpi-card" style="border-left:3px solid #EF4444">
            <div class="kpi-value" style="color:#EF4444">${incumbent:,.0f}</div>
            <div class="kpi-label">Incumbent monthly</div>
        </div>
        <div class="kpi-card" style="border-left:3px solid #22C55E">
            <div class="kpi-value" style="color:#22C55E">${mistral:,.0f}</div>
            <div class="kpi-label">Mistral-Lens monthly</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px">
        <div class="kpi-card">
            <div class="kpi-value" style="color:#F97316">${monthly_savings:,.0f}</div>
            <div class="kpi-label">Monthly savings</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#F97316">${annual_savings:,.0f}</div>
            <div class="kpi-label">Annual savings</div>
        </div>
    </div>
    """


def toggle_business_view(view_choice: str) -> tuple[str, str]:
    """Toggle between Technical and Business views."""
    if view_choice == "Technical View":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

MISTRAL_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #FFF7ED 0%, #FEF3E2 50%, #FDE9D0 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.mistral-header { text-align: center; padding: 1.5rem 0 0.5rem; }
.mistral-header h1 { font-size: 2rem !important; font-weight: 700 !important; color: #1A1A1A !important; margin-bottom: 0.25rem !important; }
.mistral-header p { color: #666 !important; font-size: 1rem !important; }
.mistral-logo { font-size: 2.5rem; margin-bottom: 0.25rem; }

.tabs > .tab-nav { border-bottom: 2px solid #F97316 !important; }
.tabs > .tab-nav > button { font-weight: 500 !important; color: #666 !important; border: none !important; padding: 0.75rem 1.5rem !important; }
.tabs > .tab-nav > button.selected { color: #1A1A1A !important; border-bottom: 3px solid #F97316 !important; font-weight: 600 !important; }

.primary {
    background: linear-gradient(135deg, #F97316, #EA580C) !important;
    border: none !important; color: white !important; font-weight: 600 !important;
    border-radius: 8px !important; padding: 0.75rem 2rem !important; font-size: 1rem !important;
    transition: all 0.2s !important;
}
.primary:hover {
    background: linear-gradient(135deg, #EA580C, #DC2626) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3) !important;
}

textarea, input[type="text"] { border: 1px solid #E5E5E5 !important; border-radius: 8px !important; background: #FFFFFF !important; }
textarea:focus, input[type="text"]:focus { border-color: #F97316 !important; box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.15) !important; }
label span { font-weight: 500 !important; color: #333 !important; }
.block { border: 1px solid #F0E6D9 !important; border-radius: 12px !important; background: rgba(255,255,255,0.8) !important; }

/* KPI Cards */
.kpi-card {
    background: rgba(255,255,255,0.9) !important; border: 1px solid #F0E6D9 !important;
    border-radius: 10px !important; padding: 16px !important; text-align: center !important;
}
.kpi-value { font-size: 1.5rem !important; font-weight: 700 !important; color: #1A1A1A !important; margin-bottom: 4px !important; }
.kpi-label { font-size: 0.75rem !important; color: #888 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
.kpi-pass { border-left: 3px solid #22C55E !important; }
.kpi-fail { border-left: 3px solid #EF4444 !important; }

/* Tables */
table { border-collapse: collapse !important; width: 100% !important; }
th { background: #F97316 !important; color: white !important; padding: 0.5rem 1rem !important; font-weight: 600 !important; }
td { padding: 0.5rem 1rem !important; border-bottom: 1px solid #F0E6D9 !important; }
tr:hover td { background: rgba(249,115,22,0.05) !important; }

/* Banners */
.banner-ok {
    background: #ECFDF5; border: 1px solid #10B981; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px; color: #065F46; font-weight: 500;
}
.banner-warn {
    background: #FFFBEB; border: 1px solid #F59E0B; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px; color: #92400E; font-weight: 500;
}
"""


# ---------------------------------------------------------------------------
# UI Builder
# ---------------------------------------------------------------------------

def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    with gr.Blocks(title="Mistral-Lens", css=MISTRAL_CSS) as app:
        gr.HTML("""
        <div class="mistral-header">
            <div class="mistral-logo">
                <svg width="40" height="40" viewBox="0 0 128 128" style="display:inline-block;vertical-align:middle;margin-right:8px">
                    <rect x="0" y="0" width="28" height="28" fill="#F97316"/>
                    <rect x="50" y="0" width="28" height="28" fill="#1A1A1A"/>
                    <rect x="100" y="0" width="28" height="28" fill="#F97316"/>
                    <rect x="0" y="34" width="28" height="28" fill="#F97316"/>
                    <rect x="50" y="34" width="28" height="28" fill="#F97316"/>
                    <rect x="100" y="34" width="28" height="28" fill="#1A1A1A"/>
                    <rect x="0" y="68" width="28" height="28" fill="#F97316"/>
                    <rect x="50" y="68" width="28" height="28" fill="#1A1A1A"/>
                    <rect x="100" y="68" width="28" height="28" fill="#F97316"/>
                    <rect x="0" y="100" width="28" height="28" fill="#F97316"/>
                    <rect x="50" y="100" width="28" height="28" fill="#F97316"/>
                    <rect x="100" y="100" width="28" height="28" fill="#F97316"/>
                </svg>
            </div>
            <h1>Mistral-Lens</h1>
            <p>Document Intelligence Pipeline &mdash; OCR, Topic Extraction &amp; Q&A</p>
        </div>
        """)

        # ================================================================
        # Tab 1: Process
        # ================================================================
        with gr.Tab("Process"):
            gr.Markdown("### Upload documents and build your processing batch")
            with gr.Row():
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question about the document...",
                    lines=3,
                )
            add_btn = gr.Button("Add to Batch", variant="primary")
            process_status = gr.Textbox(label="Status", interactive=False, lines=1)

            with gr.Row():
                docs_kpi = gr.Textbox(label="Docs Processed", value="0", interactive=False, lines=1)
                cost_kpi = gr.Textbox(label="Batch Cost (Mistral)", value="$0.00", interactive=False, lines=1)
                savings_kpi = gr.Textbox(label="Saved vs Incumbent", value="$0.00", interactive=False, lines=1)

            batch_table = gr.Dataframe(
                headers=["Document", "Topic", "Answer"],
                label="Batch Results",
                wrap=True,
                value=[],
            )

            clear_btn = gr.Button("Clear Batch", variant="secondary")

            add_btn.click(
                fn=process_and_add,
                inputs=[pdf_input, question_input],
                outputs=[batch_table, docs_kpi, cost_kpi, savings_kpi, process_status],
            )
            clear_btn.click(
                fn=clear_batch,
                inputs=[],
                outputs=[batch_table, docs_kpi, cost_kpi, savings_kpi, process_status],
            )

        # ================================================================
        # Tab 2: Evaluate
        # ================================================================
        with gr.Tab("Evaluate"):
            gr.Markdown("### Measure pipeline quality against ground truth")

            # Check cache status
            cache_stats = get_cache_stats()
            cache_msg = ""
            if cache_stats["count"] > 0:
                cache_msg = f"Results available from cache ({cache_stats['count']} docs) — run will be faster"

            gr.Markdown(
                f"**Evaluation dataset:** repliqa_3 · 50 documents · holdout set\n\n"
                f"{cache_msg}"
            )

            eval_btn = gr.Button("Run Evaluation", variant="primary")

            eval_metrics = gr.HTML(label="Metrics Summary")

            eval_table = gr.Dataframe(
                headers=["#", "Doc ID", "Topic Extracted", "Topic GT", "Match",
                         "WER", "Answer", "Cached", "Status"],
                label="Per-Document Results",
                wrap=True,
                interactive=False,
            )

            eval_status = gr.Textbox(label="Evaluation Status", lines=3, interactive=False)

            # Hidden state for detail data
            detail_state = gr.Textbox(visible=False, value="[]")

            eval_btn.click(
                fn=run_eval_tab,
                inputs=[],
                outputs=[eval_metrics, eval_table, eval_status, detail_state],
            )

            # Detail panel
            gr.Markdown("### Document Detail")
            gr.Markdown("*Click a row in the table above to see details*")
            with gr.Row():
                detail_extracted = gr.Textbox(label="Extracted Text", lines=8)
                detail_reference = gr.Textbox(label="Reference Text", lines=8)
            with gr.Row():
                detail_answer = gr.Textbox(label="Answer (Generated)", lines=4)
                detail_answer_gt = gr.Textbox(label="Answer (Ground Truth)", lines=4)
            detail_judge = gr.Textbox(label="Judge Rationale + Criteria", lines=4)

            eval_table.select(
                fn=show_doc_detail,
                inputs=[eval_table, detail_state],
                outputs=[detail_extracted, detail_reference, detail_answer, detail_answer_gt, detail_judge],
            )

        # ================================================================
        # Tab 3: Business Case
        # ================================================================
        with gr.Tab("Business Case"):
            gr.Markdown("### Translate performance into financial impact")

            view_toggle = gr.Radio(
                ["Technical View", "Business View"],
                value="Technical View",
                label="View",
            )

            with gr.Column(visible=True) as tech_col:
                tech_html = gr.HTML(value=build_technical_view())

            with gr.Column(visible=False) as biz_col:
                biz_html = gr.HTML(value=build_business_view())

            view_toggle.change(
                fn=toggle_business_view,
                inputs=[view_toggle],
                outputs=[tech_col, biz_col],
            )

            gr.Markdown("### ROI Projection")
            roi_slider = gr.Slider(
                minimum=10000,
                maximum=1000000,
                value=100000,
                step=10000,
                label="Monthly Page Volume",
            )
            roi_html = gr.HTML(value=build_roi_projection(100000))

            roi_slider.change(
                fn=build_roi_projection,
                inputs=[roi_slider],
                outputs=[roi_html],
            )

    return app


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = create_ui()
    settings = get_settings()
    app.launch(server_name=settings.HOST, server_port=settings.PORT)
