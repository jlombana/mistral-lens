"""Gradio UI entry point for Mistral-Lens v2.

Three tabs:
  1. Process — Upload & extract (production mode)
  2. Evaluate — Single-doc or batch evaluation with real-time progress
  3. Business Case — Dynamic metrics from batch results, ROI slider

No business logic here — delegates to extractor, metrics, batch, and cache modules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import gradio as gr

from app.batch import EVAL_SPLITS_CONFIG, DocResult, run_batch, save_batch_results
from app.cache import get_cache_stats
from app.config import get_settings
from app.extractor import extract_document
from app.metrics import compute_text_metrics, judge_answer, judge_topic
from app.utils import save_json, timestamp_now

logger = logging.getLogger(__name__)

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


PRICING = {"input": 2.0, "output": 6.0, "page": 0.01}


def _estimate_cost(prompt_tokens: int, completion_tokens: int, pages: int = 1) -> float:
    return (
        (prompt_tokens / 1_000_000) * PRICING["input"]
        + (completion_tokens / 1_000_000) * PRICING["output"]
        + pages * PRICING["page"]
    )


# ---------------------------------------------------------------------------
# Tab 1 — Process (production mode)
# ---------------------------------------------------------------------------

def process_upload(pdf_file, question: str) -> tuple[str, str, str, str, str, str, str]:
    """Process an uploaded PDF through the extraction pipeline.

    Returns:
        Tuple of (extracted_text, topic, answer, latency_kpi, tokens_kpi, cost_kpi, perf_detail).
    """
    pdf_path = _resolve_file(pdf_file)
    if pdf_path is None:
        return "No file uploaded", "", "", "—", "—", "—", ""
    result = extract_document(pdf_path, question=question or None)

    cost = _estimate_cost(result.tokens_prompt, result.tokens_completion)

    latency_kpi = f"{result.latency_total_s:.1f}s"
    tokens_kpi = f"{result.tokens_total:,}"
    cost_kpi = f"${cost:.4f}"

    perf_detail = (
        f"OCR: {result.latency_ocr_s:.1f}s | Topic: {result.latency_topic_s:.1f}s | Q&A: {result.latency_answer_s:.1f}s\n"
        f"Prompt: {result.tokens_prompt:,} | Completion: {result.tokens_completion:,}"
    )

    return (
        result.extracted_text,
        result.topic,
        result.answer or "No question provided",
        latency_kpi,
        tokens_kpi,
        cost_kpi,
        perf_detail,
    )


# ---------------------------------------------------------------------------
# Tab 2 — Evaluate
# ---------------------------------------------------------------------------

def run_single_evaluation(
    pdf_file,
    question: str,
    ref_text: str,
    ref_topic: str,
    ref_answer: str,
) -> tuple[str, str, str, str, str, str, str]:
    """Run evaluation on a single document against provided ground truth.

    Returns:
        (wer_kpi, rouge_kpi, topic_score_kpi, answer_score_kpi, detail_text, topic_info, answer_info)
    """
    pdf_path = _resolve_file(pdf_file)
    if pdf_path is None:
        return "—", "—", "—", "—", "No file uploaded", "", ""

    result = extract_document(pdf_path, question=question or None)

    wer_val = "—"
    rouge_val = "—"
    topic_score_val = "—"
    answer_score_val = "—"
    lines = []
    topic_info = ""
    answer_info = ""

    if ref_text:
        tm = compute_text_metrics("upload", ref_text, result.extracted_text)
        wer_val = f"{tm.wer:.4f}"
        rouge_val = f"{tm.rouge_l:.4f}"
        lines.append(f"WER: {tm.wer:.4f} (target < 0.15)")
        lines.append(f"ROUGE-L: {tm.rouge_l:.4f} (target > 0.80)")

    if ref_topic and result.topic:
        js = judge_topic("upload", ref_topic, result.topic)
        topic_score_val = f"{js.score}/5"
        topic_info = f"Predicted: {result.topic}\nReference: {ref_topic}\nScore: {js.score}/5\nRationale: {js.rationale}"
        if js.criteria:
            topic_info += f"\nCriteria: {json.dumps(js.criteria)}"

    if ref_answer and result.answer:
        ja = judge_answer("upload", question, ref_answer, result.answer)
        answer_score_val = f"{ja.score}/5"
        answer_info = f"Predicted: {result.answer}\nReference: {ref_answer}\nScore: {ja.score}/5\nRationale: {ja.rationale}"
        if ja.criteria:
            answer_info += f"\nCriteria: {json.dumps(ja.criteria)}"

    cost = _estimate_cost(result.tokens_prompt, result.tokens_completion)
    lines.append(f"\nLatency: {result.latency_total_s:.1f}s | Tokens: {result.tokens_total:,} | Cost: ${cost:.5f}")

    settings = get_settings()
    results_dir = settings.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"eval_{timestamp_now().replace(':', '-')}.json"
    save_json({"extraction": result.model_dump(), "metrics": "\n".join(lines)}, output_file)
    lines.append(f"Saved to {output_file}")

    return wer_val, rouge_val, topic_score_val, answer_score_val, "\n".join(lines), topic_info, answer_info


def run_batch_evaluation(splits_text: str, progress=gr.Progress()):
    """Run stratified batch evaluation across configured splits.

    Args:
        splits_text: JSON string of splits config, e.g. '{"repliqa_1": 20, "repliqa_2": 20, "repliqa_3": 20}'

    Returns:
        Tuple of (summary_html, results_table_data, detail_text)
    """
    try:
        splits_config = json.loads(splits_text) if splits_text.strip() else EVAL_SPLITS_CONFIG
    except json.JSONDecodeError:
        splits_config = EVAL_SPLITS_CONFIG

    results: list[DocResult] = []

    for doc_result, current, total in run_batch(splits_config):
        results.append(doc_result)
        progress(current / total, desc=f"Processing {current}/{total}: {doc_result.split}/{doc_result.document_id}")

    # Save results
    output_path = save_batch_results(results)

    # Build summary
    ok_results = [r for r in results if r.status == "ok"]
    n_ok = len(ok_results)

    if n_ok == 0:
        return "No successful results", [], "No results to display"

    avg_wer = sum(r.wer for r in ok_results) / n_ok
    avg_rouge = sum(r.rouge_l for r in ok_results) / n_ok
    avg_topic_score = sum(r.topic_score for r in ok_results) / n_ok
    avg_answer_score = sum(r.answer_score for r in ok_results) / n_ok
    topic_acc = sum(1 for r in ok_results if r.topic_match) / n_ok
    avg_latency = sum(r.latency_s for r in ok_results) / n_ok
    total_cost = sum(r.cost_usd for r in ok_results)
    n_cached = sum(1 for r in results if r.cached)
    n_errors = sum(1 for r in results if r.status == "error")
    n_skipped = sum(1 for r in results if r.status == "skipped")

    summary_html = f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">
        <div class="kpi-card">
            <div class="kpi-value">{avg_wer:.4f}</div>
            <div class="kpi-label">Avg WER</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{avg_rouge:.4f}</div>
            <div class="kpi-label">Avg ROUGE-L</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{avg_topic_score:.1f}/5</div>
            <div class="kpi-label">Topic Score</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{avg_answer_score:.1f}/5</div>
            <div class="kpi-label">Answer Score</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">
        <div class="kpi-card">
            <div class="kpi-value">{topic_acc:.1%}</div>
            <div class="kpi-label">Topic Accuracy</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{avg_latency:.1f}s</div>
            <div class="kpi-label">Avg Latency</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">${total_cost:.4f}</div>
            <div class="kpi-label">Total Cost</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{n_ok}/{len(results)}</div>
            <div class="kpi-label">Success ({n_cached} cached)</div>
        </div>
    </div>
    """

    # Build per-doc table data
    table_data = []
    for r in results:
        table_data.append([
            r.index,
            r.split,
            r.document_id[:12],
            r.status,
            f"{r.wer:.4f}" if r.status == "ok" else "—",
            f"{r.rouge_l:.4f}" if r.status == "ok" else "—",
            f"{r.topic_score}" if r.status == "ok" else "—",
            f"{r.answer_score}" if r.status == "ok" else "—",
            "Yes" if r.topic_match else "No",
            f"{r.latency_s:.1f}s" if r.status == "ok" else "—",
            "Yes" if r.cached else "No",
        ])

    detail = f"Results saved to: {output_path}\nErrors: {n_errors} | Skipped: {n_skipped} | Cached: {n_cached}"

    return summary_html, table_data, detail


# ---------------------------------------------------------------------------
# Tab 3 — Business Case
# ---------------------------------------------------------------------------

def _load_latest_batch_results() -> dict | None:
    """Load the most recent batch results file."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    batch_files = sorted(results_dir.glob("batch_*.json"), reverse=True)
    if not batch_files:
        return None
    return json.loads(batch_files[0].read_text())


def build_business_case_dynamic(volume_per_month: int = 10000) -> str:
    """Generate dynamic business case from latest batch results with ROI projection."""
    batch = _load_latest_batch_results()

    if batch:
        s = batch["summary"]
        avg_wer = s["avg_wer"]
        avg_rouge = s["avg_rouge_l"]
        topic_acc = s["topic_accuracy"]
        avg_topic_score = s["avg_topic_score"]
        avg_answer_score = s["avg_answer_score"]
        avg_latency = s["avg_latency_s"]
        total_cost = s["total_cost_usd"]
        n_docs = s["successful"]
        cost_per_doc = total_cost / n_docs if n_docs else 0
        source_label = f"from latest batch ({n_docs} docs, {s.get('timestamp', 'N/A')[:10]})"
    else:
        avg_wer = 0.017
        avg_rouge = 0.999
        topic_acc = 0.933
        avg_topic_score = 4.9
        avg_answer_score = 4.9
        avg_latency = 7.1
        cost_per_doc = 0.004
        n_docs = 15
        source_label = "from hardcoded defaults (repliqa_3, 15 docs)"

    # ROI calculations
    incumbent_cost_per_doc = 3.00
    monthly_cost_mistral = volume_per_month * cost_per_doc
    monthly_cost_incumbent = volume_per_month * incumbent_cost_per_doc
    monthly_savings = monthly_cost_incumbent - monthly_cost_mistral
    annual_savings = monthly_savings * 12
    cost_ratio = incumbent_cost_per_doc / cost_per_doc if cost_per_doc > 0 else 0

    return f"""
    <div class="business-section">
        <h3>Performance Metrics</h3>
        <p style="font-size:0.85rem;color:#888">Data source: {source_label}</p>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0">
            <div class="kpi-card">
                <div class="kpi-value">{avg_wer:.4f}</div>
                <div class="kpi-label">WER (target &lt; 0.15)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{avg_rouge:.4f}</div>
                <div class="kpi-label">ROUGE-L (target &gt; 0.80)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{topic_acc:.1%}</div>
                <div class="kpi-label">Topic Accuracy (target &gt; 80%)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{avg_latency:.1f}s</div>
                <div class="kpi-label">Avg Latency / doc</div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0">
            <div class="kpi-card">
                <div class="kpi-value">{avg_topic_score:.1f}/5</div>
                <div class="kpi-label">Topic Score (LLM judge)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{avg_answer_score:.1f}/5</div>
                <div class="kpi-label">Answer Score (LLM judge)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">${cost_per_doc:.4f}</div>
                <div class="kpi-label">Cost per document</div>
            </div>
        </div>
    </div>

    <div class="business-section">
        <h3>ROI Projection &mdash; {volume_per_month:,} docs/month</h3>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:16px 0">
            <div class="kpi-card" style="border-left:3px solid #EF4444">
                <div class="kpi-value" style="color:#EF4444">${monthly_cost_incumbent:,.0f}</div>
                <div class="kpi-label">Incumbent monthly cost</div>
            </div>
            <div class="kpi-card" style="border-left:3px solid #22C55E">
                <div class="kpi-value" style="color:#22C55E">${monthly_cost_mistral:,.0f}</div>
                <div class="kpi-label">Mistral-Lens monthly cost</div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0">
            <div class="kpi-card">
                <div class="kpi-value" style="color:#F97316">${monthly_savings:,.0f}</div>
                <div class="kpi-label">Monthly savings</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color:#F97316">${annual_savings:,.0f}</div>
                <div class="kpi-label">Annual savings</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color:#F97316">{cost_ratio:.0f}x</div>
                <div class="kpi-label">Cost reduction</div>
            </div>
        </div>
    </div>

    <div class="business-section">
        <h3>Cost Comparison vs Incumbent</h3>
        <table>
            <tr><th>Metric</th><th>Incumbent</th><th>Mistral-Lens</th><th>Improvement</th></tr>
            <tr><td>Cost per page</td><td>$0.75</td><td>~$0.001</td><td><strong>{cost_ratio:.0f}x cheaper</strong></td></tr>
            <tr><td>Cost per document</td><td>$3.00</td><td>${cost_per_doc:.4f}</td><td><strong>{cost_ratio:.0f}x cheaper</strong></td></tr>
            <tr><td>OCR accuracy (WER)</td><td>~15% error</td><td>{avg_wer:.1%} error</td><td><strong>{0.15/avg_wer:.0f}x better</strong></td></tr>
            <tr><td>Text fidelity (ROUGE-L)</td><td>~85%</td><td>{avg_rouge:.1%}</td><td><strong>Near-perfect</strong></td></tr>
            <tr><td>Processing time</td><td>Minutes</td><td>{avg_latency:.0f} seconds</td><td><strong>Real-time</strong></td></tr>
            <tr><td>Q&A capability</td><td>None</td><td>{avg_answer_score:.1f}/5</td><td><strong>New capability</strong></td></tr>
        </table>
    </div>

    <div class="methodology-card">
        <div class="methodology-title">Methodology</div>
        <div class="methodology-block">
            <div class="methodology-detail">Dataset: <strong>ServiceNow/repliqa</strong> &middot; {n_docs} documents</div>
            <div class="methodology-detail">Models: <code>mistral-ocr-latest</code> + <code>mistral-large-latest</code></div>
            <div class="methodology-detail">Topic classification: few-shot taxonomy prompt (17 categories, 4 disambiguating examples)</div>
            <div class="methodology-detail">Incumbent cost: $0.75/page assumption (industry average)</div>
        </div>
    </div>
    """


def update_business_case(volume: int) -> str:
    """Callback for ROI slider change."""
    return build_business_case_dynamic(int(volume))


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

MISTRAL_CSS = """
/* Mistral brand styling */
.gradio-container {
    background: linear-gradient(135deg, #FFF7ED 0%, #FEF3E2 50%, #FDE9D0 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.mistral-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.mistral-header h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1A1A1A !important;
    margin-bottom: 0.25rem !important;
}
.mistral-header p {
    color: #666 !important;
    font-size: 1rem !important;
}
.mistral-logo {
    font-size: 2.5rem;
    margin-bottom: 0.25rem;
}
.tabs > .tab-nav {
    border-bottom: 2px solid #F97316 !important;
}
.tabs > .tab-nav > button {
    font-weight: 500 !important;
    color: #666 !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
}
.tabs > .tab-nav > button.selected {
    color: #1A1A1A !important;
    border-bottom: 3px solid #F97316 !important;
    font-weight: 600 !important;
}
.primary {
    background: linear-gradient(135deg, #F97316, #EA580C) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    transition: all 0.2s !important;
}
.primary:hover {
    background: linear-gradient(135deg, #EA580C, #DC2626) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3) !important;
}
textarea, input[type="text"] {
    border: 1px solid #E5E5E5 !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    font-size: 0.9rem !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #F97316 !important;
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.15) !important;
}
label span {
    font-weight: 500 !important;
    color: #333 !important;
}
.block {
    border: 1px solid #F0E6D9 !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(8px) !important;
}
.file-preview {
    border: 2px dashed #F9A825 !important;
    border-radius: 12px !important;
    background: rgba(249, 168, 37, 0.05) !important;
}
textarea {
    background: #FFFFFF !important;
    color: #1A1A1A !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
}

/* KPI Cards */
.kpi-card {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid #F0E6D9 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    text-align: center !important;
}
.kpi-value {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #1A1A1A !important;
    margin-bottom: 4px !important;
}
.kpi-label {
    font-size: 0.75rem !important;
    color: #888 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Business sections */
.business-section {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid #F0E6D9;
}
.business-section h3 {
    color: #1A1A1A;
    font-weight: 600;
    margin-bottom: 8px;
}

/* Evaluation results */
#eval-output,
#eval-output * {
    color-scheme: light !important;
}
#eval-output textarea,
#eval-output textarea[disabled],
#eval-output textarea:disabled,
#eval-output .scroll-hide {
    background: #FFF8F1 !important;
    color: #1A1A1A !important;
    border: 1px solid #F4CBAA !important;
    -webkit-text-fill-color: #1A1A1A !important;
    opacity: 1 !important;
}

/* Methodology card */
.methodology-card {
    background: #F9F6F2 !important;
    border: 1px solid #E8DFD4 !important;
    border-radius: 10px !important;
    padding: 1.25rem 1.5rem !important;
    margin-top: 1.5rem !important;
    font-size: 0.82rem !important;
    color: #5A5248 !important;
    line-height: 1.6 !important;
}
.methodology-title {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #9A8D7F !important;
    margin-bottom: 0.75rem !important;
    border-bottom: 1px solid #E8DFD4 !important;
    padding-bottom: 0.5rem !important;
}
.methodology-block {
    margin-bottom: 0.75rem !important;
}
.methodology-detail {
    color: #6B6158 !important;
    margin: 0.15rem 0 !important;
}
.methodology-detail code {
    background: #EDE7DF !important;
    padding: 0.1rem 0.35rem !important;
    border-radius: 3px !important;
    font-size: 0.78rem !important;
    color: #5A4F43 !important;
}

/* Tables */
table {
    border-collapse: collapse !important;
    width: 100% !important;
}
th {
    background: #F97316 !important;
    color: white !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
}
td {
    padding: 0.5rem 1rem !important;
    border-bottom: 1px solid #F0E6D9 !important;
}
tr:hover td {
    background: rgba(249, 115, 22, 0.05) !important;
}
"""


# ---------------------------------------------------------------------------
# UI Builder
# ---------------------------------------------------------------------------

def create_ui() -> gr.Blocks:
    """Create the Gradio UI with Mistral-branded theme."""
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

        # ---- Tab 1: Process ----
        with gr.Tab("Process"):
            gr.Markdown("### Upload a PDF to extract text, topic, and Q&A answer")
            with gr.Row():
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                question_input = gr.Textbox(
                    label="Question (optional)",
                    placeholder="Ask a question about the document...",
                    lines=3,
                )
            extract_btn = gr.Button("Run Pipeline", variant="primary")

            with gr.Row():
                latency_kpi = gr.Textbox(label="Latency", interactive=False, lines=1)
                tokens_kpi = gr.Textbox(label="Tokens", interactive=False, lines=1)
                cost_kpi = gr.Textbox(label="Est. Cost", interactive=False, lines=1)

            with gr.Row():
                text_output = gr.Textbox(label="Extracted Text", lines=12)
                with gr.Column():
                    topic_output = gr.Textbox(label="Topic", lines=2)
                    answer_output = gr.Textbox(label="Answer", lines=5)
                    perf_output = gr.Textbox(label="Performance Detail", lines=3)

            extract_btn.click(
                fn=process_upload,
                inputs=[pdf_input, question_input],
                outputs=[text_output, topic_output, answer_output, latency_kpi, tokens_kpi, cost_kpi, perf_output],
            )

        # ---- Tab 2: Evaluate ----
        with gr.Tab("Evaluate"):
            eval_mode = gr.Radio(
                ["Single Document", "Batch Evaluation"],
                value="Single Document",
                label="Evaluation Mode",
            )

            # --- Single doc mode ---
            with gr.Column(visible=True) as single_mode_col:
                gr.Markdown("### Compare extraction against ground truth")
                with gr.Row():
                    eval_pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
                    eval_question = gr.Textbox(label="Question", placeholder="Question for Q&A evaluation")
                with gr.Row():
                    ref_text = gr.Textbox(label="Reference Text (ground truth)", lines=5)
                    ref_topic = gr.Textbox(label="Reference Topic", lines=2)
                    ref_answer = gr.Textbox(label="Reference Answer", lines=3)
                eval_btn = gr.Button("Run Evaluation", variant="primary")

                with gr.Row():
                    s_wer = gr.Textbox(label="WER", interactive=False, lines=1)
                    s_rouge = gr.Textbox(label="ROUGE-L", interactive=False, lines=1)
                    s_topic_score = gr.Textbox(label="Topic Score", interactive=False, lines=1)
                    s_answer_score = gr.Textbox(label="Answer Score", interactive=False, lines=1)

                eval_detail = gr.Textbox(label="Evaluation Details", lines=6, elem_id="eval-output")
                with gr.Row():
                    topic_info = gr.Textbox(label="Topic Judge Detail", lines=4)
                    answer_info = gr.Textbox(label="Answer Judge Detail", lines=4)

                eval_btn.click(
                    fn=run_single_evaluation,
                    inputs=[eval_pdf, eval_question, ref_text, ref_topic, ref_answer],
                    outputs=[s_wer, s_rouge, s_topic_score, s_answer_score, eval_detail, topic_info, answer_info],
                )

            # --- Batch mode ---
            with gr.Column(visible=False) as batch_mode_col:
                gr.Markdown("### Stratified batch evaluation across dataset splits")
                splits_input = gr.Textbox(
                    label="Splits Configuration (JSON)",
                    value=json.dumps(EVAL_SPLITS_CONFIG),
                    lines=2,
                    placeholder='{"repliqa_1": 20, "repliqa_2": 20, "repliqa_3": 20}',
                )
                batch_btn = gr.Button("Run Batch Evaluation", variant="primary")

                batch_summary = gr.HTML(label="Batch Summary")
                batch_table = gr.Dataframe(
                    headers=["#", "Split", "Doc ID", "Status", "WER", "ROUGE-L",
                             "Topic", "Answer", "Topic Match", "Latency", "Cached"],
                    label="Per-Document Results",
                    wrap=True,
                )
                batch_detail = gr.Textbox(label="Batch Details", lines=3)

                batch_btn.click(
                    fn=run_batch_evaluation,
                    inputs=[splits_input],
                    outputs=[batch_summary, batch_table, batch_detail],
                )

            # Toggle visibility based on mode
            def toggle_mode(mode):
                return (
                    gr.update(visible=(mode == "Single Document")),
                    gr.update(visible=(mode == "Batch Evaluation")),
                )

            eval_mode.change(
                fn=toggle_mode,
                inputs=[eval_mode],
                outputs=[single_mode_col, batch_mode_col],
            )

        # ---- Tab 3: Business Case ----
        with gr.Tab("Business Case"):
            gr.Markdown("### Dynamic Business Case — adjust volume to see ROI projection")

            volume_slider = gr.Slider(
                minimum=100,
                maximum=100000,
                value=10000,
                step=100,
                label="Monthly Document Volume",
            )

            business_html = gr.HTML(value=build_business_case_dynamic(10000))

            volume_slider.change(
                fn=update_business_case,
                inputs=[volume_slider],
                outputs=[business_html],
            )

            cache_stats = get_cache_stats()
            gr.Markdown(
                f"**Cache stats:** {cache_stats['count']} documents cached | "
                f"${cache_stats['cost_saved']:.4f} in API costs saved"
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
