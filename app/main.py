"""Gradio UI entry point for Mistral-Lens.

Three tabs: Upload & Extract, Evaluate, Business Case.
No business logic here — delegates to extractor and metrics modules.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from app.config import get_settings
from app.extractor import extract_document
from app.metrics import compute_text_metrics, judge_answer, judge_topic
from app.utils import save_json, timestamp_now

logger = logging.getLogger(__name__)


def process_upload(pdf_file: str, question: str) -> tuple[str, str, str]:
    """Process an uploaded PDF through the extraction pipeline.

    Args:
        pdf_file: Path to uploaded PDF file.
        question: Optional question for Q&A extraction.

    Returns:
        Tuple of (extracted_text, topic, answer).
    """
    if not pdf_file:
        return "No file uploaded", "", ""

    pdf_path = Path(pdf_file)
    result = extract_document(pdf_path, question=question or None)

    return result.extracted_text, result.topic, result.answer or "No question provided"


def run_evaluation(
    pdf_file: str,
    question: str,
    ref_text: str,
    ref_topic: str,
    ref_answer: str,
) -> str:
    """Run evaluation on a single document against provided ground truth.

    Args:
        pdf_file: Path to uploaded PDF file.
        question: Question for Q&A.
        ref_text: Reference text (ground truth).
        ref_topic: Reference topic (ground truth).
        ref_answer: Reference answer (ground truth).

    Returns:
        Formatted evaluation results string.
    """
    if not pdf_file:
        return "No file uploaded"

    pdf_path = Path(pdf_file)
    result = extract_document(pdf_path, question=question or None)

    lines = []

    # Automated metrics
    if ref_text:
        tm = compute_text_metrics("upload", ref_text, result.extracted_text)
        lines.append(f"WER: {tm.wer:.4f} (target < 0.15)")
        lines.append(f"ROUGE-L: {tm.rouge_l:.4f} (target > 0.80)")
        lines.append("")

    # LLM-as-judge: topic
    if ref_topic and result.topic:
        js = judge_topic("upload", ref_topic, result.topic)
        lines.append(f"Topic Score: {js.score}/5 (target > 4.0)")
        lines.append(f"  Rationale: {js.rationale}")
        lines.append("")

    # LLM-as-judge: answer
    if ref_answer and result.answer:
        ja = judge_answer("upload", question, ref_answer, result.answer)
        lines.append(f"Answer Score: {ja.score}/5 (target > 4.0)")
        lines.append(f"  Rationale: {ja.rationale}")
        lines.append("")

    # Save results
    settings = get_settings()
    results_dir = settings.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"eval_{timestamp_now().replace(':', '-')}.json"
    save_json({"extraction": result.model_dump(), "metrics": "\n".join(lines)}, output_file)
    lines.append(f"Results saved to {output_file}")

    return "\n".join(lines) if lines else "No ground truth provided for evaluation"


def build_business_case() -> str:
    """Generate the business case comparison text.

    Returns:
        Formatted business case string.
    """
    return """## Mistral-Lens Business Case

### Cost Comparison

| Provider | Cost per Page | Accuracy | Latency |
|----------|--------------|----------|---------|
| Incumbent | $0.75/page | ~85% | Variable |
| Mistral (OCR + Large) | ~$0.02/page* | TBD | ~3-5s |

*Estimated based on token pricing. Actual cost depends on document length.

### Key Advantages
- **Cost reduction**: Up to 97% lower cost per page
- **Accuracy**: Competitive with incumbent (target > 85%)
- **Flexibility**: Topic extraction + Q&A in single pipeline
- **No vendor lock-in**: Open-weight models available for fine-tuning

### Assumptions
- Average document: ~2 pages, ~1,500 tokens
- Mistral OCR: $0.01/page (estimated)
- Mistral Large: $0.008/1K input tokens, $0.024/1K output tokens
- Incumbent: $0.75/page flat rate, 85% reported accuracy

> Run evaluation on repliqa_3 holdout set to populate actual metrics.
"""


def create_ui() -> gr.Blocks:
    """Create the Gradio UI with 3 tabs.

    Returns:
        Gradio Blocks application.
    """
    with gr.Blocks(title="Mistral-Lens") as app:
        gr.Markdown("# Mistral-Lens — Document Intelligence Demo")
        gr.Markdown("Extract text, topic, and Q&A from PDF documents using Mistral AI models.")

        with gr.Tab("Upload & Extract"):
            with gr.Row():
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                question_input = gr.Textbox(label="Question (optional)", placeholder="Ask a question about the document...")
            extract_btn = gr.Button("Extract", variant="primary")
            with gr.Row():
                text_output = gr.Textbox(label="Extracted Text", lines=10)
                topic_output = gr.Textbox(label="Topic Summary", lines=3)
                answer_output = gr.Textbox(label="Answer", lines=5)
            extract_btn.click(
                fn=process_upload,
                inputs=[pdf_input, question_input],
                outputs=[text_output, topic_output, answer_output],
            )

        with gr.Tab("Evaluate"):
            gr.Markdown("### Compare extraction against ground truth")
            with gr.Row():
                eval_pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
                eval_question = gr.Textbox(label="Question", placeholder="Question for Q&A evaluation")
            with gr.Row():
                ref_text = gr.Textbox(label="Reference Text (ground truth)", lines=5)
                ref_topic = gr.Textbox(label="Reference Topic (ground truth)", lines=2)
                ref_answer = gr.Textbox(label="Reference Answer (ground truth)", lines=3)
            eval_btn = gr.Button("Run Evaluation", variant="primary")
            eval_output = gr.Textbox(label="Evaluation Results", lines=10)
            eval_btn.click(
                fn=run_evaluation,
                inputs=[eval_pdf, eval_question, ref_text, ref_topic, ref_answer],
                outputs=[eval_output],
            )

        with gr.Tab("Business Case"):
            gr.Markdown(build_business_case())

    return app


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = create_ui()
    settings = get_settings()
    app.launch(server_name=settings.HOST, server_port=settings.PORT)
