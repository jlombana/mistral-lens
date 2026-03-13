"""Extractor module — 3-step pipeline: OCR → Topic → Q&A using Mistral models.

Step 1: PDF → raw text via mistral-ocr-latest (/v1/ocr)
Step 2: Raw text → topic summary via mistral-large-latest
Step 3: Raw text + question → long-form answer via mistral-large-latest
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pydantic import BaseModel

from app.config import get_settings
from app.prompts import QA_EXTRACTION_PROMPT, TOPIC_EXTRACTION_PROMPT
from app.retry import retry
from app.utils import pdf_to_base64, timestamp_now

# Load category taxonomy and optional fine-tuned model at import time
_CATEGORY_LIST_PATH = Path(__file__).resolve().parent.parent / "data" / "category_list.txt"
_FINETUNED_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "finetuned_model.txt"

TOPIC_CATEGORIES: str | None = None
if _CATEGORY_LIST_PATH.exists():
    TOPIC_CATEGORIES = _CATEGORY_LIST_PATH.read_text().strip()

FINETUNED_TOPIC_MODEL: str | None = None
if _FINETUNED_MODEL_PATH.exists():
    FINETUNED_TOPIC_MODEL = _FINETUNED_MODEL_PATH.read_text().strip()

logger = logging.getLogger(__name__)


class ExtractionResult(BaseModel):
    """Result of the full extraction pipeline for a single document."""

    document_id: str
    timestamp: str
    ocr_model: str
    chat_model: str
    extracted_text: str
    topic: str
    answer: str | None = None
    question: str | None = None
    latency_ocr_s: float = 0.0
    latency_topic_s: float = 0.0
    latency_answer_s: float = 0.0
    latency_total_s: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0


@retry(max_retries=10, base_delay=2.0, max_delay=60.0)
def _call_ocr(pdf_base64: str, model: str, api_key: str) -> str:
    """Call Mistral OCR API to extract text from a PDF.

    Args:
        pdf_base64: Base64-encoded PDF content.
        model: OCR model identifier.
        api_key: Mistral API key.

    Returns:
        Extracted text string.
    """
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    response = client.ocr.process(
        model=model,
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_base64}",
        },
    )

    pages_text = []
    for page in response.pages:
        pages_text.append(page.markdown)
    return "\n\n".join(pages_text)


class _ChatResult:
    """Container for chat response text + token usage."""

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


@retry(max_retries=10, base_delay=2.0, max_delay=60.0)
def _call_chat(prompt: str, model: str, api_key: str) -> _ChatResult:
    """Call Mistral chat API with a prompt.

    Args:
        prompt: The prompt text.
        model: Chat model identifier.
        api_key: Mistral API key.

    Returns:
        _ChatResult with text and token counts.
    """
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    return _ChatResult(
        text=response.choices[0].message.content,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF using Mistral OCR.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text string.
    """
    settings = get_settings()
    pdf_base64 = pdf_to_base64(pdf_path)
    logger.info("OCR extracting text from %s", pdf_path.name)
    return _call_ocr(pdf_base64, settings.OCR_MODEL, settings.MISTRAL_API_KEY)


def extract_topic(text: str) -> str:
    """Extract topic from document text using taxonomy classification.

    Uses a fixed category taxonomy when available (from data/category_list.txt).
    Uses fine-tuned model if available (from data/finetuned_model.txt),
    otherwise falls back to mistral-large-latest.

    Args:
        text: Raw text extracted from the document.

    Returns:
        Topic category string.
    """
    settings = get_settings()
    model = FINETUNED_TOPIC_MODEL or settings.CHAT_MODEL

    if TOPIC_CATEGORIES:
        prompt = TOPIC_EXTRACTION_PROMPT.format(
            categories=TOPIC_CATEGORIES, text=text[:2000]
        )
    else:
        from app.prompts import TOPIC_EXTRACTION_PROMPT_OPEN
        prompt = TOPIC_EXTRACTION_PROMPT_OPEN.format(text=text)

    logger.info("Extracting topic (model=%s, taxonomy=%s)", model, bool(TOPIC_CATEGORIES))
    result = _call_chat(prompt, model, settings.MISTRAL_API_KEY)
    return result.text


def extract_answer(text: str, question: str) -> str:
    """Extract answer to a question from document text.

    Args:
        text: Raw text extracted from the document.
        question: Question to answer.

    Returns:
        Answer string (2-5 sentences).
    """
    settings = get_settings()
    prompt = QA_EXTRACTION_PROMPT.format(text=text, question=question)
    logger.info("Extracting answer for question: %s", question[:50])
    result = _call_chat(prompt, settings.CHAT_MODEL, settings.MISTRAL_API_KEY)
    return result.text


def extract_document(
    pdf_path: Path,
    question: str | None = None,
    document_id: str | None = None,
) -> ExtractionResult:
    """Run the full 3-step extraction pipeline on a single document.

    Args:
        pdf_path: Path to the PDF file.
        question: Optional question for Q&A extraction.
        document_id: Optional document identifier (defaults to filename).

    Returns:
        ExtractionResult with all extracted fields.
    """
    settings = get_settings()
    doc_id = document_id or pdf_path.stem
    total_prompt_tokens = 0
    total_completion_tokens = 0

    logger.info("Processing document: %s", doc_id)

    # Step 1: OCR
    t0 = time.perf_counter()
    pdf_base64 = pdf_to_base64(pdf_path)
    extracted_text = _call_ocr(pdf_base64, settings.OCR_MODEL, settings.MISTRAL_API_KEY)
    latency_ocr = time.perf_counter() - t0

    # Step 2: Topic (taxonomy classification)
    t0 = time.perf_counter()
    topic_model = FINETUNED_TOPIC_MODEL or settings.CHAT_MODEL
    if TOPIC_CATEGORIES:
        topic_prompt = TOPIC_EXTRACTION_PROMPT.format(
            categories=TOPIC_CATEGORIES, text=extracted_text[:2000]
        )
    else:
        from app.prompts import TOPIC_EXTRACTION_PROMPT_OPEN
        topic_prompt = TOPIC_EXTRACTION_PROMPT_OPEN.format(text=extracted_text)
    topic_result = _call_chat(topic_prompt, topic_model, settings.MISTRAL_API_KEY)
    latency_topic = time.perf_counter() - t0
    total_prompt_tokens += topic_result.prompt_tokens
    total_completion_tokens += topic_result.completion_tokens

    # Step 3: Q&A (optional)
    answer = None
    latency_answer = 0.0
    if question:
        t0 = time.perf_counter()
        qa_prompt = QA_EXTRACTION_PROMPT.format(text=extracted_text, question=question)
        qa_result = _call_chat(qa_prompt, settings.CHAT_MODEL, settings.MISTRAL_API_KEY)
        latency_answer = time.perf_counter() - t0
        answer = qa_result.text
        total_prompt_tokens += qa_result.prompt_tokens
        total_completion_tokens += qa_result.completion_tokens

    latency_total = latency_ocr + latency_topic + latency_answer

    return ExtractionResult(
        document_id=doc_id,
        timestamp=timestamp_now(),
        ocr_model=settings.OCR_MODEL,
        chat_model=settings.CHAT_MODEL,
        extracted_text=extracted_text,
        topic=topic_result.text,
        answer=answer,
        question=question,
        latency_ocr_s=round(latency_ocr, 3),
        latency_topic_s=round(latency_topic, 3),
        latency_answer_s=round(latency_answer, 3),
        latency_total_s=round(latency_total, 3),
        tokens_prompt=total_prompt_tokens,
        tokens_completion=total_completion_tokens,
        tokens_total=total_prompt_tokens + total_completion_tokens,
    )


def extract_batch(
    pdf_paths: list[Path],
    questions: list[str] | None = None,
) -> list[ExtractionResult]:
    """Run extraction pipeline on multiple documents.

    Args:
        pdf_paths: List of PDF file paths.
        questions: Optional list of questions (one per document).

    Returns:
        List of ExtractionResult objects.
    """
    results = []
    for i, path in enumerate(pdf_paths):
        question = questions[i] if questions and i < len(questions) else None
        logger.info("Processing document %d/%d: %s", i + 1, len(pdf_paths), path.name)
        result = extract_document(path, question=question)
        results.append(result)
    return results
