"""Extractor module — 3-step pipeline: OCR → Topic → Q&A using Mistral models.

Step 1: PDF → raw text via mistral-ocr-latest (/v1/ocr)
Step 2: Raw text → topic summary via mistral-large-latest
Step 3: Raw text + question → long-form answer via mistral-large-latest
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from app.config import get_settings
from app.prompts import QA_EXTRACTION_PROMPT, TOPIC_EXTRACTION_PROMPT
from app.retry import retry
from app.utils import pdf_to_base64, timestamp_now

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


@retry(max_retries=10, base_delay=2.0, max_delay=60.0)
def _call_chat(prompt: str, model: str, api_key: str) -> str:
    """Call Mistral chat API with a prompt.

    Args:
        prompt: The prompt text.
        model: Chat model identifier.
        api_key: Mistral API key.

    Returns:
        Model response text.
    """
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


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
    """Extract topic summary from document text.

    Args:
        text: Raw text extracted from the document.

    Returns:
        Topic summary string (1-3 sentences).
    """
    settings = get_settings()
    prompt = TOPIC_EXTRACTION_PROMPT.format(text=text)
    logger.info("Extracting topic summary")
    return _call_chat(prompt, settings.CHAT_MODEL, settings.MISTRAL_API_KEY)


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
    return _call_chat(prompt, settings.CHAT_MODEL, settings.MISTRAL_API_KEY)


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

    logger.info("Processing document: %s", doc_id)

    # Step 1: OCR
    extracted_text = extract_text(pdf_path)

    # Step 2: Topic
    topic = extract_topic(extracted_text)

    # Step 3: Q&A (optional)
    answer = None
    if question:
        answer = extract_answer(extracted_text, question)

    return ExtractionResult(
        document_id=doc_id,
        timestamp=timestamp_now(),
        ocr_model=settings.OCR_MODEL,
        chat_model=settings.CHAT_MODEL,
        extracted_text=extracted_text,
        topic=topic,
        answer=answer,
        question=question,
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
