"""Extractor module — 3-step pipeline: OCR → Topic → Q&A using Mistral models.

Step 1: PDF → raw text via mistral-ocr-latest (/v1/ocr)
Step 2: Raw text → topic summary via mistral-large-latest
Step 3: Raw text + question → long-form answer via mistral-large-latest
"""

from __future__ import annotations

import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel

from app.config import get_settings
from app.prompts import QA_EXTRACTION_PROMPT, TOPIC_EXTRACTION_PROMPT
from app.retry import retry
from app.utils import pdf_to_base64, timestamp_now

GROUNDING_PROMPT = """You are a strict grounding evaluator. Your only job is \
to verify whether an answer is fully supported by the provided source text. \
Do not use any external knowledge.

SOURCE TEXT:
{ocr_text}

QUESTION:
{question}

ANSWER:
{answer}

Score the answer from 1 to 5:
5 = every claim in the answer is directly supported by the source text
4 = almost fully supported, minor inference acceptable
3 = partially supported, some claims go beyond the text
2 = mostly unsupported
1 = answer contradicts or ignores the source text

Reply with JSON only: {{"score": <int>, "reason": "<one sentence>"}}"""

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


def _clean_topic(raw: str) -> str:
    """Clean model output and fuzzy-match against known categories.

    Strips whitespace/quotes, then checks for exact match against taxonomy.
    If no exact match, tries case-insensitive containment as a fallback.
    """
    cleaned = raw.strip().strip('"').strip("'").strip()
    cleaned = cleaned.splitlines()[0].strip() if cleaned else cleaned
    cleaned = re.sub(r"^\s*(topic|category)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;!-")
    if not TOPIC_CATEGORIES:
        return cleaned

    categories = [c.strip() for c in TOPIC_CATEGORIES.splitlines() if c.strip()]

    # Exact match (case-insensitive)
    for cat in categories:
        if cleaned.lower() == cat.lower():
            return cat

    # Containment: model returned extra text but category name is in there
    for cat in categories:
        if cat.lower() in cleaned.lower():
            return cat

    # Fuzzy fallback: map near-miss labels to the closest taxonomy category.
    # Helps convert outputs like "Local Politics" -> "Local Politics and Governance".
    best_cat = ""
    best_score = 0.0
    for cat in categories:
        score = SequenceMatcher(None, cleaned.lower(), cat.lower()).ratio()
        if score > best_score:
            best_cat = cat
            best_score = score
    if best_cat and best_score >= 0.72:
        logger.info(
            "Fuzzy-mapped topic '%s' -> '%s' (score=%.2f)",
            cleaned,
            best_cat,
            best_score,
        )
        return best_cat

    logger.warning("Topic '%s' not in taxonomy, returning as-is", cleaned)
    return cleaned


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
    num_pages: int = 4
    grounding_score: int = 0
    grounding_reason: str = ""


class _OcrResult:
    """Container for OCR response text + page count."""

    def __init__(self, text: str, num_pages: int):
        self.text = text
        self.num_pages = num_pages


@retry(max_retries=10, base_delay=2.0, max_delay=60.0)
def _call_ocr(pdf_base64: str, model: str, api_key: str) -> _OcrResult:
    """Call Mistral OCR API to extract text from a PDF.

    Args:
        pdf_base64: Base64-encoded PDF content.
        model: OCR model identifier.
        api_key: Mistral API key.

    Returns:
        _OcrResult with extracted text and page count.
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
    return _OcrResult("\n\n".join(pages_text), len(response.pages))


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
    ocr_result = _call_ocr(pdf_base64, settings.OCR_MODEL, settings.MISTRAL_API_KEY)
    return ocr_result.text


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
            categories=TOPIC_CATEGORIES, text=text[:6000]
        )
    else:
        from app.prompts import TOPIC_EXTRACTION_PROMPT_OPEN
        prompt = TOPIC_EXTRACTION_PROMPT_OPEN.format(text=text)

    logger.info("Extracting topic (model=%s, taxonomy=%s)", model, bool(TOPIC_CATEGORIES))
    result = _call_chat(prompt, model, settings.MISTRAL_API_KEY)
    return _clean_topic(result.text)


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


def compute_grounding_score(ocr_text: str, question: str, answer: str) -> dict:
    """Score whether the answer is grounded in the OCR text (no ground truth needed).

    Args:
        ocr_text: Extracted text from OCR.
        question: The question asked.
        answer: The generated answer.

    Returns:
        Dict with 'score' (int 1-5) and 'reason' (str).
    """
    settings = get_settings()
    prompt = GROUNDING_PROMPT.format(
        ocr_text=ocr_text[:6000], question=question, answer=answer
    )
    from mistralai import Mistral

    @retry(max_retries=10, base_delay=2.0, max_delay=60.0)
    def _call(prompt_text: str) -> dict:
        client = Mistral(api_key=settings.MISTRAL_API_KEY)
        response = client.chat.complete(
            model=settings.CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict grounding evaluator."},
                {"role": "user", "content": prompt_text},
            ],
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return {"score": 0, "reason": "parse_error"}

    try:
        result = _call(prompt)
        return {
            "score": int(result.get("score", 0)),
            "reason": result.get("reason", ""),
        }
    except Exception as exc:
        logger.warning("Grounding score failed: %s", exc)
        return {"score": 0, "reason": str(exc)}


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
    ocr_result = _call_ocr(pdf_base64, settings.OCR_MODEL, settings.MISTRAL_API_KEY)
    extracted_text = ocr_result.text
    num_pages = ocr_result.num_pages
    latency_ocr = time.perf_counter() - t0

    # Step 2: Topic (taxonomy classification)
    t0 = time.perf_counter()
    topic_model = FINETUNED_TOPIC_MODEL or settings.CHAT_MODEL
    if TOPIC_CATEGORIES:
        topic_prompt = TOPIC_EXTRACTION_PROMPT.format(
            categories=TOPIC_CATEGORIES, text=extracted_text[:6000]
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

    # Step 4: Grounding score (reference-free, not counted in pipeline latency)
    grounding = {"score": 0, "reason": ""}
    if answer and question:
        grounding = compute_grounding_score(extracted_text, question, answer)

    return ExtractionResult(
        document_id=doc_id,
        timestamp=timestamp_now(),
        ocr_model=settings.OCR_MODEL,
        chat_model=settings.CHAT_MODEL,
        extracted_text=extracted_text,
        topic=_clean_topic(topic_result.text),
        answer=answer,
        question=question,
        latency_ocr_s=round(latency_ocr, 3),
        latency_topic_s=round(latency_topic, 3),
        latency_answer_s=round(latency_answer, 3),
        latency_total_s=round(latency_total, 3),
        tokens_prompt=total_prompt_tokens,
        tokens_completion=total_completion_tokens,
        tokens_total=total_prompt_tokens + total_completion_tokens,
        num_pages=num_pages,
        grounding_score=grounding["score"],
        grounding_reason=grounding["reason"],
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
