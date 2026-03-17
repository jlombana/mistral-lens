"""Metrics engine — WER, ROUGE-L, and LLM-as-judge scoring.

This module is independent of the extractor module.
It accepts text pairs and computes evaluation metrics.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from app.config import get_settings
from app.prompts import LLM_JUDGE_ANSWER_PROMPT, LLM_JUDGE_TOPIC_PROMPT
from app.retry import retry
from app.utils import timestamp_now

logger = logging.getLogger(__name__)


class TextMetrics(BaseModel):
    """Automated text metrics for a single document."""

    document_id: str
    wer: float
    rouge_l: float


class JudgeScore(BaseModel):
    """LLM-as-judge score for a single evaluation."""

    document_id: str
    metric_type: str  # "topic" or "answer"
    score: int  # 1-5
    rationale: str
    criteria: dict = {}  # {correctness, completeness, grounding} each 1-5


class MetricsReport(BaseModel):
    """Complete metrics report for an evaluation run."""

    timestamp: str
    ocr_model: str
    chat_model: str
    total_documents: int
    avg_wer: float
    avg_rouge_l: float
    avg_topic_score: float
    avg_answer_score: float
    avg_topic_accuracy: float = 0.0
    avg_latency_s: float = 0.0
    avg_cost_per_doc_usd: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    text_metrics: list[TextMetrics]
    judge_scores: list[JudgeScore]


def compute_extraction_density(text: str, num_pages: int) -> int:
    """Compute words extracted per page — proxy for OCR completeness.

    Args:
        text: Extracted OCR text.
        num_pages: Number of pages in the document.

    Returns:
        Words per page (integer).
    """
    if not text or num_pages <= 0:
        return 0
    word_count = len(text.split())
    return word_count // num_pages


def compute_ttr(text: str) -> float:
    """Compute Type-Token Ratio (TTR) on text.

    Measures vocabulary diversity — very low TTR signals repetitive
    or garbled OCR output (e.g. repeated artifacts, header loops).

    Args:
        text: Input text.

    Returns:
        TTR value between 0.0 and 1.0.
    """
    if not text:
        return 0.0
    # Case-insensitive, strip punctuation
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_topic_accuracy(predicted: str, reference: str) -> float:
    """Exact match accuracy for topic classification.

    Returns 1.0 if predicted matches reference (case-insensitive, stripped), 0.0 otherwise.
    """
    return float(predicted.strip().lower() == reference.strip().lower())


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis text.

    Args:
        reference: Ground truth text.
        hypothesis: Extracted/predicted text.

    Returns:
        WER score (lower is better, 0.0 = perfect).
    """
    from jiwer import wer
    if not reference or not hypothesis:
        return 1.0
    return wer(reference, hypothesis)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F-measure between reference and hypothesis text.

    Args:
        reference: Ground truth text.
        hypothesis: Extracted/predicted text.

    Returns:
        ROUGE-L F-measure (higher is better, 1.0 = perfect).
    """
    from rouge_score.rouge_scorer import RougeScorer
    if not reference or not hypothesis:
        return 0.0
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def compute_text_metrics(
    document_id: str,
    reference: str,
    hypothesis: str,
) -> TextMetrics:
    """Compute automated text metrics for a single document.

    Args:
        document_id: Document identifier.
        reference: Ground truth text.
        hypothesis: Extracted text.

    Returns:
        TextMetrics with WER and ROUGE-L scores.
    """
    return TextMetrics(
        document_id=document_id,
        wer=compute_wer(reference, hypothesis),
        rouge_l=compute_rouge_l(reference, hypothesis),
    )


@retry(max_retries=10, base_delay=2.0, max_delay=60.0)
def _call_judge(prompt: str, model: str, api_key: str) -> dict:
    """Call Mistral chat API for LLM-as-judge scoring.

    Args:
        prompt: Judge prompt with reference and prediction.
        model: Chat model identifier.
        api_key: Mistral API key.

    Returns:
        Parsed JSON dict with 'score', 'rationale', and 'criteria'.
    """
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        result = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning("Judge JSON parse error: %s", exc)
        return {"score": None, "rationale": "parse_error", "criteria": {}}

    if "criteria" not in result:
        result["criteria"] = {}
    return result


def judge_topic(
    document_id: str,
    reference: str,
    prediction: str,
) -> JudgeScore:
    """Score topic extraction quality using LLM-as-judge.

    Args:
        document_id: Document identifier.
        reference: Ground truth topic.
        prediction: Extracted topic.

    Returns:
        JudgeScore with 1-5 score and rationale.
    """
    settings = get_settings()
    prompt = LLM_JUDGE_TOPIC_PROMPT.format(
        reference=reference, prediction=prediction
    )
    result = _call_judge(prompt, settings.CHAT_MODEL, settings.MISTRAL_API_KEY)
    score_val = result.get("score")
    return JudgeScore(
        document_id=document_id,
        metric_type="topic",
        score=int(score_val) if score_val is not None else 0,
        rationale=result.get("rationale", ""),
        criteria=result.get("criteria", {}),
    )


def judge_answer(
    document_id: str,
    question: str,
    reference: str,
    prediction: str,
) -> JudgeScore:
    """Score Q&A answer quality using LLM-as-judge.

    Args:
        document_id: Document identifier.
        question: The question asked.
        reference: Ground truth answer.
        prediction: Extracted answer.

    Returns:
        JudgeScore with 1-5 score and rationale.
    """
    settings = get_settings()
    prompt = LLM_JUDGE_ANSWER_PROMPT.format(
        question=question, reference=reference, prediction=prediction
    )
    result = _call_judge(prompt, settings.CHAT_MODEL, settings.MISTRAL_API_KEY)
    score_val = result.get("score")
    return JudgeScore(
        document_id=document_id,
        metric_type="answer",
        score=int(score_val) if score_val is not None else 0,
        rationale=result.get("rationale", ""),
        criteria=result.get("criteria", {}),
    )


def compute_metrics(
    extractions: list[dict],
    ground_truth: list[dict],
) -> MetricsReport:
    """Compute all metrics for a batch of extractions vs ground truth.

    Args:
        extractions: List of extraction dicts with document_id, extracted_text, topic, answer.
        ground_truth: List of ground truth dicts with document_id, text, topic, answer, question.

    Returns:
        MetricsReport with all automated and judge scores.
    """
    gt_by_id = {item["document_id"]: item for item in ground_truth}

    text_metrics_list = []
    judge_scores_list = []
    topic_accuracies = []

    for extraction in extractions:
        doc_id = extraction["document_id"]
        gt = gt_by_id.get(doc_id)
        if gt is None:
            logger.warning("No ground truth for document %s, skipping", doc_id)
            continue

        # Automated text metrics
        tm = compute_text_metrics(
            document_id=doc_id,
            reference=gt.get("text", ""),
            hypothesis=extraction.get("extracted_text", ""),
        )
        text_metrics_list.append(tm)

        # Topic: exact match accuracy + LLM-as-judge
        if gt.get("topic") and extraction.get("topic"):
            topic_acc = compute_topic_accuracy(extraction["topic"], gt["topic"])
            topic_accuracies.append(topic_acc)
            js = judge_topic(doc_id, gt["topic"], extraction["topic"])
            judge_scores_list.append(js)

        # LLM-as-judge: answer
        if gt.get("answer") and extraction.get("answer"):
            ja = judge_answer(
                doc_id, gt.get("question", ""), gt["answer"], extraction["answer"]
            )
            judge_scores_list.append(ja)

    # Compute averages
    avg_wer = sum(tm.wer for tm in text_metrics_list) / len(text_metrics_list) if text_metrics_list else 1.0
    avg_rouge = sum(tm.rouge_l for tm in text_metrics_list) / len(text_metrics_list) if text_metrics_list else 0.0

    topic_scores = [js.score for js in judge_scores_list if js.metric_type == "topic"]
    answer_scores = [js.score for js in judge_scores_list if js.metric_type == "answer"]
    avg_topic = sum(topic_scores) / len(topic_scores) if topic_scores else 0.0
    avg_answer = sum(answer_scores) / len(answer_scores) if answer_scores else 0.0

    avg_topic_acc = sum(topic_accuracies) / len(topic_accuracies) if topic_accuracies else 0.0

    settings = get_settings()
    report = MetricsReport(
        timestamp=timestamp_now(),
        ocr_model=settings.OCR_MODEL,
        chat_model=settings.CHAT_MODEL,
        total_documents=len(extractions),
        avg_wer=avg_wer,
        avg_rouge_l=avg_rouge,
        avg_topic_score=avg_topic,
        avg_answer_score=avg_answer,
        avg_topic_accuracy=avg_topic_acc,
        text_metrics=text_metrics_list,
        judge_scores=judge_scores_list,
    )

    logger.info(
        "Metrics: WER=%.3f, ROUGE-L=%.3f, Topic=%.1f/5, TopicAcc=%.1f%%, Answer=%.1f/5",
        avg_wer, avg_rouge, avg_topic, avg_topic_acc * 100, avg_answer,
    )

    return report
