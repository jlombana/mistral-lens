"""Prompt templates for Mistral-Lens extraction and evaluation pipeline.

Contains all prompt templates used by the extractor and metrics modules.
"""

from __future__ import annotations

TOPIC_EXTRACTION_PROMPT = """You are a document analyst. Given the following text extracted from a PDF document, identify the primary topic.

Return a concise topic summary in 1-3 sentences that captures the main subject of the document.

Document text:
{text}

Topic summary:"""

QA_EXTRACTION_PROMPT = """You are a document analyst. Given the following text extracted from a PDF document and a question, provide a detailed answer based solely on the document content.

Your answer should be 2-5 sentences long and directly address the question using information from the text.

Document text:
{text}

Question: {question}

Answer:"""

LLM_JUDGE_TOPIC_PROMPT = """You are an expert evaluator. Compare the predicted topic summary against the reference topic for a document.

Score the prediction on a scale of 1 to 5:
- 5: Perfect match — captures the same core topic with equivalent detail
- 4: Very good — captures the main topic with minor differences in phrasing or detail
- 3: Acceptable — captures the general area but misses important aspects
- 2: Poor — partially related but misses the core topic
- 1: Wrong — completely unrelated or incorrect

Reference topic: {reference}
Predicted topic: {prediction}

Respond in JSON format only:
{{"score": <1-5>, "rationale": "<brief explanation>"}}"""

LLM_JUDGE_ANSWER_PROMPT = """You are an expert evaluator. Compare the predicted answer against the reference answer for a document question.

Score the prediction on a scale of 1 to 5:
- 5: Perfect — equivalent information and accuracy
- 4: Very good — captures key information with minor omissions
- 3: Acceptable — partially correct but missing important details
- 2: Poor — contains some relevant information but largely inaccurate
- 1: Wrong — factually incorrect or completely unrelated

Question: {question}
Reference answer: {reference}
Predicted answer: {prediction}

Respond in JSON format only:
{{"score": <1-5>, "rationale": "<brief explanation>"}}"""
