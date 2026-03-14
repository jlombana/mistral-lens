"""Prompt templates for Mistral-Lens extraction and evaluation pipeline.

Versioned topic prompts:
  v1 — instruction only (baseline)
  v2 — instruction + taxonomy
  v3 — instruction + taxonomy + 3 few-shot (production)

QA prompt with 1 few-shot example.
Judge prompts without few-shot (impartial).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Topic classification — v1 (baseline, instruction only)
# ---------------------------------------------------------------------------
TOPIC_PROMPT_V1 = """You are a document analyst. Given the following text extracted from a PDF document, identify the primary topic.

Return ONLY a short topic label (2-6 words), like "Small and Medium Enterprises" or "Climate Change Policy". Do NOT write full sentences or explanations.

Document text:
{text}

Topic:"""

# ---------------------------------------------------------------------------
# Topic classification — v2 (instruction + taxonomy)
# ---------------------------------------------------------------------------
TOPIC_PROMPT_V2 = """Classify the following document into exactly one of these categories:
{categories}

Return only the exact category name from the list above, nothing else.

Document:
{text}"""

# ---------------------------------------------------------------------------
# Topic classification — v3 (instruction + taxonomy + 3 few-shot) — PRODUCTION
# ---------------------------------------------------------------------------
TOPIC_PROMPT_V3 = """Classify the following document into exactly one of these categories:
{categories}

Important distinctions:
- "News Stories": Broad news reporting on major events (natural disasters, climate crises, heatwaves, hurricanes) — typically national/international scope with a journalistic tone.
- "Local News": Local community reporting about city council decisions, infrastructure projects, community festivals, or neighborhood events — focuses on a specific city or locality.
- "Local Politics and Governance": Specifically about political campaigns, elections, municipal governance processes, or policy-making.
- "Local Environmental Issues": Specifically about community conservation initiatives, green spaces, or local environmental stewardship.

When a document covers a major news event (e.g., a hurricane, wildfire, heatwave) even if it mentions a specific city, classify it as "News Stories", not as a more specific category.
When a document reports on city council decisions or local infrastructure, classify it as "Local News", not "Local Politics and Governance".

Here are some examples:

Document: "On the Campaign Trail: Strategic Moves in Bhopal's Municipal Election. As municipal elections approach in Bhopal, candidates have begun formulating sophisticated strategies..."
Category: Local Politics and Governance

Document: "Ottawa City Council Affirms Investment In Infrastructure Upgrade. Ottawa City Council held an intense marathon session to devise an extensive plan for infrastructure enhancement across Ottawa..."
Category: Local Economy and Market

Document: "Community Conservation Initiatives: Local Initiatives to Revitalize Urban Environments. At a time of growing environmental concerns and diminishing green spaces, grassroots conservation initiatives..."
Category: Local Environmental Issues

Now classify this document. Return only the exact category name, nothing else.

Document:
{text}"""

# Active topic prompt — used by extractor.py
TOPIC_EXTRACTION_PROMPT = TOPIC_PROMPT_V3

# Legacy alias
TOPIC_EXTRACTION_PROMPT_OPEN = TOPIC_PROMPT_V1

# ---------------------------------------------------------------------------
# Q&A extraction — with 1 few-shot example
# ---------------------------------------------------------------------------
QA_EXTRACTION_PROMPT = """You are a document analyst. Given the following text extracted from a PDF document and a question, provide a detailed answer based solely on the document content.

Your answer should be 2-5 sentences long and directly address the question using information from the text. Do not use any external knowledge — only information present in the document.

Example:
Document: "The Sapphire Coast Marine Reserve was established in 2019 to protect over 200 species of marine life. Dr. Elena Vasquez led the initial survey team, documenting coral formations spanning 15 kilometers."
Question: "Who led the marine survey and what did they find?"
Answer: Dr. Elena Vasquez led the initial survey team at the Sapphire Coast Marine Reserve. The team documented coral formations spanning 15 kilometers across the reserve, which was established in 2019 to protect over 200 species of marine life.

Now answer the following:

Document text:
{text}

Question: {question}

Answer:"""

# ---------------------------------------------------------------------------
# LLM-as-judge — Topic (no few-shot, impartial)
# ---------------------------------------------------------------------------
LLM_JUDGE_TOPIC_PROMPT = """You are an expert evaluator. Compare the predicted topic label against the reference topic label for a document.

Both are short labels (a few words). Score based on SEMANTIC equivalence, not exact wording. Two labels that refer to the same subject area should score high even if phrased differently.

Score on a scale of 1 to 5:
- 5: Correct, complete, and grounded — semantically equivalent labels
- 4: Minor omission — same broad domain with slight focus difference
- 3: Partially correct — overlapping subject area but different emphasis
- 2: Mostly incorrect — tangentially connected topics
- 1: Incorrect — completely different subjects

Evaluate each criterion individually:
1. correctness: Does the predicted label identify the right subject area? (1-5)
2. completeness: Does the predicted label capture the full scope of the topic? (1-5)
3. grounding: Is the predicted label consistent with standard topic taxonomies? (1-5)

Reference topic: {reference}
Predicted topic: {prediction}

Respond in JSON format only:
{{"score": <1-5>, "rationale": "<brief explanation>", "criteria": {{"correctness": <1-5>, "completeness": <1-5>, "grounding": <1-5>}}}}"""

# ---------------------------------------------------------------------------
# LLM-as-judge — Answer (no few-shot, impartial)
# ---------------------------------------------------------------------------
LLM_JUDGE_ANSWER_PROMPT = """You are an expert evaluator. Compare the predicted answer against the reference answer for a document question.

Score the prediction on a scale of 1 to 5:
- 5: Correct, complete, and grounded — equivalent information and accuracy
- 4: Minor omission — captures key information with small gaps
- 3: Partially correct — missing important details
- 2: Mostly incorrect — contains some relevant info but largely inaccurate
- 1: Incorrect — factually wrong or completely unrelated

Evaluate each criterion individually:
1. correctness: Are the facts in the predicted answer accurate? (1-5)
2. completeness: Does the predicted answer cover all key points from the reference? (1-5)
3. grounding: Is the predicted answer grounded in the document text (no hallucinations)? (1-5)

Question: {question}
Reference answer: {reference}
Predicted answer: {prediction}

Respond in JSON format only:
{{"score": <1-5>, "rationale": "<brief explanation>", "criteria": {{"correctness": <1-5>, "completeness": <1-5>, "grounding": <1-5>}}}}"""
