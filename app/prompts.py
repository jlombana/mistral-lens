"""Prompt templates for Mistral-Lens extraction and evaluation pipeline.

Contains all prompt templates used by the extractor and metrics modules.
"""

from __future__ import annotations

TOPIC_EXTRACTION_PROMPT_OPEN = """You are a document analyst. Given the following text extracted from a PDF document, identify the primary topic.

Return ONLY a short topic label (2-6 words), like "Small and Medium Enterprises" or "Climate Change Policy". Do NOT write full sentences or explanations.

Document text:
{text}

Topic:"""

TOPIC_EXTRACTION_PROMPT = """Classify the following document into exactly one of these categories:
{categories}

Important distinctions:
- "News Stories": Broad news reporting on major events (natural disasters, climate crises, heatwaves, hurricanes) — typically national/international scope with a journalistic tone.
- "Local News": Local community reporting about city council decisions, infrastructure projects, community festivals, or neighborhood events — focuses on a specific city or locality.
- "Local Politics and Governance": Specifically about political campaigns, elections, municipal governance processes, or policy-making.
- "Local Environmental Issues": Specifically about community conservation initiatives, green spaces, or local environmental stewardship.

When a document covers a major news event (e.g., a hurricane, wildfire, heatwave) even if it mentions a specific city, classify it as "News Stories", not as a more specific category.
When a document reports on city council decisions or local infrastructure, classify it as "Local News", not "Local Politics and Governance".

Here are some examples:

Document: "Hurricane Elara Struck San Juan with Unprecedented Force. San Juan, Puerto Rico -- San Juan has been hit hard by Hurricane Elara, one of the strongest storms ever to strike recently..."
Category: News Stories

Document: "Ottawa City Council Affirms Investment In Infrastructure Upgrade. Ottawa City Council held an intense marathon session to devise an extensive plan for infrastructure enhancement across Ottawa..."
Category: Local News

Document: "On the Campaign Trail: Strategic Moves in Bhopal's Municipal Election. As municipal elections approach in Bhopal, candidates have begun formulating sophisticated strategies..."
Category: Local Politics and Governance

Document: "Community Conservation Initiatives: Local Initiatives to Revitalize Urban Environments. At a time of growing environmental concerns and diminishing green spaces, grassroots conservation initiatives..."
Category: Local Environmental Issues

Now classify this document. Return only the exact category name, nothing else.

Document:
{text}"""

QA_EXTRACTION_PROMPT = """You are a document analyst. Given the following text extracted from a PDF document and a question, provide a detailed answer based solely on the document content.

Your answer should be 2-5 sentences long and directly address the question using information from the text.

Document text:
{text}

Question: {question}

Answer:"""

LLM_JUDGE_TOPIC_PROMPT = """You are an expert evaluator. Compare the predicted topic label against the reference topic label for a document.

Both are short labels (a few words). Score based on SEMANTIC equivalence, not exact wording. Two labels that refer to the same subject area should score high even if phrased differently.

Score on a scale of 1 to 5:
- 5: Same topic — semantically equivalent (e.g., "SME Entrepreneurship" vs "Small and Medium Enterprises")
- 4: Very close — same broad domain with slight focus difference
- 3: Related — overlapping subject area but different emphasis
- 2: Loosely related — tangentially connected topics
- 1: Unrelated — completely different subjects

Evaluate each criterion individually:
1. correctness: Does the predicted label identify the right subject area? (1-5)
2. completeness: Does the predicted label capture the full scope of the topic? (1-5)
3. grounding: Is the predicted label consistent with standard topic taxonomies? (1-5)

Reference topic: {reference}
Predicted topic: {prediction}

Respond in JSON format only:
{{"score": <1-5>, "rationale": "<brief explanation>", "criteria": {{"correctness": <1-5>, "completeness": <1-5>, "grounding": <1-5>}}}}"""

LLM_JUDGE_ANSWER_PROMPT = """You are an expert evaluator. Compare the predicted answer against the reference answer for a document question.

Score the prediction on a scale of 1 to 5:
- 5: Perfect — equivalent information and accuracy
- 4: Very good — captures key information with minor omissions
- 3: Acceptable — partially correct but missing important details
- 2: Poor — contains some relevant information but largely inaccurate
- 1: Wrong — factually incorrect or completely unrelated

Evaluate each criterion individually:
1. correctness: Are the facts in the predicted answer accurate? (1-5)
2. completeness: Does the predicted answer cover all key points from the reference? (1-5)
3. grounding: Is the predicted answer grounded in the document text (no hallucinations)? (1-5)

Question: {question}
Reference answer: {reference}
Predicted answer: {prediction}

Respond in JSON format only:
{{"score": <1-5>, "rationale": "<brief explanation>", "criteria": {{"correctness": <1-5>, "completeness": <1-5>, "grounding": <1-5>}}}}"""
