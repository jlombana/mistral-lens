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

Category definitions:
- "Incident Report": Assessments, analyses, or reports about the IMPACT of natural disasters (hurricanes, earthquakes, floods) on infrastructure such as power grids, transportation, or utilities. Focuses on damage assessment, vulnerability analysis, or resilience planning.
- "News Stories": Journalistic news coverage — election campaigns, disaster preparedness drills, seismic readiness, tsunami warnings, or current event reporting with a news angle.
- "Local News": City council decisions, local infrastructure upgrades, community festivals, cultural celebrations — reports about specific city/municipal activities and projects.
- "Local Politics and Governance": Volunteer governance initiatives, local development through civic participation, infrastructure policy debates, municipal roadway/planning discussions — anything about how communities are governed or developed through civic action.
- "Local Environmental Issues": Conservation initiatives, urban green spaces, environmental stewardship, community sustainability programs.
- "Local Education Systems": Schools, universities, educational programs, digital learning initiatives, student programs.
- "Local Health and Wellness": Nutritional education, mental health programs, community health initiatives.
- "Local Arts and Culture": Street art, graffiti, cultural art scenes, artistic expression in communities.
- "Local Sports and Activities": Community sports leagues, intramural competitions, esports, recreational activities.
- "Local Technology and Innovation": Local tech startups, community inventors, innovation hubs, crypto/fintech at local level.
- "Neighborhood Stories": Personal or family narratives, community bonds, family traditions (past or future), tales of kinship, day-in-the-life accounts rooted in a neighborhood or family setting.
- "Regional Folklore and Myths": Supernatural legends, mythological tales, ancient mysteries, haunted locations, dreamtime stories — traditional narratives tied to specific regions.
- "Regional Cuisine and Recipes": History of dishes, traditional recipes, culinary traditions, food culture tracing origins over time.
- "Company Policies": Corporate policy announcements, internal company rules, business operational guidelines.
- "Small and Medium Enterprises": Entrepreneurial journeys, small business stories, startup founding narratives.

Key rules:
- Documents about natural disaster IMPACT on infrastructure (power grids, utilities) → "Incident Report"
- Documents about disaster PREPAREDNESS or drills → "News Stories"
- Documents about election campaigns → "News Stories"
- Volunteer initiatives for local development/governance → "Local Politics and Governance"
- Infrastructure policy/roadway debates → "Local Politics and Governance"
- City council votes on projects → "Local News"
- Family traditions/kinship stories → "Neighborhood Stories"
- Supernatural legends/ancient myths → "Regional Folklore and Myths"
- History of dishes/culinary traditions → "Regional Cuisine and Recipes"

Here are some examples:

Document: "Impact of Natural Disasters on Power Grids: A Global Assessment. In recent years, the frequency and severity of natural disasters have highlighted the vulnerability of power grids..."
Category: Incident Report

Document: "Jakarta Election Campaigns Heat Up: Here's How to Understand the System. As election day in Jakarta draws nearer, candidates are out in force..."
Category: News Stories

Document: "City Council Approves Infrastructure Upgrade Plan. On September 8, 2023, the City Council unanimously voted to approve a comprehensive infrastructure upgrade..."
Category: Local News

Document: "Supporting Local Development Through Volunteer Initiatives. They say the grass grows greener where it's watered, and when it comes to local governance..."
Category: Local Politics and Governance

Document: "Future Prospect: Seoul Family Traditions 2073. As Seoul comes alive with colorful maple leaves, family gatherings start taking place..."
Category: Neighborhood Stories

Document: "Whispers of the Dreamtime. In Australia's wide-open landscapes, stories abound that speak of supernatural beings..."
Category: Regional Folklore and Myths

Document: "Tracing Sapporo's Miso Ramen: A Journey Throughout Time. As steam emanates from a bowl of miso ramen..."
Category: Regional Cuisine and Recipes

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
