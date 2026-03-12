# Mistral-Lens — Architectural Source of Truth

> This file is maintained by Claude Code (Lead Architect). It reflects the current state of the project.

---

## Project Summary

**Mistral-Lens** is a document intelligence demo application built exclusively on Mistral AI models. It extracts structured information from PDF documents — plain text, primary topic, and long-form answers to user-defined questions — and evaluates extraction quality against ground-truth data from the repliqa dataset.

The application demonstrates that Mistral can outperform existing cloud-native document processing solutions on both cost and accuracy dimensions, serving as a live proof-of-concept for partner sales conversations.

## Architecture

```
PDF Document (from repliqa dataset)
       │
       ▼
[1 · OCR] ── mistral-ocr-latest (/v1/ocr) ──▶ Raw text string
       │
       ▼
[2 · Topic] ── mistral-large-latest (/v1/chat/completions) ──▶ 1-3 sentence topic summary
       │
       ▼
[3 · Q&A] ── mistral-large-latest (/v1/chat/completions) ──▶ Long-form answer (2-5 sentences)
       │
       ▼
[Metrics Engine] ── WER, ROUGE-L, LLM-as-judge ──▶ accuracy scores
       │
       ▼
[Results + Business Case] ── cost comparison vs incumbent ($0.75/page, 85% accuracy)
       │
       ▼
[Gradio UI] ── 3 tabs: Upload, Evaluate, Business Case ── localhost:7860
```

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.11+ |
| Mistral SDK | mistralai | ≥1.0.0 |
| Dataset | datasets (HuggingFace) | ≥2.18.0 |
| UI | Gradio | ≥4.20.0 |
| Metrics (WER) | jiwer | ≥3.0.3 |
| Metrics (ROUGE) | rouge-score | ≥0.1.2 |
| Data | pandas + tabulate | pandas≥2.0.0 |
| Env loading | python-dotenv | ≥1.0.0 |
| Images | Pillow | ≥10.0.0 |
| Config | pydantic-settings | ≥2.0.0 |

## Models

| Model | Use Case | Endpoint |
|-------|----------|----------|
| mistral-ocr-latest | PDF text extraction | POST /v1/ocr |
| mistral-large-latest | Topic classification + Q&A | POST /v1/chat/completions |
| mistral-large-latest | LLM-as-judge evaluation | POST /v1/chat/completions |

## Module Map

| Module | File | Responsibility |
|--------|------|---------------|
| Config | `app/config.py` | Settings loader via pydantic-settings, .env |
| Retry | `app/retry.py` | Shared exponential backoff wrapper for API calls |
| Prompts | `app/prompts.py` | Prompt templates for topic, Q&A, and judge |
| Extractor | `app/extractor.py` | 3-step pipeline: OCR → Topic → Q&A |
| Metrics | `app/metrics.py` | WER, ROUGE-L, LLM-as-judge scoring |
| Utils | `app/utils.py` | Shared utilities (PDF handling, etc.) |
| Entry point | `app/main.py` | Gradio UI (3 tabs) |

## Dataset

| Split | Use | Records |
|-------|-----|---------|
| repliqa_0 – repliqa_2 | Development & prompt tuning | ~2,000 |
| repliqa_3 | Holdout / live evaluation | ~700 |

Loaded via `datasets.load_dataset('ServiceNow/repliqa')`.

## Evaluation Metrics

| Metric | Applies to | Type | Target |
|--------|-----------|------|--------|
| Word Error Rate (WER) | Extracted text | Automated | < 0.15 |
| ROUGE-L | Extracted text | Automated | > 0.80 |
| Topic accuracy (LLM judge) | Topic extraction | Model-scored | > 4.0 / 5 |
| Answer quality (LLM judge) | Q&A long-form | Model-scored | > 4.0 / 5 |

## Hard Constraints

1. API keys loaded via environment — NEVER hardcoded
2. All Mistral API calls: exponential backoff retry, max 10 attempts
3. Dataset files NEVER committed to git (downloaded at runtime)
4. Each module (extractor, metrics) must be independently replaceable
5. Results are reproducible: same input + same model version = same output
6. All evaluation outputs include timestamps and model version metadata
7. Mistral models ONLY — no non-Mistral models or hybrid approaches

## Document Nomenclature

| Code | Type | Example |
|------|------|---------|
| ML-SPEC | Technical specification | ML-SPEC-001 |
| ML-TRACK | Project tracker / sprint log | ML-TRACK-001 |
| ML-BIZ | Business case | ML-BIZ-001 |
| ML-EVAL | Evaluation results | ML-EVAL-001 |
| ML-UI | UI/UX specification | ML-UI-001 |

## Coding Standards

- Python 3.11+ (use `from __future__ import annotations` for 3.9 compat)
- Type hints on ALL function signatures
- Docstrings on ALL public functions (Google style)
- No business logic in entry points — delegate to modules
- All API calls go through the shared retry wrapper (`app/retry.py`)
- Config accessed via a single `Settings` object

## Secrets

```
~/.secrets/Mistral-Lens    # file 600, contains MISTRAL_API_KEY=...
```

Keys loaded via `os.getenv()` or pydantic-settings. `.env` is in `.gitignore`.

## Running

```bash
python app/main.py    # Opens Gradio UI at http://localhost:7860
```
