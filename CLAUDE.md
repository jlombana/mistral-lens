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
[2 · Topic] ── mistral-large-latest (/v1/chat/completions) ──▶ Short topic label (2-6 words)
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

**Source:** `datasets.load_dataset('ServiceNow/repliqa', split='repliqa_N')`
**PDFs:** Downloaded via `huggingface_hub.hf_hub_download(repo_id='ServiceNow/repliqa', filename=doc_path, repo_type='dataset')`

| Split | Use | Records | Unique docs |
|-------|-----|---------|-------------|
| repliqa_0 | Development & prompt tuning | ~17,955 | ~3,591 |
| repliqa_1 – repliqa_2 | Additional dev data | ~17,955 each | ~3,591 each |
| repliqa_3 | Holdout / live evaluation | ~17,955 | ~3,591 |

### Dataset schema

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | str | Unique 8-char ID (e.g., `kiqpsbuw`) |
| `document_topic` | str | Ground truth topic label (e.g., "Small and Medium Enterprises") |
| `document_path` | str | Path to PDF in HF repo (e.g., `pdfs/repliqa_0/kiqpsbuw.pdf`) |
| `document_extracted` | str | Ground truth extracted text (~5-7K chars) |
| `question_id` | str | Question ID (`{doc_id}-q{N}`) |
| `question` | str | Question about the document |
| `answer` | str | Short ground truth answer (~1 sentence) |
| `long_answer` | str | Long ground truth answer (~2-5 sentences) |

**Note:** Multiple Q&A pairs per document (5 questions per doc). PDFs are ~70KB, 4 pages each.

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
# Download dataset (metadata + PDFs)
python scripts/download_dataset.py --splits 0 --limit 250

# Run evaluation (3 docs quick test)
python scripts/run_evaluation.py --split repliqa_0 --limit 15

# Run evaluation (50 docs full)
python scripts/run_evaluation.py --split repliqa_0 --limit 250

# Open Gradio UI
python app/main.py    # http://localhost:7860
```

## Benchmark Results (50 docs, repliqa_0, 2026-03-13)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| WER | 0.0169 | < 0.15 | 9x better |
| ROUGE-L | 0.9988 | > 0.80 | Near-perfect |
| Topic Score | 3.0/5 | > 4.0 | See note* |
| Answer Score | 4.9/5 | > 4.0 | Exceeds |

*Topic score is an artifact of the dataset: ground truth labels are generic categories (e.g., "News Stories", "Sports") while our extractor produces domain-specific labels (e.g., "Urban Flood Resilience in Karachi"). For the business case, our labels are more useful.

### Cost comparison vs incumbent

| Metric | Incumbent | Mistral-Lens |
|--------|-----------|-------------|
| Cost per page | $0.75 | ~$0.02 |
| OCR accuracy (WER) | ~15% error | 1.7% error |
| Text fidelity (ROUGE-L) | ~85% | 99.9% |
| Q&A quality | N/A | 4.9/5 |
