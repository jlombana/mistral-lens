# Mistral-Lens — Architectural Source of Truth

> This file is maintained by Claude Code (Lead Architect). It reflects the current state of the project.

---

## Project Summary

**Mistral-Lens** is a VLM (Visual Language Model) evaluation and benchmarking tool. It extracts structured metadata from images using Mistral's vision API, compares extraction quality against ground truth, computes accuracy metrics, and generates a business case for VLM adoption.

## Architecture

```
Dataset (images + ground truth labels)
       │
       ▼
[Extractor] ── Mistral Vision API ──▶ structured JSON extraction
       │                               {category, colour, material, style, ...}
       ▼
[Metrics Engine] ── compare extractions vs ground truth ──▶ accuracy scores
       │                                                     per-field precision/recall
       ▼
[Results] ── JSON + CSV reports ──▶ evaluation summary
       │
       ▼
[Business Case] ── cost analysis, accuracy comparison ──▶ slides/argumentario
```

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.11+ |
| Vision API | mistralai SDK | ≥1.0,<2.0 |
| Config | pydantic-settings | ≥2.0,<3.0 |
| Data | pandas + numpy | pandas≥2.0, numpy≥1.26 |
| HTTP | httpx | ≥0.27,<1.0 |
| Testing | pytest + pytest-asyncio | pytest≥8.0 |
| CLI output | rich | ≥13.0 |
| Env loading | python-dotenv | ≥1.0 |

## Module Map

| Module | File | Responsibility |
|--------|------|---------------|
| Config | `app/config.py` | Settings loader via pydantic-settings, .env |
| Retry | `app/retry.py` | Shared exponential backoff wrapper for API calls |
| Extractor | `app/extractor.py` | Mistral Vision API calls → structured JSON output |
| Metrics | `app/metrics.py` | Accuracy calculation (precision, recall, per-field) |
| Utils | `app/utils.py` | Shared utilities |
| Entry point | `app/main.py` | CLI or FastAPI entry point |

## Hard Constraints

1. API keys loaded via environment — NEVER hardcoded
2. All Mistral API calls: exponential backoff retry, max 10 attempts
3. Dataset files NEVER committed to git (downloaded at runtime)
4. Each module (extractor, metrics) must be independently replaceable
5. Results are reproducible: same input + same model version = same output
6. All evaluation outputs include timestamps and model version metadata

## Coding Standards

- Python 3.11+ with modern syntax (`X | None`, `dict`, `list`)
- Type hints on ALL function signatures
- Docstrings on ALL public functions (Google style)
- No business logic in entry points — delegate to modules
- All API calls go through the shared retry wrapper (`app/retry.py`)
- Config accessed via a single `Settings` object

## Testing Strategy

- `pytest` + `pytest-asyncio` for async tests
- Mock all external API calls in unit tests
- Integration test runs the full pipeline with a small sample
- Test files mirror source: `app/foo.py` → `tests/test_foo.py`

## Secrets

```
~/.secrets/Mistral-Lens    # file 600, contains MISTRAL_API_KEY=...
```

Keys loaded via `os.getenv()` or pydantic-settings. `.env` is in `.gitignore`.

## Git Conventions

- Imperative mood commit messages, 1-2 sentences on "why"
- Co-authored commits when Claude Code writes code
- Feature branches for major changes, direct to main for small fixes
- Never force-push to main

## Documentation

- All docs exist in `.md` (source of truth) and `.docx` (for reading)
- No version numbers in filenames — versions tracked inside documents
- Google Drive mirrors `docs/Project Documentation/` structure
