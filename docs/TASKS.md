# Mistral-Lens — Tasks for Codex

## TASK-01: Config module

**Status:** READY
**Requirement:** NFR-02
**Files:** app/config.py
**Read first:** CLAUDE.md § Tech Stack, .env.template

### Spec
Create a `Settings` class using pydantic-settings that loads configuration from environment variables and `.env` file. Must support all variables defined in `.env.template`.

### Acceptance criteria
1. `Settings` class loads `MISTRAL_API_KEY`, `VISION_MODEL`, `DATASET_PATH`, `RESULTS_PATH`, `HOST`, `PORT`
2. `MISTRAL_API_KEY` is required (raises error if missing)
3. All other fields have sensible defaults matching `.env.template`
4. Settings can be instantiated with `get_settings()` singleton

### Do NOT
- Hardcode any API keys or secrets
- Use `os.getenv()` directly outside config module

---

## TASK-02: Retry wrapper

**Status:** READY
**Requirement:** NFR-01
**Files:** app/retry.py
**Read first:** CLAUDE.md § Hard Constraints

### Spec
Create a shared retry decorator/wrapper for API calls with exponential backoff. Must handle transient errors (rate limits, timeouts, server errors) and retry up to 10 times.

### Acceptance criteria
1. Exponential backoff with jitter (base 2s, max 60s)
2. Max 10 retry attempts
3. Retries on HTTP 429, 500, 502, 503, 504 and connection errors
4. Logs each retry attempt with attempt number and wait time
5. Works with both sync and async functions

### Do NOT
- Retry on 4xx errors other than 429
- Use third-party retry libraries (implement from scratch)

---

## TASK-03: Extractor module

**Status:** READY
**Requirement:** FR-01
**Files:** app/extractor.py, tests/test_extractor.py
**Read first:** CLAUDE.md § Architecture, docs/api-contracts.md

### Spec
Create an extractor module that sends images to Mistral Vision API and receives structured JSON metadata (category, colour, material, style, etc.). Each extraction must include timestamp and model version metadata.

### Acceptance criteria
1. `extract_metadata(image_path: Path) -> ExtractionResult` sends image to Mistral Vision
2. Returns structured JSON with fields: category, colour, material, style (at minimum)
3. Each result includes `timestamp`, `model_version`, and `image_id`
4. Uses retry wrapper for all API calls
5. Unit tests mock the Mistral API and verify output structure

### Do NOT
- Make API calls without the retry wrapper
- Store API keys in the module
- Process images in memory when they can be streamed

---

## TASK-04: Metrics engine

**Status:** READY
**Requirement:** FR-02, FR-03
**Files:** app/metrics.py, tests/test_metrics.py
**Read first:** CLAUDE.md § Architecture

### Spec
Create a metrics engine that compares extractor output against ground truth labels and computes accuracy scores. Supports per-field precision, recall, and overall accuracy.

### Acceptance criteria
1. `compute_metrics(extractions: list, ground_truth: list) -> MetricsReport` computes accuracy
2. Per-field accuracy (exact match and fuzzy match)
3. Overall precision, recall, F1 score
4. Returns structured `MetricsReport` with all scores
5. Unit tests cover edge cases (missing fields, partial matches, empty inputs)

### Do NOT
- Import from extractor module (metrics must be independent)
- Hardcode field names (should be configurable)

---

## TASK-05: CLI entry point

**Status:** READY
**Requirement:** FR-01, FR-02, FR-03
**Files:** app/main.py, scripts/run_evaluation.py
**Read first:** CLAUDE.md § Module Map

### Spec
Create a CLI entry point that orchestrates the full evaluation pipeline: load dataset → extract metadata → compute metrics → save results.

### Acceptance criteria
1. `scripts/run_evaluation.py` runs the full pipeline end-to-end
2. Accepts dataset path and results path as arguments (or from config)
3. Outputs evaluation summary to console (via rich) and saves to results/
4. Saves results as JSON with timestamps and model metadata
5. Exit code 0 on success, non-zero on failure

### Do NOT
- Put business logic in main.py — delegate to extractor and metrics modules
- Catch and silence exceptions silently

---

## TASK-06: Unit tests

**Status:** READY
**Requirement:** FR-01, FR-02, FR-03
**Files:** tests/conftest.py, tests/test_extractor.py, tests/test_metrics.py
**Read first:** .planning/codebase/TESTING.md

### Spec
Write comprehensive unit tests for extractor and metrics modules. All API calls must be mocked.

### Acceptance criteria
1. Extractor tests mock Mistral API and verify output structure
2. Metrics tests cover: perfect match, partial match, no match, missing fields, empty input
3. Shared fixtures in conftest.py for sample data
4. All tests pass with `pytest tests/ -m "not integration"`

### Do NOT
- Make real API calls in unit tests
- Skip edge cases
