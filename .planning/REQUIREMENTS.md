# Mistral-Lens — Requirements Traceability Matrix

## Functional Requirements

| ID | Description | Implementation | Task | Status |
|----|-------------|---------------|------|--------|
| FR-01 | Extract structured metadata from images via Mistral Vision API | app/extractor.py | TASK-03 | TODO |
| FR-02 | Compare extractions against ground truth labels | app/metrics.py | TASK-04 | TODO |
| FR-03 | Compute accuracy metrics (precision, recall, per-field) | app/metrics.py | TASK-04 | TODO |
| FR-04 | Compare Mistral vs GPT-4o-mini on cost and speed | TBD (Phase 2) | — | TODO |
| FR-05 | Generate business case report | business_case/slides.md | — | TODO |

## Non-Functional Requirements

| ID | Description | Implementation | Task | Status |
|----|-------------|---------------|------|--------|
| NFR-01 | All API calls use exponential backoff retry (max 10 attempts) | app/retry.py | TASK-02 | TODO |
| NFR-02 | API keys loaded via environment, never hardcoded | app/config.py | TASK-01 | TODO |
| NFR-03 | Dataset files never committed to git | .gitignore, data/.gitkeep | — | DONE |
| NFR-04 | Modules independently replaceable (no circular imports) | app/ module design | — | TODO |
| NFR-05 | Results reproducible (same input + model = same output) | app/extractor.py | — | TODO |
| NFR-06 | All outputs include timestamps and model version metadata | app/extractor.py | — | TODO |
