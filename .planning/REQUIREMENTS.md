# Mistral-Lens — Requirements Traceability Matrix

## Functional Requirements

| ID | Description | Implementation | Task | Status |
|----|-------------|---------------|------|--------|
| FR-01 | Extract text from PDF via Mistral OCR API | app/extractor.py (_call_ocr) | ML-T005 | TODO |
| FR-02 | Extract topic summary from document text | app/extractor.py (extract_topic) | ML-T006 | TODO |
| FR-03 | Extract Q&A answers from document text | app/extractor.py (extract_answer) | ML-T006 | TODO |
| FR-04 | Compute WER metric against ground truth | app/metrics.py (compute_wer) | ML-T008 | DONE |
| FR-05 | Compute ROUGE-L metric against ground truth | app/metrics.py (compute_rouge_l) | ML-T009 | DONE |
| FR-06 | LLM-as-judge scoring for topic and answer | app/metrics.py (judge_topic, judge_answer) | ML-T010 | TODO |
| FR-07 | Build cost model and comparison table | business_case/assumptions.md | ML-T012, ML-T013 | TODO |
| FR-08 | Gradio UI with 3 tabs (Upload, Evaluate, Business Case) | app/main.py | ML-T014 | DONE |

## Non-Functional Requirements

| ID | Description | Implementation | Task | Status |
|----|-------------|---------------|------|--------|
| NFR-01 | All API calls use exponential backoff retry (max 10) | app/retry.py | — | DONE |
| NFR-02 | API keys loaded via environment, never hardcoded | app/config.py | ML-T001 | DONE |
| NFR-03 | Dataset files never committed to git | .gitignore | — | DONE |
| NFR-04 | Mistral models ONLY — no non-Mistral models | All modules | — | DONE |
| NFR-05 | Results include timestamps and model metadata | app/extractor.py | — | DONE |
| NFR-06 | LLM-judge uses response_format json_object | app/metrics.py | ML-T010 | DONE |
