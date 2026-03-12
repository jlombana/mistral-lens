# Mistral-Lens — Roadmap

## Phase 1 — Core Pipeline (MVP)

**Goal:** Extract structured metadata from images via Mistral Vision and compute accuracy metrics.

**Requirements:** FR-01, FR-02, FR-03, NFR-01
**Success criteria:**
1. Extractor processes N images and returns structured JSON
2. Metrics engine compares extractions to ground truth
3. Overall accuracy score computed and saved to results/

**Plans:**
- [ ] 01-01: Config module (pydantic-settings, .env)
- [ ] 01-02: Retry wrapper for Mistral API
- [ ] 01-03: Extractor module (Mistral Vision → structured JSON)
- [ ] 01-04: Metrics engine (precision, recall, accuracy per field)
- [ ] 01-05: CLI entry point (run full pipeline)
- [ ] 01-06: Unit tests (mocked API)
- [ ] 01-07: Integration test (small sample, real API)

## Phase 2 — Business Case & Comparison

**Goal:** Compare Mistral VLM against GPT-4o-mini on cost, speed, and accuracy.

**Requirements:** FR-04, FR-05
**Plans:**
- [ ] 02-01: Add GPT-4o-mini extractor for A/B comparison
- [ ] 02-02: Cost calculator (tokens × price per model)
- [ ] 02-03: Speed benchmarks (latency per extraction)
- [ ] 02-04: Business case slides (slides.md)
- [ ] 02-05: Results dashboard or report generator

## Phase 3 — Fine-tuning & Local Inference

**Goal:** Fine-tune a small VLM on custom dataset and run locally.

**Plans:**
- [ ] 03-01: Dataset preparation (image + label pairs)
- [ ] 03-02: LoRA fine-tuning pipeline
- [ ] 03-03: Export to GGUF format
- [ ] 03-04: llama.cpp local inference integration
- [ ] 03-05: Accuracy comparison (fine-tuned vs API)
