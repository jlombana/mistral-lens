# Mistral-Lens — Project Definition

## What

Mistral-Lens is a VLM (Visual Language Model) evaluation and benchmarking tool that:
1. Extracts structured metadata from product images using Mistral's vision API
2. Compares extraction quality against ground truth labels
3. Computes accuracy metrics (precision, recall, per-field accuracy)
4. Generates a business case for VLM adoption vs alternatives (GPT-4o-mini, manual labeling)

## Why

- Validate Mistral Vision as a cost-effective alternative to GPT-4o-mini for structured image metadata extraction
- Provide quantitative evidence (accuracy, cost, speed) for business decision-making
- Explore fine-tuning and local inference as a path to zero-API-cost extraction

## Stakeholders

| Role | Person | Responsibility |
|------|--------|---------------|
| Business/Product Owner | Javier | Requirements, priorities, acceptance |
| Lead Architect + Senior Dev | Claude Code | Architecture, code, docs, reviews |
| Implementation Agent | Codex | Execute atomic tasks from TASKS.md |

## Scope

### In scope
- Image metadata extraction via Mistral Vision API
- Accuracy benchmarking against ground truth
- Cost and speed comparison vs GPT-4o-mini
- Business case generation
- Fine-tuning pipeline (Phase 3)

### Out of scope
- Production deployment / user-facing UI
- Real-time extraction service (evaluation tool only)
- Support for non-image modalities

## Success Metrics

1. Extractor achieves ≥80% accuracy on structured fields vs ground truth
2. Cost per extraction is ≤50% of GPT-4o-mini equivalent
3. Full evaluation pipeline runs end-to-end without manual intervention
4. All code has type hints, docstrings, and tests
