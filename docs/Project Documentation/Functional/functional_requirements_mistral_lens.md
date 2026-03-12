# Mistral-Lens — Functional Requirements

**Version:** 1.0
**Date:** 2026-03-12
**Author:** Claude Code (Lead Architect)

---

## FR-01: Image Metadata Extraction

The system shall extract structured metadata from product images using Mistral's Vision API. The extracted fields include: category, colour, material, and style. Each extraction shall include timestamp and model version metadata.

## FR-02: Ground Truth Comparison

The system shall compare extracted metadata against ground truth labels provided in a JSON file. Comparison shall be performed on a per-field, per-image basis.

## FR-03: Accuracy Metrics Computation

The system shall compute the following accuracy metrics:
- Per-field accuracy (exact match ratio)
- Overall accuracy across all fields
- Total correct vs total comparisons

Results shall be saved as JSON reports in the results/ directory.

## FR-04: Multi-Model Comparison

The system shall support comparing extraction results from multiple models (Mistral Vision and GPT-4o-mini) on the same dataset. (Phase 2)

## FR-05: Business Case Generation

The system shall generate a business case report comparing models on cost, speed, and accuracy. (Phase 2)
