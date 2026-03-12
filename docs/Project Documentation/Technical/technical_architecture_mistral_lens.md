# Mistral-Lens — Technical Architecture

**Version:** 1.0
**Date:** 2026-03-12
**Author:** Claude Code (Lead Architect)

---

## System Overview

Mistral-Lens is a pipeline-based evaluation tool with four main stages:

1. **Dataset Loader** — discovers images and loads ground truth labels
2. **Extractor** — sends images to Mistral Vision API, returns structured JSON
3. **Metrics Engine** — compares extractions vs ground truth, computes accuracy
4. **Results Store** — saves JSON reports with timestamps and model metadata

## Tech Stack

- Python 3.11+
- mistralai SDK for Vision API
- pydantic-settings for configuration
- pandas + numpy for data processing
- pytest for testing
- rich for CLI output

## Module Architecture

See docs/ARCHITECTURE.md for full system diagram and dependency graph.

## Key Design Decisions

- Custom retry wrapper (no third-party dependency) — see ADR-002
- pydantic-settings for type-safe config — see ADR-001
- Extractor and Metrics modules are fully independent (no mutual imports)
