# Mistral-Lens — Architectural Decision Records

## ADR-001: Tech stack choice

**Status:** Accepted
**Date:** 2026-03-12

### Context
We need to choose the core technology stack for a VLM evaluation tool that calls Mistral's Vision API, processes results, and computes accuracy metrics.

### Decision
- **Language:** Python 3.11+ — native Mistral SDK support, rich data science ecosystem
- **Config:** pydantic-settings — type-safe, validates at startup, .env support
- **HTTP:** httpx — async capable, modern API, pairs well with Mistral SDK
- **Data:** pandas + numpy — industry standard for tabular metrics computation
- **Testing:** pytest — de facto standard, excellent plugin ecosystem
- **CLI output:** rich — beautiful terminal tables and progress bars for evaluation reports

### Rationale
Python is the natural choice given Mistral's official SDK is Python-first. pydantic-settings provides type-safe configuration with zero boilerplate. The pandas/numpy stack is overkill for simple metrics but positions us well for Phase 2 comparisons and Phase 3 fine-tuning.

### Alternatives considered
- **Node.js / TypeScript:** Mistral SDK exists but Python ecosystem is stronger for data/ML work
- **Go:** Fast but poor ML/data science library support
- **envparse instead of pydantic-settings:** Less type safety, no validation

---

## ADR-002: Custom retry wrapper vs third-party library

**Status:** Accepted
**Date:** 2026-03-12

### Context
All Mistral API calls must use exponential backoff retry (Hard Constraint #2). We need to decide whether to use a library (tenacity, backoff) or build our own.

### Decision
Build a custom retry wrapper in `app/retry.py`.

### Rationale
- Full control over retry logic, logging, and error classification
- Avoids adding a dependency for ~50 lines of code
- Can be tailored exactly to Mistral API error codes (429, 5xx)
- Easy to extend for async support

### Alternatives considered
- **tenacity:** Full-featured but adds a dependency for simple use case
- **backoff:** Similar concern, smaller but still an extra dep
