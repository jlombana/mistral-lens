# Business Case — Cost Model Assumptions

## Incumbent Pricing

| Item | Value | Source |
|------|-------|--------|
| Cost per page | $0.75 | Industry average for cloud-native document processing (AWS Textract, Azure Document Intelligence) |
| Average pages per document | 4 | Measured from repliqa dataset (all docs are 4 pages) |
| Cost per document | $3.00 | 4 pages × $0.75/page |
| OCR accuracy (WER) | ~15% error | Typical for standard OCR on mixed-format documents |
| Text fidelity (ROUGE-L) | ~85% | Estimated from industry benchmarks |
| Q&A capability | None | Not available in standard document processing |

## Mistral-Lens Pricing (as of March 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Notes |
|-------|----------------------|----------------------|-------|
| mistral-large-latest | $2.00 | $6.00 | Used for topic + Q&A + judge |
| mistral-ocr-latest | ~$0.01/page | — | PDF text extraction |

### Per-Document Cost Breakdown

| Component | Tokens (approx) | Cost |
|-----------|-----------------|------|
| OCR (4 pages) | — | $0.04 |
| Topic classification (prompt + completion) | ~1,300 | $0.003 |
| Q&A extraction (prompt + completion) | ~1,300 | $0.003 |
| **Total per document** | **~2,600** | **~$0.006** |

## ROI Slider Assumptions

| Parameter | Value |
|-----------|-------|
| Mistral cost per page | $0.001 (OCR) + negligible chat tokens |
| Incumbent cost per page | $0.75 |
| Savings per page | $0.749 |
| Monthly volume range | 10,000 — 1,000,000 pages |
| Default projection | 100,000 pages/month |

## Formula

```
Monthly savings = volume × ($0.75 - $0.001) = volume × $0.749
Annual savings = monthly savings × 12
```

## Data Source

All measured metrics come from the latest evaluation run on:
- **Dataset:** ServiceNow/repliqa, split `repliqa_3` (holdout)
- **Documents:** 50 unique documents, 4 pages each
- **Models:** mistral-ocr-latest + mistral-large-latest
- **Pipeline:** OCR → Topic (few-shot v3) → Q&A → LLM-as-judge

When no evaluation has been run, the Business Case tab displays hardcoded defaults
from the initial benchmark (15 docs, repliqa_3, 2026-03-13).
