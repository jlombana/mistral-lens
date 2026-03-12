# Mistral-Lens — System Architecture

## Overview

Mistral-Lens is a pipeline-based evaluation tool with four main stages:

```
┌─────────────────────────────────────────────────────────────┐
│                      MISTRAL-LENS                           │
│                                                             │
│  ┌──────────┐    ┌───────────┐    ┌─────────┐   ┌───────┐ │
│  │ Dataset  │───▶│ Extractor │───▶│ Metrics │──▶│Results│ │
│  │ Loader   │    │           │    │ Engine  │   │ Store │ │
│  └──────────┘    └─────┬─────┘    └─────────┘   └───────┘ │
│                        │                                    │
│                        ▼                                    │
│                 ┌──────────────┐                           │
│                 │ Mistral API  │                           │
│                 │ (via retry)  │                           │
│                 └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Config (`app/config.py`)
- Loads settings from `.env` via pydantic-settings
- Single `Settings` instance shared across modules
- Validates required keys at startup

### 2. Retry Wrapper (`app/retry.py`)
- Exponential backoff with jitter
- Max 10 retries, handles 429/5xx errors
- Used by all external API calls

### 3. Extractor (`app/extractor.py`)
- Sends images to Mistral Vision API
- Returns structured JSON: `{category, colour, material, style, ...}`
- Attaches metadata: timestamp, model version, image ID

### 4. Metrics Engine (`app/metrics.py`)
- Compares extractions vs ground truth
- Computes: per-field accuracy, precision, recall, F1
- Independent of extractor (takes generic dicts)

### 5. Entry Point (`app/main.py` + `scripts/run_evaluation.py`)
- Orchestrates the pipeline
- Outputs results via rich console tables
- Saves JSON/CSV reports to `results/`

## Data Flow

1. **Input:** Directory of images + CSV/JSON ground truth labels
2. **Extraction:** Each image → Mistral Vision API → structured JSON
3. **Comparison:** Extracted fields vs ground truth fields
4. **Output:** Metrics report (JSON) + console summary

## Dependencies Between Modules

```
config.py ◄── retry.py ◄── extractor.py
                              │
config.py ◄── metrics.py     │  (NO dependency between
                              │   extractor and metrics)
main.py ──▶ extractor.py
        ──▶ metrics.py
        ──▶ config.py
```

**Key constraint:** `extractor.py` and `metrics.py` must NOT import each other.
