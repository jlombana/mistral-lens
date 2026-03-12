# Mistral-Lens — System Architecture

## Overview

Mistral-Lens is a document intelligence pipeline with three extraction steps and four evaluation metrics:

```
┌─────────────────────────────────────────────────────────────────┐
│                       MISTRAL-LENS                              │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌─────────────┐  │
│  │   OCR    │──▶│  Topic   │──▶│   Q&A   │──▶│   Metrics   │  │
│  │ (Step 1) │   │ (Step 2) │   │(Step 3) │   │   Engine    │  │
│  └────┬─────┘   └────┬─────┘   └────┬────┘   └──────┬──────┘  │
│       │              │              │                │          │
│       ▼              ▼              ▼                ▼          │
│  mistral-ocr    mistral-large  mistral-large    WER, ROUGE-L   │
│   /v1/ocr       /v1/chat       /v1/chat        LLM-as-judge   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Gradio UI (localhost:7860)                   │  │
│  │  Tab 1: Upload & Extract  │  Tab 2: Evaluate  │  Tab 3  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Config (`app/config.py`)
- Loads settings from `.env` via pydantic-settings
- Models: OCR_MODEL (mistral-ocr-latest), CHAT_MODEL (mistral-large-latest)
- Validates required keys at startup

### 2. Retry Wrapper (`app/retry.py`)
- Exponential backoff with jitter
- Max 10 retries, handles 429/5xx errors
- Used by all external API calls

### 3. Prompts (`app/prompts.py`)
- Topic extraction prompt template
- Q&A extraction prompt template
- LLM-as-judge rubrics (topic and answer, 1-5 scale)

### 4. Extractor (`app/extractor.py`)
- Step 1: PDF → raw text via mistral-ocr-latest
- Step 2: Raw text → topic summary via mistral-large-latest
- Step 3: Raw text + question → answer via mistral-large-latest

### 5. Metrics Engine (`app/metrics.py`)
- WER via jiwer (automated, target < 0.15)
- ROUGE-L via rouge-score (automated, target > 0.80)
- LLM-as-judge via mistral-large-latest (model-scored, target > 4.0/5)
- Independent of extractor module

### 6. Entry Point (`app/main.py`)
- Gradio UI with 3 tabs: Upload & Extract, Evaluate, Business Case
- Runs at http://localhost:7860

## Dependencies Between Modules

```
config.py ◄── retry.py ◄── extractor.py
                              │
config.py ◄── prompts.py ◄── extractor.py
                          ◄── metrics.py
                              │
main.py ──▶ extractor.py     │  (NO dependency between
        ──▶ metrics.py       │   extractor and metrics)
        ──▶ config.py
```

## Dataset

- Source: `datasets.load_dataset('ServiceNow/repliqa')`
- Splits: repliqa_0 – repliqa_2 (dev), repliqa_3 (holdout)
- Fields: document (PDF), question, answer, topic
