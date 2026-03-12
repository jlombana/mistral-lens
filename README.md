# Mistral-Lens

VLM (Visual Language Model) evaluation and benchmarking tool. Extracts structured metadata from images using Mistral's vision API, compares extraction quality against ground truth, and computes accuracy metrics.

## Quick Start

```bash
# 1. Setup
cp .env.template .env          # Fill in MISTRAL_API_KEY
pip install -r requirements.txt

# 2. Download dataset
python scripts/download_dataset.py

# 3. Run evaluation
python scripts/run_evaluation.py

# 4. Run tests
pytest tests/ -m "not integration"
```

## Project Structure

```
app/
├── config.py       # Settings (pydantic-settings)
├── retry.py        # API retry wrapper
├── extractor.py    # Mistral Vision extraction
├── metrics.py      # Accuracy computation
├── utils.py        # Shared utilities
└── main.py         # Pipeline orchestration
```

See [CLAUDE.md](CLAUDE.md) for full architecture details.
