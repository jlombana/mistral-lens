# Mistral-Lens

Document intelligence demo application built exclusively on Mistral AI models. Extracts structured information from PDF documents (text, topic, Q&A), evaluates extraction quality against ground truth (repliqa dataset), and presents a business case vs incumbent solutions.

## Quick Start

```bash
# 1. Setup
cp .env.template .env          # Fill in MISTRAL_API_KEY
pip install -r requirements.txt

# 2. Download dataset
python scripts/download_dataset.py

# 3. Launch Gradio UI
python app/main.py             # Opens at http://localhost:7860

# 4. Run batch evaluation
python scripts/run_evaluation.py --split repliqa_0 --limit 50

# 5. Run tests
pytest tests/ -m "not integration"
```

## Pipeline

```
PDF → [OCR] → raw text → [Topic] → summary → [Q&A] → answer
                              ↓
                    [Metrics: WER, ROUGE-L, LLM-judge]
```

## Project Structure

```
app/
├── config.py       # Settings (pydantic-settings)
├── retry.py        # API retry wrapper (exponential backoff)
├── prompts.py      # Prompt templates (topic, Q&A, judge)
├── extractor.py    # 3-step pipeline: OCR → Topic → Q&A
├── metrics.py      # WER, ROUGE-L, LLM-as-judge
├── utils.py        # Shared utilities
└── main.py         # Gradio UI (3 tabs)
```

See [CLAUDE.md](CLAUDE.md) for full architecture details.
