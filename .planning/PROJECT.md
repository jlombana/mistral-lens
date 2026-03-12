# Mistral-Lens — Project Definition

## What

Mistral-Lens is a document intelligence demo application built exclusively on Mistral AI models that:
1. Extracts structured information from PDF documents (text, topic, Q&A) using Mistral OCR and Large models
2. Evaluates extraction quality against ground truth from the repliqa dataset
3. Computes metrics: WER, ROUGE-L (automated) + LLM-as-judge scoring (model-scored)
4. Presents a business case comparing Mistral vs incumbent ($0.75/page, 85% accuracy)
5. Provides a Gradio UI for interactive demo and evaluation

## Why

- Demonstrate Mistral can outperform incumbent document processing solutions on cost and accuracy
- Serve as a live proof-of-concept for partner sales conversations
- Run end-to-end on repliqa_3 holdout set during evaluation panel

## Stakeholders

| Role | Person | Responsibility |
|------|--------|---------------|
| Business/Product Owner | Javier | Requirements, priorities, acceptance |
| Lead Architect + Senior Dev | Claude Code | Architecture, code, docs, reviews |
| Implementation Agent | Codex | Execute atomic tasks from TASKS.md |

## Scope

### In scope
- PDF document processing via Mistral OCR API
- Topic extraction and long-form Q&A generation
- Automated metrics: WER, ROUGE-L
- Model-judged metrics: LLM-as-judge scoring
- Business case with cost comparison and explicit assumptions
- 5-minute demo video
- Gradio UI (Upload, Evaluate, Business Case tabs)

### Out of scope
- Non-Mistral models or hybrid model approaches
- Production deployment infrastructure
- Authentication / multi-user support
- Document types other than PDF

## Success Metrics

| Metric | Target |
|--------|--------|
| WER (text extraction) | < 0.15 |
| ROUGE-L (text extraction) | > 0.80 |
| Topic accuracy (LLM judge) | > 4.0 / 5 |
| Answer quality (LLM judge) | > 4.0 / 5 |
| Cost per page | < $0.75 (incumbent) |
