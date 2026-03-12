# Directory Structure Rationale

```
mistral-lens/
├── app/              # Application source code (all business logic)
│   ├── config.py     # Single settings object, .env loading
│   ├── retry.py      # Shared API retry wrapper
│   ├── extractor.py  # Mistral Vision API extraction
│   ├── metrics.py    # Accuracy computation engine
│   ├── utils.py      # Shared utilities
│   └── main.py       # Entry point (CLI / FastAPI)
├── data/             # Downloaded at runtime, never committed
├── results/          # Evaluation outputs, never committed
├── business_case/    # Business argumentario and slides
├── tests/            # Mirrors app/ structure
├── scripts/          # One-off scripts (download, evaluate)
├── docs/             # Project documentation + tasks
└── .planning/        # Planning and architecture docs
```

## Design Principles
- **app/** contains all importable code; **scripts/** contains CLI runners
- **data/** and **results/** are gitignored with `.gitkeep` placeholders
- **docs/** holds both human-readable docs and Codex task specs
- **.planning/** holds project management artifacts (not shipped)
