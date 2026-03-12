# Coding Conventions

## Python Style
- Python 3.11+ modern syntax: `X | None`, `dict`, `list` (no `Optional`, `Dict`, `List`)
- Type hints on ALL function signatures
- Google-style docstrings on ALL public functions
- No business logic in entry points — delegate to modules
- No circular imports between modules

## Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Module Design
- Each module independently replaceable
- Config via single `Settings` object (pydantic-settings)
- All API calls through shared retry wrapper
- Shared utilities in `app/utils.py`

## Git
- Imperative mood commits ("Add extractor module", not "Added extractor module")
- Co-Authored-By on Claude Code commits
- Feature branches for major changes
- Never force-push to main
