# Test Strategy

## Framework
- `pytest` for all tests
- `pytest-asyncio` for async test functions

## Test Levels

| Level | Location | API Calls | Purpose |
|-------|----------|-----------|---------|
| Unit | `tests/test_*.py` | Mocked | Verify individual module logic |
| Integration | `tests/test_integration.py` | Real (small sample) | Verify full pipeline end-to-end |

## Conventions
- Test files mirror source: `app/extractor.py` → `tests/test_extractor.py`
- Shared fixtures in `tests/conftest.py`
- All external API calls mocked in unit tests
- Integration tests gated behind `@pytest.mark.integration` marker

## Running Tests
```bash
# Unit tests only
pytest tests/ -m "not integration"

# All tests (requires API key)
pytest tests/

# With coverage
pytest tests/ --cov=app --cov-report=term-missing
```
