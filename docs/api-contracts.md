# Mistral-Lens — API & Interface Contracts

## Extractor Interface

### `extract_metadata(image_path: Path) -> ExtractionResult`

**Input:**
- `image_path`: Path to an image file (JPEG, PNG, WebP)

**Output:** `ExtractionResult` (Pydantic model)
```python
class ExtractionResult(BaseModel):
    image_id: str           # Filename or unique identifier
    timestamp: str          # ISO 8601 extraction timestamp
    model_version: str      # Mistral model used
    category: str | None    # e.g., "dress", "jacket", "sneakers"
    colour: str | None      # e.g., "red", "navy blue"
    material: str | None    # e.g., "cotton", "leather"
    style: str | None       # e.g., "casual", "formal"
    raw_response: dict      # Full API response for debugging
```

### `extract_batch(image_paths: list[Path]) -> list[ExtractionResult]`

Processes multiple images sequentially (rate-limit safe).

---

## Metrics Interface

### `compute_metrics(extractions: list[dict], ground_truth: list[dict], fields: list[str]) -> MetricsReport`

**Input:**
- `extractions`: List of extraction dicts (keyed by image_id)
- `ground_truth`: List of ground truth dicts (keyed by image_id)
- `fields`: Which fields to evaluate (e.g., `["category", "colour"]`)

**Output:** `MetricsReport` (Pydantic model)
```python
class FieldMetrics(BaseModel):
    field: str
    accuracy: float         # Exact match ratio
    total: int              # Total comparisons
    correct: int            # Correct matches

class MetricsReport(BaseModel):
    timestamp: str
    model_version: str
    total_images: int
    fields: list[FieldMetrics]
    overall_accuracy: float
```

---

## CLI Interface

### `scripts/run_evaluation.py`

```bash
python scripts/run_evaluation.py [--dataset-path PATH] [--results-path PATH]
```

- Defaults from `Settings` if arguments not provided
- Exits 0 on success, 1 on failure
- Prints summary table to console via rich

---

## Config Interface

### `app/config.py`

```python
from app.config import get_settings

settings = get_settings()
settings.MISTRAL_API_KEY   # str (required)
settings.VISION_MODEL      # str = "mistral-small-latest"
settings.DATASET_PATH      # str = "data/"
settings.RESULTS_PATH      # str = "results/"
settings.HOST              # str = "0.0.0.0"
settings.PORT              # int = 8000
```
