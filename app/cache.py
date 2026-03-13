"""PDF result cache — avoids re-processing already-seen documents.

Cache key is the MD5 hash of the raw PDF binary. Results are stored
as individual JSON files under results/cache/<hash>.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path("results/cache")


def _pdf_hash(pdf_path: Path) -> str:
    """Compute MD5 hash of a PDF file."""
    return hashlib.md5(pdf_path.read_bytes()).hexdigest()


def _cache_path(pdf_hash: str) -> Path:
    return CACHE_DIR / f"{pdf_hash}.json"


def load_from_cache(pdf_path: Path) -> dict[str, Any] | None:
    """Load cached extraction result for a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Cached result dict, or None if not cached.
    """
    h = _pdf_hash(pdf_path)
    p = _cache_path(h)
    if p.exists():
        logger.info("Cache hit for %s (hash=%s)", pdf_path.name, h)
        return json.loads(p.read_text())
    return None


def save_to_cache(pdf_path: Path, result: dict[str, Any]) -> None:
    """Save extraction result to cache.

    Args:
        pdf_path: Path to the PDF file.
        result: Extraction result dict to cache.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = _pdf_hash(pdf_path)
    p = _cache_path(h)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    logger.info("Cached result for %s (hash=%s)", pdf_path.name, h)


def get_cache_stats() -> dict[str, Any]:
    """Return cache statistics.

    Returns:
        Dict with count, total_cost_saved, and total_docs.
    """
    if not CACHE_DIR.exists():
        return {"count": 0, "cost_saved": 0.0}

    count = 0
    cost_saved = 0.0
    for f in CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            count += 1
            cost_saved += data.get("cost_usd", 0.0)
        except (json.JSONDecodeError, OSError):
            continue

    return {"count": count, "cost_saved": round(cost_saved, 4)}
