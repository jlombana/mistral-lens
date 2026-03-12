"""Shared utilities for Mistral-Lens."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp_now() -> str:
    """Return current UTC timestamp in ISO 8601 format.

    Returns:
        ISO 8601 formatted timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def pdf_to_base64(pdf_path: Path) -> str:
    """Encode a PDF file to base64 string.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Base64 encoded string of the PDF.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return base64.b64encode(pdf_path.read_bytes()).decode("utf-8")


def save_json(data: Any, path: Path) -> None:
    """Save data as formatted JSON file.

    Args:
        data: Data to serialize (must be JSON-serializable).
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def load_json(path: Path) -> Any:
    """Load JSON file and return parsed data.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text())
