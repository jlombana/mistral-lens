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


def image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64 string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 encoded string of the image.

    Raises:
        FileNotFoundError: If image_path does not exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def get_image_mime_type(image_path: Path) -> str:
    """Determine MIME type from image file extension.

    Args:
        image_path: Path to the image file.

    Returns:
        MIME type string (e.g., 'image/jpeg').
    """
    suffix_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return suffix_map.get(image_path.suffix.lower(), "image/jpeg")


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
