"""Extractor module — sends images to Mistral Vision API for structured metadata extraction.

This module handles all interaction with the Mistral Vision API.
Each extraction includes timestamp and model version metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from app.config import get_settings
from app.retry import retry
from app.utils import get_image_mime_type, image_to_base64, timestamp_now

logger = logging.getLogger(__name__)


class ExtractionResult(BaseModel):
    """Structured metadata extracted from an image."""

    image_id: str
    timestamp: str
    model_version: str
    category: str | None = None
    colour: str | None = None
    material: str | None = None
    style: str | None = None
    raw_response: dict = {}


EXTRACTION_PROMPT = """Analyze this product image and extract the following metadata as JSON:
{
  "category": "product category (e.g., dress, jacket, sneakers, bag)",
  "colour": "primary colour (e.g., red, navy blue, black)",
  "material": "primary material (e.g., cotton, leather, polyester)",
  "style": "style descriptor (e.g., casual, formal, sporty)"
}

Return ONLY the JSON object, no additional text."""


@retry(max_retries=10, base_delay=2.0, max_delay=60.0)
def _call_mistral_vision(image_path: Path, model: str, api_key: str) -> dict:
    """Call Mistral Vision API with an image and return the response.

    Args:
        image_path: Path to the image file.
        model: Mistral model identifier.
        api_key: Mistral API key.

    Returns:
        Parsed JSON response from the model.
    """
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    base64_image = image_to_base64(image_path)
    mime_type = get_image_mime_type(image_path)

    response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{base64_image}",
                    },
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT,
                    },
                ],
            }
        ],
    )

    raw_text = response.choices[0].message.content
    # Parse JSON from response (handle markdown code blocks)
    if "```json" in raw_text:
        raw_text = raw_text.split("```json")[1].split("```")[0]
    elif "```" in raw_text:
        raw_text = raw_text.split("```")[1].split("```")[0]

    return json.loads(raw_text.strip())


def extract_metadata(image_path: Path) -> ExtractionResult:
    """Extract structured metadata from a single image.

    Args:
        image_path: Path to the image file.

    Returns:
        ExtractionResult with extracted fields and metadata.
    """
    settings = get_settings()

    logger.info("Extracting metadata from %s", image_path.name)
    parsed = _call_mistral_vision(
        image_path=image_path,
        model=settings.VISION_MODEL,
        api_key=settings.MISTRAL_API_KEY,
    )

    return ExtractionResult(
        image_id=image_path.stem,
        timestamp=timestamp_now(),
        model_version=settings.VISION_MODEL,
        category=parsed.get("category"),
        colour=parsed.get("colour"),
        material=parsed.get("material"),
        style=parsed.get("style"),
        raw_response=parsed,
    )


def extract_batch(image_paths: list[Path]) -> list[ExtractionResult]:
    """Extract metadata from multiple images sequentially.

    Args:
        image_paths: List of image file paths.

    Returns:
        List of ExtractionResult objects.
    """
    results = []
    for i, path in enumerate(image_paths, 1):
        logger.info("Processing image %d/%d: %s", i, len(image_paths), path.name)
        result = extract_metadata(path)
        results.append(result)
    return results
