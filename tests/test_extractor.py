"""Unit tests for the extractor module (mocked API)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.extractor import ExtractionResult, extract_metadata


@pytest.fixture
def mock_image(tmp_path: Path) -> Path:
    """Create a fake image file for testing."""
    img = tmp_path / "test_image.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Minimal JPEG header
    return img


@pytest.fixture
def mock_mistral_response() -> dict:
    """Mock Mistral API response."""
    return {
        "category": "dress",
        "colour": "red",
        "material": "cotton",
        "style": "casual",
    }


class TestExtractMetadata:
    """Tests for extract_metadata with mocked API."""

    @patch("app.extractor._call_mistral_vision")
    @patch("app.extractor.get_settings")
    def test_returns_extraction_result(
        self, mock_settings, mock_call, mock_image, mock_mistral_response
    ):
        """Should return a valid ExtractionResult."""
        mock_settings.return_value = MagicMock(
            MISTRAL_API_KEY="test-key",
            VISION_MODEL="test-model",
        )
        mock_call.return_value = mock_mistral_response

        result = extract_metadata(mock_image)

        assert isinstance(result, ExtractionResult)
        assert result.image_id == "test_image"
        assert result.model_version == "test-model"
        assert result.category == "dress"
        assert result.colour == "red"
        assert result.material == "cotton"
        assert result.style == "casual"
        assert result.timestamp

    @patch("app.extractor._call_mistral_vision")
    @patch("app.extractor.get_settings")
    def test_handles_missing_fields(self, mock_settings, mock_call, mock_image):
        """Should handle missing fields gracefully."""
        mock_settings.return_value = MagicMock(
            MISTRAL_API_KEY="test-key",
            VISION_MODEL="test-model",
        )
        mock_call.return_value = {"category": "jacket"}

        result = extract_metadata(mock_image)

        assert result.category == "jacket"
        assert result.colour is None
        assert result.material is None
        assert result.style is None

    @patch("app.extractor._call_mistral_vision")
    @patch("app.extractor.get_settings")
    def test_includes_raw_response(
        self, mock_settings, mock_call, mock_image, mock_mistral_response
    ):
        """Should include the raw API response."""
        mock_settings.return_value = MagicMock(
            MISTRAL_API_KEY="test-key",
            VISION_MODEL="test-model",
        )
        mock_call.return_value = mock_mistral_response

        result = extract_metadata(mock_image)

        assert result.raw_response == mock_mistral_response
