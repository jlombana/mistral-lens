"""Unit tests for the extractor module (mocked API)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.extractor import ExtractionResult, _ChatResult, extract_document


@pytest.fixture
def mock_pdf(tmp_path: Path) -> Path:
    """Create a fake PDF file for testing."""
    pdf = tmp_path / "test_doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    return pdf


class TestExtractDocument:
    """Tests for extract_document with mocked API calls."""

    @patch("app.extractor._call_chat")
    @patch("app.extractor._call_ocr")
    @patch("app.extractor.get_settings")
    def test_full_pipeline(self, mock_settings, mock_ocr, mock_chat, mock_pdf):
        """Should return a complete ExtractionResult with latency and tokens."""
        mock_settings.return_value = MagicMock(
            MISTRAL_API_KEY="test-key",
            OCR_MODEL="mistral-ocr-latest",
            CHAT_MODEL="mistral-large-latest",
        )
        mock_ocr.return_value = "Extracted text from PDF."
        mock_chat.side_effect = [
            _ChatResult("Document about testing.", prompt_tokens=100, completion_tokens=10),
            _ChatResult("The answer is 42.", prompt_tokens=120, completion_tokens=15),
        ]

        result = extract_document(mock_pdf, question="What is the answer?")

        assert isinstance(result, ExtractionResult)
        assert result.document_id == "test_doc"
        assert result.extracted_text == "Extracted text from PDF."
        assert result.topic == "Document about testing."
        assert result.answer == "The answer is 42."
        assert result.question == "What is the answer?"
        assert result.ocr_model == "mistral-ocr-latest"
        assert result.chat_model == "mistral-large-latest"
        assert result.timestamp
        assert result.tokens_prompt == 220
        assert result.tokens_completion == 25
        assert result.tokens_total == 245
        assert result.latency_total_s >= 0

    @patch("app.extractor._call_chat")
    @patch("app.extractor._call_ocr")
    @patch("app.extractor.get_settings")
    def test_without_question(self, mock_settings, mock_ocr, mock_chat, mock_pdf):
        """Should work without a question (answer is None)."""
        mock_settings.return_value = MagicMock(
            MISTRAL_API_KEY="test-key",
            OCR_MODEL="mistral-ocr-latest",
            CHAT_MODEL="mistral-large-latest",
        )
        mock_ocr.return_value = "Some text."
        mock_chat.return_value = _ChatResult("Some topic.", prompt_tokens=80, completion_tokens=5)

        result = extract_document(mock_pdf)

        assert result.answer is None
        assert result.question is None
        assert result.extracted_text == "Some text."
        assert result.topic == "Some topic."
        assert result.tokens_prompt == 80
        assert result.tokens_completion == 5
        assert result.latency_answer_s == 0.0

    @patch("app.extractor._call_chat")
    @patch("app.extractor._call_ocr")
    @patch("app.extractor.get_settings")
    def test_custom_document_id(self, mock_settings, mock_ocr, mock_chat, mock_pdf):
        """Should use custom document_id when provided."""
        mock_settings.return_value = MagicMock(
            MISTRAL_API_KEY="test-key",
            OCR_MODEL="mistral-ocr-latest",
            CHAT_MODEL="mistral-large-latest",
        )
        mock_ocr.return_value = "Text."
        mock_chat.return_value = _ChatResult("Topic.", prompt_tokens=50, completion_tokens=3)

        result = extract_document(mock_pdf, document_id="custom-id")

        assert result.document_id == "custom-id"
