"""Shared test fixtures for Mistral-Lens."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_extraction() -> dict:
    """Sample extraction result for testing."""
    return {
        "document_id": "doc_001",
        "timestamp": "2026-03-12T00:00:00+00:00",
        "ocr_model": "mistral-ocr-latest",
        "chat_model": "mistral-large-latest",
        "extracted_text": "The quick brown fox jumps over the lazy dog.",
        "topic": "A fox jumping over a dog in a field.",
        "answer": "The fox jumped over the dog because it was in its path.",
        "question": "Why did the fox jump over the dog?",
    }


@pytest.fixture
def sample_ground_truth() -> dict:
    """Sample ground truth for testing."""
    return {
        "document_id": "doc_001",
        "text": "The quick brown fox jumps over the lazy dog.",
        "topic": "An animal encounter involving a fox and a dog.",
        "answer": "The fox jumped over the dog to continue on its way.",
        "question": "Why did the fox jump over the dog?",
    }


@pytest.fixture
def sample_extractions() -> list[dict]:
    """Batch of sample extraction results."""
    return [
        {
            "document_id": "doc_001",
            "extracted_text": "The quick brown fox jumps over the lazy dog.",
            "topic": "A fox jumping over a dog.",
            "answer": "The fox jumped over the dog.",
            "question": "Why did the fox jump?",
        },
        {
            "document_id": "doc_002",
            "extracted_text": "Climate change is affecting global weather patterns.",
            "topic": "Climate change and weather.",
            "answer": "Rising temperatures cause extreme weather events.",
            "question": "How does climate change affect weather?",
        },
    ]


@pytest.fixture
def sample_ground_truths() -> list[dict]:
    """Batch of sample ground truth data."""
    return [
        {
            "document_id": "doc_001",
            "text": "The quick brown fox jumps over the lazy dog.",
            "topic": "An animal encounter involving a fox and a dog.",
            "answer": "The fox jumped over the dog to continue on its way.",
            "question": "Why did the fox jump?",
        },
        {
            "document_id": "doc_002",
            "text": "Climate change is affecting global weather patterns significantly.",
            "topic": "Impact of climate change on global weather systems.",
            "answer": "Climate change leads to more frequent extreme weather events.",
            "question": "How does climate change affect weather?",
        },
    ]
