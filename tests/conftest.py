"""Shared test fixtures for Mistral-Lens."""

import pytest


@pytest.fixture
def sample_extractions() -> list[dict]:
    """Sample extraction results for testing."""
    return [
        {
            "image_id": "img_001",
            "timestamp": "2026-03-12T00:00:00+00:00",
            "model_version": "mistral-small-latest",
            "category": "dress",
            "colour": "red",
            "material": "cotton",
            "style": "casual",
        },
        {
            "image_id": "img_002",
            "timestamp": "2026-03-12T00:00:01+00:00",
            "model_version": "mistral-small-latest",
            "category": "jacket",
            "colour": "black",
            "material": "leather",
            "style": "formal",
        },
        {
            "image_id": "img_003",
            "timestamp": "2026-03-12T00:00:02+00:00",
            "model_version": "mistral-small-latest",
            "category": "sneakers",
            "colour": "white",
            "material": "synthetic",
            "style": "sporty",
        },
    ]


@pytest.fixture
def sample_ground_truth() -> list[dict]:
    """Sample ground truth labels for testing."""
    return [
        {
            "image_id": "img_001",
            "category": "dress",
            "colour": "red",
            "material": "cotton",
            "style": "casual",
        },
        {
            "image_id": "img_002",
            "category": "jacket",
            "colour": "navy blue",  # Mismatch: "black" vs "navy blue"
            "material": "leather",
            "style": "formal",
        },
        {
            "image_id": "img_003",
            "category": "trainers",  # Mismatch: "sneakers" vs "trainers"
            "colour": "white",
            "material": "mesh",     # Mismatch: "synthetic" vs "mesh"
            "style": "sporty",
        },
    ]


@pytest.fixture
def default_fields() -> list[str]:
    """Default fields used for evaluation."""
    return ["category", "colour", "material", "style"]
