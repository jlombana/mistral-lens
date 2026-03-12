"""Unit tests for the metrics engine."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from app.metrics import (
    TextMetrics,
    compute_text_metrics,
    compute_wer,
    compute_rouge_l,
)


class TestComputeWer:
    """Tests for WER computation."""

    def test_perfect_match(self):
        """Identical texts should have WER = 0."""
        assert compute_wer("hello world", "hello world") == 0.0

    def test_complete_mismatch(self):
        """Completely different texts should have high WER."""
        wer = compute_wer("hello world", "foo bar baz")
        assert wer > 0.5

    def test_empty_reference(self):
        """Empty reference should return 1.0."""
        assert compute_wer("", "some text") == 1.0

    def test_empty_hypothesis(self):
        """Empty hypothesis should return 1.0."""
        assert compute_wer("some text", "") == 1.0

    def test_partial_match(self):
        """Partial match should have WER between 0 and 1."""
        wer = compute_wer("the quick brown fox", "the quick red fox")
        assert 0.0 < wer < 1.0


class TestComputeRougeL:
    """Tests for ROUGE-L computation."""

    def test_perfect_match(self):
        """Identical texts should have ROUGE-L = 1.0."""
        score = compute_rouge_l("hello world", "hello world")
        assert score == 1.0

    def test_no_overlap(self):
        """Non-overlapping texts should have low ROUGE-L."""
        score = compute_rouge_l("hello world", "foo bar")
        assert score < 0.5

    def test_empty_reference(self):
        """Empty reference should return 0.0."""
        assert compute_rouge_l("", "some text") == 0.0

    def test_empty_hypothesis(self):
        """Empty hypothesis should return 0.0."""
        assert compute_rouge_l("some text", "") == 0.0

    def test_partial_overlap(self):
        """Partial overlap should have ROUGE-L between 0 and 1."""
        score = compute_rouge_l("the quick brown fox", "the quick red fox")
        assert 0.0 < score < 1.0


class TestComputeTextMetrics:
    """Tests for compute_text_metrics."""

    def test_returns_text_metrics(self):
        """Should return a TextMetrics object."""
        result = compute_text_metrics("doc1", "hello world", "hello world")
        assert isinstance(result, TextMetrics)
        assert result.document_id == "doc1"
        assert result.wer == 0.0
        assert result.rouge_l == 1.0

    def test_mismatched_text(self):
        """Should compute non-zero WER for mismatched texts."""
        result = compute_text_metrics("doc1", "hello world", "hello there")
        assert result.wer > 0.0
        assert result.rouge_l < 1.0
