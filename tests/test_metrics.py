"""Unit tests for the metrics engine."""

from app.metrics import MetricsReport, compute_field_accuracy, compute_metrics


class TestComputeFieldAccuracy:
    """Tests for compute_field_accuracy."""

    def test_perfect_match(self, sample_extractions, sample_ground_truth):
        """All values match for the 'style' field."""
        result = compute_field_accuracy(
            sample_extractions, sample_ground_truth, "style"
        )
        assert result.accuracy == 1.0
        assert result.correct == 3
        assert result.total == 3

    def test_partial_match(self, sample_extractions, sample_ground_truth):
        """Some values match for the 'colour' field."""
        result = compute_field_accuracy(
            sample_extractions, sample_ground_truth, "colour"
        )
        assert result.correct == 2  # img_001 and img_003 match
        assert result.total == 3
        assert abs(result.accuracy - 2 / 3) < 1e-9

    def test_no_match(self):
        """No values match at all."""
        extractions = [{"image_id": "a", "colour": "red"}]
        ground_truth = [{"image_id": "a", "colour": "blue"}]
        result = compute_field_accuracy(extractions, ground_truth, "colour")
        assert result.accuracy == 0.0
        assert result.correct == 0

    def test_missing_field_in_extraction(self):
        """Field missing from extraction should not count as correct."""
        extractions = [{"image_id": "a"}]
        ground_truth = [{"image_id": "a", "colour": "red"}]
        result = compute_field_accuracy(extractions, ground_truth, "colour")
        assert result.correct == 0
        assert result.total == 1

    def test_missing_field_in_ground_truth(self):
        """Field missing from ground truth should be skipped."""
        extractions = [{"image_id": "a", "colour": "red"}]
        ground_truth = [{"image_id": "a"}]
        result = compute_field_accuracy(extractions, ground_truth, "colour")
        assert result.total == 0

    def test_empty_inputs(self):
        """Empty inputs should return zero accuracy."""
        result = compute_field_accuracy([], [], "colour")
        assert result.accuracy == 0.0
        assert result.total == 0

    def test_case_insensitive_comparison(self):
        """Values should be compared case-insensitively."""
        extractions = [{"image_id": "a", "colour": "Red"}]
        ground_truth = [{"image_id": "a", "colour": "red"}]
        result = compute_field_accuracy(extractions, ground_truth, "colour")
        assert result.correct == 1

    def test_whitespace_normalization(self):
        """Leading/trailing whitespace should be stripped."""
        extractions = [{"image_id": "a", "colour": " red "}]
        ground_truth = [{"image_id": "a", "colour": "red"}]
        result = compute_field_accuracy(extractions, ground_truth, "colour")
        assert result.correct == 1


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_report_structure(self, sample_extractions, sample_ground_truth, default_fields):
        """Report should have correct structure and types."""
        report = compute_metrics(
            sample_extractions, sample_ground_truth, default_fields, "test-model"
        )
        assert isinstance(report, MetricsReport)
        assert report.model_version == "test-model"
        assert report.total_images == 3
        assert len(report.fields) == 4
        assert report.timestamp

    def test_overall_accuracy(self, sample_extractions, sample_ground_truth, default_fields):
        """Overall accuracy should reflect aggregate correctness."""
        report = compute_metrics(
            sample_extractions, sample_ground_truth, default_fields
        )
        # style: 3/3, colour: 2/3, category: 2/3, material: 2/3 = 9/12 = 0.75
        assert abs(report.overall_accuracy - 9 / 12) < 1e-9

    def test_empty_extractions(self, default_fields):
        """Empty extractions should produce zero accuracy."""
        report = compute_metrics([], [], default_fields)
        assert report.overall_accuracy == 0.0
        assert report.total_images == 0

    def test_single_field(self, sample_extractions, sample_ground_truth):
        """Should work with a single field."""
        report = compute_metrics(
            sample_extractions, sample_ground_truth, ["style"]
        )
        assert len(report.fields) == 1
        assert report.overall_accuracy == 1.0
