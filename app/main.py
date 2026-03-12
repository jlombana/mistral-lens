"""Entry point for Mistral-Lens evaluation pipeline.

Orchestrates: load dataset → extract metadata → compute metrics → save results.
No business logic here — delegates to extractor and metrics modules.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.extractor import extract_batch
from app.metrics import MetricsReport, compute_metrics
from app.utils import load_json, save_json, timestamp_now

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_FIELDS = ["category", "colour", "material", "style"]


def discover_images(dataset_path: Path) -> list[Path]:
    """Find all image files in the dataset directory.

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        Sorted list of image file paths.
    """
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [
        p for p in dataset_path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(images)


def load_ground_truth(dataset_path: Path) -> list[dict]:
    """Load ground truth labels from a JSON file in the dataset directory.

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        List of ground truth dicts with 'image_id' keys.
    """
    gt_path = dataset_path / "ground_truth.json"
    if not gt_path.exists():
        console.print("[yellow]Warning: ground_truth.json not found[/yellow]")
        return []
    return load_json(gt_path)


def print_report(report: MetricsReport) -> None:
    """Print metrics report as a rich table.

    Args:
        report: MetricsReport to display.
    """
    table = Table(title="Mistral-Lens Evaluation Results")
    table.add_column("Field", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")

    for fm in report.fields:
        table.add_row(
            fm.field,
            f"{fm.accuracy:.1%}",
            str(fm.correct),
            str(fm.total),
        )

    table.add_section()
    table.add_row(
        "OVERALL",
        f"{report.overall_accuracy:.1%}",
        str(sum(fm.correct for fm in report.fields)),
        str(sum(fm.total for fm in report.fields)),
        style="bold",
    )

    console.print(table)
    console.print(f"\nModel: {report.model_version}")
    console.print(f"Images: {report.total_images}")
    console.print(f"Timestamp: {report.timestamp}")


def run_evaluation(
    dataset_path: Path | None = None,
    results_path: Path | None = None,
    fields: list[str] | None = None,
) -> MetricsReport:
    """Run the full evaluation pipeline.

    Args:
        dataset_path: Path to dataset directory (defaults to settings).
        results_path: Path to results directory (defaults to settings).
        fields: Fields to evaluate (defaults to DEFAULT_FIELDS).

    Returns:
        MetricsReport with evaluation results.
    """
    settings = get_settings()
    dataset_path = dataset_path or settings.dataset_dir
    results_path = results_path or settings.results_dir
    fields = fields or DEFAULT_FIELDS

    console.print(f"[bold]Mistral-Lens Evaluation[/bold]")
    console.print(f"Dataset: {dataset_path}")
    console.print(f"Model: {settings.VISION_MODEL}\n")

    # Discover images
    images = discover_images(dataset_path)
    if not images:
        console.print("[red]No images found in dataset directory[/red]")
        sys.exit(1)

    console.print(f"Found {len(images)} images\n")

    # Extract metadata
    console.print("[bold]Extracting metadata...[/bold]")
    results = extract_batch(images)
    extractions = [r.model_dump() for r in results]

    # Load ground truth
    ground_truth = load_ground_truth(dataset_path)

    # Compute metrics
    console.print("[bold]Computing metrics...[/bold]\n")
    report = compute_metrics(
        extractions=extractions,
        ground_truth=ground_truth,
        fields=fields,
        model_version=settings.VISION_MODEL,
    )

    # Print report
    print_report(report)

    # Save results
    results_path.mkdir(parents=True, exist_ok=True)
    output_file = results_path / f"evaluation_{timestamp_now().replace(':', '-')}.json"
    save_json(
        {
            "report": report.model_dump(),
            "extractions": extractions,
        },
        output_file,
    )
    console.print(f"\n[green]Results saved to {output_file}[/green]")

    return report
