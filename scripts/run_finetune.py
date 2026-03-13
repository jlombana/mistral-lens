#!/usr/bin/env python3
"""Launch a Mistral fine-tuning job with preflight checks.

This script supports:
- Uploading training/validation JSONL files (or reusing existing file IDs)
- `job_type` selection (`completion` or `classifier`)
- API preflight via `dry_run=true` before creating the real job
- Actionable diagnostics for entitlement/model availability errors

Usage examples:
    python scripts/run_finetune.py
    python scripts/run_finetune.py --job-type classifier --model mistral-large-latest
    python scripts/run_finetune.py --train-file-id <id> --val-file-id <id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
MISTRAL_API_BASE = "https://api.mistral.ai/v1"


def _api_request(
    *,
    api_key: str,
    method: str,
    path: str,
    payload: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any] | str]:
    """Perform a JSON API request against Mistral endpoints."""
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(
        url=f"{MISTRAL_API_BASE}{path}",
        method=method,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            try:
                return resp.getcode(), json.loads(body)
            except json.JSONDecodeError:
                return resp.getcode(), body
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Mistral fine-tuning job")
    parser.add_argument("--model", default="open-mistral-nemo")
    parser.add_argument(
        "--job-type",
        default="completion",
        choices=["completion", "classifier"],
        help="Fine-tuning task type.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--suffix", default="topic-classifier")
    parser.add_argument(
        "--train-path",
        default=str(DATA_DIR / "topic_finetune_train.jsonl"),
        help="Training JSONL path if --train-file-id is not provided.",
    )
    parser.add_argument(
        "--val-path",
        default=str(DATA_DIR / "topic_finetune_val.jsonl"),
        help="Validation JSONL path if --val-file-id is not provided.",
    )
    parser.add_argument("--train-file-id", default=None)
    parser.add_argument("--val-file-id", default=None)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip dry-run validation call.",
    )
    return parser.parse_args()


def _upload_file_if_needed(
    *,
    client: Any,
    provided_file_id: str | None,
    file_path: Path,
    label: str,
) -> str:
    """Upload file unless a file_id was provided."""
    if provided_file_id:
        print(f"{label} file ID provided: {provided_file_id}")
        return provided_file_id

    if not file_path.exists():
        raise FileNotFoundError(f"{label} file not found: {file_path}")

    print(f"Uploading {label} file: {file_path}")
    with open(file_path, "rb") as f:
        uploaded = client.files.upload(
            file={"file_name": file_path.name, "content": f.read()},
            purpose="fine-tune",
        )
    print(f"  Uploaded {label} file ID: {uploaded.id}")
    return uploaded.id


def _extract_detail(payload: Dict[str, Any] | str) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            return json.dumps(detail, ensure_ascii=False)
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _print_entitlement_guidance(job_type: str, model: str, detail: str) -> None:
    print("\nERROR: Fine-tuning request rejected by API.")
    print(f"  job_type: {job_type}")
    print(f"  model: {model}")
    print(f"  detail: {detail}")
    print("\nLikely cause:")
    print("  Your account/project does not currently have an eligible model")
    print("  for this fine-tuning type (entitlement issue, not connectivity).")
    print("\nAsk Mistral support to enable fine-tuning for your project and job type.")
    print("Include:")
    print(f"  - Requested job_type: {job_type}")
    print(f"  - Requested model: {model}")
    print("  - Error: 'Model not available for this type of fine-tuning ...'")


def main() -> None:
    from mistralai import Mistral

    args = _parse_args()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: MISTRAL_API_KEY not set")
        sys.exit(1)

    client = Mistral(api_key=api_key)

    try:
        train_file_id = _upload_file_if_needed(
            client=client,
            provided_file_id=args.train_file_id,
            file_path=Path(args.train_path),
            label="Train",
        )
        val_file_id = _upload_file_if_needed(
            client=client,
            provided_file_id=args.val_file_id,
            file_path=Path(args.val_path),
            label="Validation",
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    payload: Dict[str, Any] = {
        "job_type": args.job_type,
        "model": args.model,
        "training_files": [{"file_id": train_file_id, "weight": 1}],
        "validation_files": [val_file_id],
        "hyperparameters": {
            "training_steps": args.steps,
            "learning_rate": args.learning_rate,
        },
        "suffix": args.suffix,
    }

    if not args.skip_preflight:
        dry_payload = dict(payload)
        dry_payload["dry_run"] = True
        print("\nPreflight check (dry_run=true)...")
        code, resp = _api_request(
            api_key=api_key,
            method="POST",
            path="/fine_tuning/jobs",
            payload=dry_payload,
        )
        if code >= 400:
            detail = _extract_detail(resp)
            if "Model not available for this type of fine-tuning" in detail:
                _print_entitlement_guidance(args.job_type, args.model, detail)
            else:
                print(f"ERROR: Preflight failed (HTTP {code})")
                print(f"  detail: {detail}")
            sys.exit(2)
        print("  Preflight OK")

    print("\nLaunching fine-tuning job...")
    code, resp = _api_request(
        api_key=api_key,
        method="POST",
        path="/fine_tuning/jobs",
        payload=payload,
    )
    if code >= 400:
        detail = _extract_detail(resp)
        print(f"ERROR: Job creation failed (HTTP {code})")
        print(f"  detail: {detail}")
        if "Model not available for this type of fine-tuning" in detail:
            _print_entitlement_guidance(args.job_type, args.model, detail)
        sys.exit(2)

    if not isinstance(resp, dict):
        print("ERROR: Unexpected API response format.")
        print(resp)
        sys.exit(2)

    job_id = str(resp.get("id", "")).strip()
    status = resp.get("status", "unknown")
    model = resp.get("model", args.model)

    print("\nFine-tuning job created.")
    print(f"  Job ID: {job_id}")
    print(f"  Status: {status}")
    print(f"  Model: {model}")

    if job_id:
        job_file = DATA_DIR / "finetune_job.txt"
        job_file.write_text(job_id + "\n")
        print(f"\nJob ID saved to {job_file}")


if __name__ == "__main__":
    main()
