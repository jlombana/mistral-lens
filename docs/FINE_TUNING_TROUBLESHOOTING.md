# Fine-Tuning Troubleshooting (Mistral API)

## Current blocker

If you see:

`Model not available for this type of fine-tuning (...)`

the issue is typically **entitlements on your Mistral project/account**, not code or connectivity.

## Verify with preflight

Use the project script with existing uploaded file IDs:

```bash
python3 scripts/run_finetune.py \
  --train-file-id <TRAIN_FILE_ID> \
  --val-file-id <VAL_FILE_ID> \
  --model open-mistral-nemo \
  --job-type completion
```

Try also:

```bash
python3 scripts/run_finetune.py \
  --train-file-id <TRAIN_FILE_ID> \
  --val-file-id <VAL_FILE_ID> \
  --model open-mistral-nemo \
  --job-type classifier
```

The script runs a `dry_run=true` preflight first.

## What to ask Mistral support

Request enablement for your project with:

- Fine-tuning entitlement enabled
- Target job type(s): `completion` and/or `classifier`
- Target model(s): `open-mistral-nemo` (and any alternatives you need)

Include:

- Your project/account identifier
- The exact error message returned by `POST /v1/fine_tuning/jobs`
- The model and job type used

