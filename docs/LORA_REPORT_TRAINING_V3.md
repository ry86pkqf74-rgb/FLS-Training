# LoRA Report Training V3

Do not train or continue training a report LoRA on old markdown reports without relabeling. Legacy reports may contain hard-coded denominators, unsupported readiness language, and z-score framing that v003 forbids.

## Label Flow

1. Load existing score JSON records.
2. Load the task rubric.
3. Normalize scores into the v003 schema.
4. Recompute score math centrally.
5. Generate report v3 JSON.
6. Validate the report.
7. Save the validated structured report as the training target.

Scripts:

- `scripts/060_generate_report_v3_labels.py`
- `scripts/061_validate_report_v3_labels.py`
- `scripts/062_prepare_lora_report_v3_dataset.py`
- `scripts/063_train_report_lora_v3.py`

## Re-Score Triggers

Only rescore video when:

- `task_id` is missing
- task-specific assessments are missing
- critical errors are unclear
- score math cannot be reconciled
- confidence is below 0.60

## Rejection Rules

Reject any label if:

- formula does not match score
- denominator is wrong
- report uses readiness language despite critical error
- report says no penalties despite penalties greater than zero
- z-score appears in primary assessment
- task-specific priority is missing
- improvement advice is generic
- report contains unsupported observations

The report LoRA should learn structured report JSON. Markdown should be rendered from structured fields whenever possible.
