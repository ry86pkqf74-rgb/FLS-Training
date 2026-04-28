# Reporting V3

Reporting v003 makes every report task-aware, rubric-faithful, clinically cautious, and coaching-rich.

## Architecture

- `src/rubrics/loader.py` is the source of truth for task names, denominators, timing limits, penalties, and official/custom task labels.
- `src.scoring.frontier_scorer.recompute_score_from_components` recomputes score math from rubric max score, completion time, and penalties.
- `src.reporting.readiness.determine_readiness` gates readiness language before any report text is rendered.
- `src.reporting.report_v3.generate_report_v3` creates structured report JSON.
- `src.reporting.render_markdown_v3.render_markdown_v3` renders markdown from structured fields.
- `src.reporting.validator.validate_report_v3` rejects denominator, math, contradiction, z-score, and specificity errors.

## Task 6 Policy

Task 6 remains fully supported as a platform training task:

- task id: `task6`
- canonical task: `task6_rings_needle_manipulation`
- display name: Rings of Rings Needle Manipulation
- label: custom FLS-adjacent training task
- denominator: 315
- certification eligible: false

Task 6 reports must use training readiness language, not official FLS pass/fail language.

## Readiness Labels

Reports use exact labels from `determine_readiness`:

- `unscorable`
- `automatic_fail`
- `needs_human_review`
- `needs_focused_remediation`
- `borderline`
- `on_track_for_training`
- `meets_local_training_target`

Readiness labels are not certification outcomes.

## Prohibited Primary Language

Do not use unsupported phrases such as "pass likely", "clinically competent", "above average", or "no significant penalties" unless the structured gate and evidence allow it. Reports with critical errors should focus on correctness before speed.
