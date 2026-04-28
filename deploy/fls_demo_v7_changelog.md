# fls_demo_v7 (v003 deployment) — 2026-04-27

Live at http://38.242.238.209:7860/ on Contabo `m26209.contaboserver.net`.

## Patches applied to v5 to produce v7

1. `ADAPTER_PATH` flipped from `/opt/fls/adapters/v16` → `/opt/fls/adapters/v17_v003`.
2. `TASK_MAX_SCORES` corrected: `task3_endoloop` 300 → **180**, `task4_extracorporeal_knot` 300 → **420**. Task 1/2/5/6 already correct.
3. `TASK_SCORING_NOTES` rewritten for task 3 + 4 to match the corrected denominators and drop the legacy "Proficient cutoff" language.
4. Header re-labelled `FLS Video Scoring Platform v7 (v003)`.
5. `Qwen2.5-VL-7B + v16 multimodal LoRA` → `Qwen2.5-VL-7B + v17 multimodal LoRA (v003 schema)` in the docstring.
6. v003 readiness gating replaces the legacy `pct >= 75 → "excellent"` ladder. Critical findings (any non-empty error flags) force `needs_focused_remediation` regardless of percentage; otherwise the percentage is mapped to `meets_local_training_target / on_track_for_training / borderline / needs_focused_remediation`.
7. Narrative builder language switched from "Excellent / Proficient / Developing" to "Strong / On-track / Borderline / Focused-remediation" *training-score* framing. Every overall message now ends with the v003 disclaimer:
   > This is AI-assisted training feedback, not an official FLS certification result.
8. GRS z-scores moved under an explicit experimental block:
   - `Global Rating Scale z-score:` lines now print under "Experimental AI-derived metrics (not part of official FLS scoring)".
   - The component-score table is renamed "Experimental AI-derived metrics (not part of official FLS scoring) — GRS z-scores".

## Backups left on the box

- `/opt/fls/fls_demo_v5_pre_v003_backup.py` — pre-patch v5.
- `/opt/fls/fls_demo_v5_active.py.bak` — the exact v5 file that was running before swap.
- `/opt/fls/results.db.pre_v003.bak` — copy of the resident DB at the moment of swap.

## Rollback

If anything regresses, on the box:

```bash
kill $(pgrep -fa python3
