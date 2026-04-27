#!/usr/bin/env python3
"""Build the v19 multimodal training dataset.

Two changes over the v18 dataset (``030d_build_v003_pod_dataset.py``):

1. **Length-clip the assistant JSON.** v18's outputs grew to 2k+ characters
   and got truncated mid-object at the eval's max_new_tokens=1024. We trim
   verbose narrative fields (``technique_summary``, ``improvement_suggestions``,
   ``task_specific_assessments.frame_analyses``, etc.) so a typical
   v19 assistant message comes in under ~700 tokens / ~2200 characters.
2. **Upsample critical-error examples.** Only ~7 of the 193 training rows
   carry non-empty ``critical_errors``. To teach the LoRA the v003 contract
   without a custom loss, we duplicate those rows N=5 times so the model
   sees them ~25× more often than baseline. v18's 71% critical-error F1 came
   from ~3% of training data; with upsampling we can get there with shorter
   training.

Output: ``/workspace/v003_multimodal_v19/{train,val,test}.jsonl``. Test set
is left unchanged so v17/v18/v19 are evaluated on identical examples.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path("/workspace/FLS-Training")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Maximum number of characters in the assistant JSON. ~3.3 chars per Qwen
# token → 2300 chars ≈ 700 tokens. Leaves headroom for the assistant header
# and ample room under a 1024-token decode budget.
ASSISTANT_CHAR_BUDGET = 2300

# Fields we trim hard: long free-text and arrays that are nice-to-have but
# not load-bearing for the v003 contract.
TRIMMABLE_TEXT_FIELDS = ("technique_summary",)
TRIMMABLE_LIST_FIELDS = (
    "strengths",
    "improvement_suggestions",
    "cannot_determine",
)
# task_specific_assessments dict can hold large frame_analyses / phase_timings;
# strip the heaviest sub-fields if the JSON is still over budget.
HEAVY_TSA_FIELDS = ("frame_analyses", "phase_timings", "phases_detected")

CRITICAL_UPSAMPLE_FACTOR = 5
SEED = 42


def _truncate_str(s: str, budget: int) -> str:
    if len(s) <= budget:
        return s
    return s[: max(0, budget - 1)].rstrip() + "…"


def _trim_target(target: dict) -> dict:
    """Mutate-and-return a copy of the target with verbose fields trimmed."""
    t = json.loads(json.dumps(target))  # deep copy

    # Pass 1: clip free-text fields.
    if t.get("technique_summary"):
        t["technique_summary"] = _truncate_str(t["technique_summary"], 250)

    for f in TRIMMABLE_LIST_FIELDS:
        if isinstance(t.get(f), list):
            t[f] = [_truncate_str(str(x), 140) for x in t[f][:3]]

    # Trim penalty descriptions but keep counts/severity intact.
    for p in t.get("penalties") or []:
        if isinstance(p, dict) and isinstance(p.get("description"), str):
            p["description"] = _truncate_str(p["description"], 140)
        if isinstance(p, dict) and isinstance(p.get("frame_evidence"), list):
            p["frame_evidence"] = p["frame_evidence"][:6]

    # Heavy task_specific_assessments: keep keys but cap nested arrays.
    tsa = t.get("task_specific_assessments")
    if isinstance(tsa, dict):
        for f in HEAVY_TSA_FIELDS:
            if isinstance(tsa.get(f), list):
                tsa[f] = tsa[f][:6]

    # Pass 2: if still over budget, drop heavy_tsa fields entirely, then
    # technique_summary, then strengths/suggestions.
    serialized = json.dumps(t, default=str)
    if len(serialized) > ASSISTANT_CHAR_BUDGET and isinstance(tsa, dict):
        for f in HEAVY_TSA_FIELDS:
            tsa.pop(f, None)
        serialized = json.dumps(t, default=str)
    if len(serialized) > ASSISTANT_CHAR_BUDGET:
        t.pop("technique_summary", None)
        serialized = json.dumps(t, default=str)
    if len(serialized) > ASSISTANT_CHAR_BUDGET:
        for f in TRIMMABLE_LIST_FIELDS:
            t[f] = []
        serialized = json.dumps(t, default=str)

    return t


def _has_critical(target: dict) -> bool:
    errors = target.get("critical_errors") or []
    if isinstance(errors, list) and any(
        isinstance(e, dict) and e.get("present") is not False for e in errors
    ):
        return True
    # Penalty severity ≥ major counts as a teachable critical-error pattern.
    for p in target.get("penalties") or []:
        if isinstance(p, dict) and (p.get("severity") in {"major", "critical", "auto_fail"}):
            return True
    return False


def _process_split(in_path: Path, out_path: Path, upsample: bool) -> tuple[int, int]:
    if not in_path.exists():
        return 0, 0
    rows_in = 0
    rows_out = 0
    critical_count = 0
    with in_path.open() as fh, out_path.open("w") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_in += 1

            # Trim the assistant turn.
            assistant = row["messages"][-1]
            target = json.loads(assistant["content"])
            target = _trim_target(target)
            assistant["content"] = json.dumps(target, default=str)

            out.write(json.dumps(row) + "\n")
            rows_out += 1

            if upsample and _has_critical(target):
                critical_count += 1
                for _ in range(CRITICAL_UPSAMPLE_FACTOR - 1):
                    out.write(json.dumps(row) + "\n")
                    rows_out += 1

    return rows_in, rows_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("/workspace/v003_multimodal"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/v003_multimodal_v19"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_in, train_out = _process_split(
        args.source_dir / "train.jsonl",
        args.output_dir / "train.jsonl",
        upsample=True,
    )
    val_in, val_out = _process_split(
        args.source_dir / "val.jsonl",
        args.output_dir / "val.jsonl",
        upsample=False,
    )
    test_in, test_out = _process_split(
        args.source_dir / "test.jsonl",
        args.output_dir / "test.jsonl",
        upsample=False,
    )

    # Shuffle train.jsonl after upsampling so duplicates aren't adjacent.
    train_path = args.output_dir / "train.jsonl"
    rows = train_path.read_text().splitlines()
    random.Random(SEED).shuffle(rows)
    train_path.write_text("\n".join(rows) + "\n")

    manifest = {
        "dataset": "v003_multimodal_v19",
        "schema_version": "v003",
        "source_dataset": str(args.source_dir),
        "assistant_char_budget": ASSISTANT_CHAR_BUDGET,
        "critical_upsample_factor": CRITICAL_UPSAMPLE_FACTOR,
        "totals_in": {"train": train_in, "val": val_in, "test": test_in},
        "totals_out": {"train": train_out, "val": val_out, "test": test_out},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
