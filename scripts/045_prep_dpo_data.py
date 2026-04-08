#!/usr/bin/env python3
"""Build DPO preference pairs from teacher comparisons and coaching variants.

This second-stage dataset turns existing teacher disagreement artifacts into
pairwise preferences. For scoring pairs, the teacher response closer to the
 consensus score becomes the chosen answer. For coaching pairs, richer and more
specific feedback variants are ranked above thinner alternatives using an
explicit heuristic so the DPO stage can prefer higher-signal coaching behavior.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCORING_RESPONSE_KEYS = [
    "task_id",
    "frame_analyses",
    "completion_time_seconds",
    "phase_timings",
    "knot_assessments",
    "suture_placement",
    "drain_assessment",
    "estimated_penalties",
    "estimated_fls_score",
    "confidence_score",
    "technique_summary",
    "improvement_suggestions",
    "strengths",
    "penalties",
    "score_components",
    "phases_detected",
    "reasoning",
]

FEEDBACK_STRIP_KEYS = {
    "feedback_id",
    "score_id",
    "generated_at",
    "generator",
}

TASK_LABELS = {
    "task1": "FLS Task 1 peg transfer",
    "task2": "FLS Task 2 pattern cut",
    "task3": "FLS Task 3 ligating loop",
    "task4": "FLS Task 4 extracorporeal suture",
    "task5": "FLS Task 5 intracorporeal suture",
}


def _canonical_task_id(task_id: str | None) -> str:
    raw = str(task_id or "task5").strip().lower()
    aliases = {
        "task1_peg_transfer": "task1",
        "task2_pattern_cut": "task2",
        "task3_endoloop": "task3",
        "task3_ligating_loop": "task3",
        "task4_extracorporeal_knot": "task4",
        "task4_extracorporeal_suture": "task4",
        "task5_intracorporeal_suturing": "task5",
        "task5_intracorporeal_suture": "task5",
    }
    if raw in aliases:
        return aliases[raw]
    if raw.isdigit():
        return f"task{raw}"
    if raw.startswith("task"):
        return raw.split("_", 1)[0]
    return "task5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DPO preference data from repo memory")
    parser.add_argument("--base-dir", default=".", help="Repository root")
    parser.add_argument(
        "--comparisons-dir",
        default="memory/comparisons",
        help="Directory containing teacher comparison and consensus files",
    )
    parser.add_argument(
        "--feedback-dir",
        default="memory/feedback",
        help="Directory containing coaching/feedback variants",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training/dpo_v1",
        help="Destination directory for DPO train/val JSONL files",
    )
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_score_path(base_dir: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    joined = base_dir / candidate
    if joined.exists():
        return joined

    scores_root = base_dir / "memory" / "scores"
    fallback_name = candidate.name
    matches = sorted(scores_root.rglob(fallback_name))
    if matches:
        return matches[-1]
    return None


def _normalize_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key in SCORING_RESPONSE_KEYS:
        if key in payload:
            normalized[key] = payload[key]
    return normalized


def _normalize_feedback_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in FEEDBACK_STRIP_KEYS}


def _render_scoring_prompt(system_prompt: str, video_id: str, task_id: str | None) -> str:
    canonical = _canonical_task_id(task_id)
    label = TASK_LABELS.get(canonical, TASK_LABELS["task5"])
    return (
        f"{system_prompt}\n\n"
        "User:\n"
        f"[FRAMES_PLACEHOLDER] Score this {label} video.\nTask ID: {canonical}\nVideo ID: {video_id}"
    )


def _render_coaching_prompt(
    coach_prompt: str,
    video_id: str,
    consensus_payload: dict[str, Any] | None,
) -> str:
    task_id = _canonical_task_id((consensus_payload or {}).get("task_id"))
    parts = [f"Video ID: {video_id}"]
    parts.append(f"Task ID: {task_id}")
    if consensus_payload:
        parts.extend([
            "Consensus Score JSON:",
            json.dumps(_normalize_score_payload(consensus_payload), indent=2),
        ])
    parts.append("Trainee History: unavailable")
    parts.append("Generate progression-aware coaching feedback.")
    user_content = "\n\n".join(parts)
    return f"{coach_prompt}\n\nUser:\n{user_content}"


def _comparison_distance(candidate: dict[str, Any], consensus: dict[str, Any]) -> float:
    candidate_score = float(candidate.get("estimated_fls_score") or 0.0)
    consensus_score = float(consensus.get("estimated_fls_score") or 0.0)
    candidate_time = float(candidate.get("completion_time_seconds") or 0.0)
    consensus_time = float(consensus.get("completion_time_seconds") or 0.0)
    candidate_penalties = float(candidate.get("estimated_penalties") or 0.0)
    consensus_penalties = float(consensus.get("estimated_penalties") or 0.0)

    distance = abs(candidate_score - consensus_score) / 20.0
    distance += abs(candidate_time - consensus_time) / 15.0
    distance += abs(candidate_penalties - consensus_penalties) / 5.0

    candidate_drain = candidate.get("drain_assessment") or {}
    consensus_drain = consensus.get("drain_assessment") or {}
    if bool(candidate_drain.get("gap_visible")) != bool(consensus_drain.get("gap_visible")):
        distance += 1.0
    if bool(candidate_drain.get("drain_avulsed")) != bool(consensus_drain.get("drain_avulsed")):
        distance += 2.0

    candidate_phases = {
        item.get("phase")
        for item in candidate.get("phase_timings", [])
        if isinstance(item, dict) and item.get("phase")
    }
    consensus_phases = {
        item.get("phase")
        for item in consensus.get("phase_timings", [])
        if isinstance(item, dict) and item.get("phase")
    }
    distance += 0.25 * len(candidate_phases.symmetric_difference(consensus_phases))
    return distance


def _feedback_quality_score(payload: dict[str, Any]) -> float:
    score = 0.0
    score += 4.0 if payload.get("overall_assessment") else 0.0
    score += 3.0 if payload.get("practice_plan") else 0.0
    score += 2.0 if payload.get("progress_context") else 0.0
    score += len(payload.get("technique_coaching", [])) * 2.0
    score += len(payload.get("strengths_to_reinforce", [])) * 1.5
    score += len(payload.get("rubric_insights", [])) * 1.0
    score += len(payload.get("top_priorities", [])) * 0.5
    score += len(payload.get("phase_coaching", [])) * 0.25
    score += float(payload.get("confidence") or 0.0)
    score += 1.0 if payload.get("coach_version") else 0.0
    score += 1.0 if payload.get("next_session_plan") else 0.0
    return score


def _latest_consensus_map(comparisons_dir: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted(comparisons_dir.glob("*_consensus_*.json")):
        payload = _load_json(path)
        video_id = str((payload.get("_meta") or {}).get("video_id") or payload.get("video_id") or "")
        if not video_id:
            continue
        timestamp = str((payload.get("_meta") or {}).get("timestamp") or payload.get("created_at") or path.name)
        previous = latest.get(video_id)
        if previous is None or timestamp >= previous[0]:
            latest[video_id] = (timestamp, payload)
    return {video_id: payload for video_id, (_, payload) in latest.items()}


def _build_scoring_pairs(
    base_dir: Path,
    comparisons_dir: Path,
    system_prompt: str,
) -> tuple[list[dict[str, Any]], list[float]]:
    consensus_map = _latest_consensus_map(comparisons_dir)
    pairs: list[dict[str, Any]] = []
    score_deltas: list[float] = []

    for path in sorted(comparisons_dir.glob("*teacher_comparison*.json")):
        payload = _load_json(path)
        teachers = payload.get("teachers") or {}
        if not isinstance(teachers, dict):
            continue
        claude_meta = teachers.get("claude") or {}
        gpt_meta = teachers.get("gpt-4o") or {}
        claude_path = _resolve_score_path(base_dir, str(claude_meta.get("score_file") or ""))
        gpt_path = _resolve_score_path(base_dir, str(gpt_meta.get("score_file") or ""))
        if claude_path is None or gpt_path is None:
            continue

        claude_payload = _load_json(claude_path)
        gpt_payload = _load_json(gpt_path)
        video_id = str(payload.get("video_id") or claude_payload.get("video_id") or "")
        consensus_payload = consensus_map.get(video_id, {}).get("consensus_score")
        if not video_id or not consensus_payload:
            continue

        claude_distance = _comparison_distance(claude_payload, consensus_payload)
        gpt_distance = _comparison_distance(gpt_payload, consensus_payload)
        if claude_distance == gpt_distance:
            continue

        chosen_payload = claude_payload if claude_distance < gpt_distance else gpt_payload
        rejected_payload = gpt_payload if claude_distance < gpt_distance else claude_payload

        pairs.append(
            {
                "prompt": _render_scoring_prompt(system_prompt, video_id, consensus_payload.get("task_id")),
                "chosen": json.dumps(_normalize_score_payload(chosen_payload), ensure_ascii=True),
                "rejected": json.dumps(_normalize_score_payload(rejected_payload), ensure_ascii=True),
                "metadata": {
                    "video_id": video_id,
                    "pair_type": "scoring",
                    "task": _canonical_task_id(consensus_payload.get("task_id")),
                },
            }
        )
        score_deltas.append(
            abs(
                float(chosen_payload.get("estimated_fls_score") or 0.0)
                - float(rejected_payload.get("estimated_fls_score") or 0.0)
            )
        )

    return pairs, score_deltas


def _build_feedback_pairs(
    feedback_dir: Path,
    comparisons_dir: Path,
    coach_prompt: str,
) -> list[dict[str, Any]]:
    consensus_map = _latest_consensus_map(comparisons_dir)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(feedback_dir.glob("*.json")):
        payload = _load_json(path)
        video_id = str(payload.get("video_id") or "")
        if not video_id:
            continue
        grouped[video_id].append(payload)

    pairs: list[dict[str, Any]] = []
    for video_id, variants in grouped.items():
        if len(variants) < 2:
            continue
        ranked = sorted(variants, key=_feedback_quality_score)
        rejected_payload = ranked[0]
        chosen_payload = ranked[-1]
        if _feedback_quality_score(chosen_payload) == _feedback_quality_score(rejected_payload):
            continue
        consensus_payload = consensus_map.get(video_id, {}).get("consensus_score")
        pairs.append(
            {
                "prompt": _render_coaching_prompt(coach_prompt, video_id, consensus_payload),
                "chosen": json.dumps(_normalize_feedback_payload(chosen_payload), ensure_ascii=True),
                "rejected": json.dumps(_normalize_feedback_payload(rejected_payload), ensure_ascii=True),
                "metadata": {
                    "video_id": video_id,
                    "pair_type": "coaching",
                    "task": _canonical_task_id((consensus_payload or {}).get("task_id") or (chosen_payload.get("_meta") or {}).get("task_id")),
                },
            }
        )
    return pairs


def _split_pairs(
    pairs: list[dict[str, Any]],
    val_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    val_size = max(1, int(round(len(shuffled) * val_split)))
    if val_size >= len(shuffled):
        val_size = len(shuffled) - 1
    return shuffled[val_size:], shuffled[:val_size]


def _write_jsonl(pairs: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w") as handle:
        for pair in pairs:
            output = {
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }
            handle.write(json.dumps(output, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    comparisons_dir = base_dir / args.comparisons_dir
    feedback_dir = base_dir / args.feedback_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    universal_prompt = base_dir / "prompts" / "v002_universal_scoring_system.md"
    system_prompt = universal_prompt.read_text() if universal_prompt.exists() else (base_dir / "prompts" / "v001_task5_system.md").read_text()
    coach_prompt = (base_dir / "prompts" / "v001_task5_coach.md").read_text()

    scoring_pairs, score_deltas = _build_scoring_pairs(base_dir, comparisons_dir, system_prompt)
    feedback_pairs = _build_feedback_pairs(feedback_dir, comparisons_dir, coach_prompt)
    all_pairs = scoring_pairs + feedback_pairs
    train_pairs, val_pairs = _split_pairs(all_pairs, val_split=args.val_split, seed=args.seed)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    _write_jsonl(train_pairs, train_path)
    _write_jsonl(val_pairs, val_path)

    task_distribution = Counter(
        pair.get("metadata", {}).get("task", "unknown") for pair in all_pairs
    )
    pair_type_distribution = Counter(
        pair.get("metadata", {}).get("pair_type", "unknown") for pair in all_pairs
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "val_split": args.val_split,
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "avg_score_delta": round(sum(score_deltas) / len(score_deltas), 4) if score_deltas else 0.0,
        "task_distribution": dict(task_distribution),
        "pair_type_distribution": dict(pair_type_distribution),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote DPO data to {output_dir}")
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"Avg score delta: {summary['avg_score_delta']:.2f}")
    print(f"Task distribution: {summary['task_distribution']}")
    print(f"Pair types: {summary['pair_type_distribution']}")


if __name__ == "__main__":
    main()