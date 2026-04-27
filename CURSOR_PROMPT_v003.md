# Cursor Agent Prompt — FLS-Training v003 Scoring & Reporting Refactor

Paste everything below the `---` into Cursor (Composer / Agent mode) on a new branch `v003-scoring-report`. Local repo is already synced to `587e44e` on `main`.

The body of this prompt is the full ChatGPT specification verbatim, with a short factual prefix listing what's already verified in the repo so the agent doesn't waste tokens re-discovering it.

---

## Repo facts already verified (do not re-discover; act on these)

- Branch: `main` @ `587e44e` (v5 demo with markdown reports, resident history, admin viewer, email).
- Rubric YAMLs in `rubrics/` already have correct per-task `max_score`/`max_time_seconds`: task1=300, task2=300, task3=180, task4=420, task5=600, task6=315. Use them as the source of truth.
- `demo/fls_demo_v5.py` `TASK_MAX_SCORES` has WRONG values for **task3 (300, should be 180)** and **task4 (300, should be 420)**. Fix on the way through.
- `src/scoring/schema.py` `ScoreComponents` only has `time_score / penalty_deductions / total_fls_score`. `PenaltyItem` only has `type/count/description`. There is no `CriticalError` model. `ScoringResult` is missing `task_name`, `max_score`, `max_time_seconds`, `critical_errors`, `cannot_determine`, `confidence_rationale`, `task_specific_assessments`.
- `src/feedback/feedback_generator.py` hard-codes `max_time_seconds: 600` and computes `pass_likely` from `score>0 and time<600`, and renders `/ 600`. Task-5-centric.
- `src/feedback/generator.py` hard-codes the 475 proficiency cutoff with a Task-5 comment.
- `src/scoring/fls_formula.py` is `calculate_task5_score` only.
- v002 prompts exist (`prompts/v002_*.md`); no v003 prompts yet.
- No `src/reporting/` package and no `src/rubrics/loader.py` — both must be created.
- Demo v5 grades performance with `excellent ≥75%` / `proficient ≥55%` / etc., and renders GRS z-scores as primary language. Both behaviors violate v003 gating rules and must be replaced.

Working agreement:
- Branch: `v003-scoring-report`. Small, reviewable commits per implementation step.
- Do not delete `src/feedback/feedback_generator.py` or `src/feedback/generator.py` in the same commit you add `src/reporting/report_v3.py`. Keep both running until `report_v3` is wired in, then move the old generators under `archive/deprecated/feedback_v1/` with a deprecation note.
- Do not retrain the LoRA. Stop after the validator step until human review.
- Run `pytest tests/test_report_v3_*.py tests/test_frontend_task_selection.py` and report failures before finishing.
- When all tasks are implemented and tests pass, push the branch and open a PR titled `v003: rubric-faithful scoring + reporting + Task 6 custom-task labeling`.

---

## You are updating the FLS-Training repo reporting/scoring pipeline to v003.

Goal:
Make the reporting system task-specific, rubric-faithful, clinically cautious, and coaching-rich. The system must continue identifying strengths, weaknesses, and specific improvement drills, but it must not overstate proficiency or present AI-derived z-scores as official FLS scoring.

Repository:
https://github.com/ry86pkqf74-rgb/FLS-Training

Primary problems to fix:
1. Reports are Task-5-centric and sometimes hard-code 600 as the denominator.
2. Reports may say "proficient" despite major technical errors.
3. Reports may claim "no significant penalties" while also describing major penalties.
4. Z-scores are presented as if clinically validated FLS metrics.
5. Priority recommendations are too speed-focused and not sufficiently safety/correctness-first.
6. v002 prompt output fields do not fully align with the Pydantic scoring schema.
7. Task 6 is a custom/non-official extension and must not be labeled as official FLS.

Create a v003 reporting/scoring update with the following tasks.

---

### Task 1 — Create a task-aware rubric loader

Files to inspect

```
rubrics/*.yaml
src/scoring/frontier_scorer.py
src/scoring/schema.py
src/feedback/generator.py
src/feedback/feedback_generator.py
```

Required changes

Create or update:

`src/rubrics/loader.py`

Implement:

```
load_rubric(task_id: str) -> dict
canonical_task_id(task_id: str | int) -> str
get_task_max_score(task_id: str) -> float
get_task_max_time(task_id: str) -> float
get_task_name(task_id: str) -> str
get_task_phase_benchmarks(task_id: str) -> dict
get_task_penalty_definitions(task_id: str) -> dict
is_official_fls_task(task_id: str) -> bool
```

Acceptance criteria

- Task 1 denominator = 300
- Task 2 denominator = 300
- Task 3 denominator = 180
- Task 4 denominator = 420
- Task 5 denominator = 600
- Task 6 denominator = 315
- Task 6 is labeled as:
  Custom FLS-adjacent training task, not one of the five official FLS manual skills tasks.
- Official FLS reports should only describe Tasks 1–5 as official FLS tasks. SAGES lists the five FLS manual tasks as peg transfer, precision cutting, ligating loop, extracorporeal suturing, and intracorporeal suturing.

---

### Task 2 — Fix scoring schema mismatch

File

`src/scoring/schema.py`

The current prompt expects rich scoring fields such as:

```json
{
  "score_components": {
    "max_score": 600,
    "time_used": 147.0,
    "total_penalties": 2.5,
    "total_fls_score": 450.5,
    "formula_applied": "600 - 147.0 - 2.5 = 450.5"
  }
}
```

The v002 prompt explicitly requires evidence-based penalties, frame analyses, confidence rationale, cannot-determine fields, and transparent formula math.

Update the schema so it can preserve all of that.

Add or update models

```python
class ScoreComponents(BaseModel):
    max_score: float
    time_used: float
    total_penalties: float
    total_fls_score: float
    formula_applied: str

    # Backward-compatible aliases if needed
    time_score: Optional[float] = None
    penalty_deductions: Optional[float] = None


class PenaltyItem(BaseModel):
    type: str
    description: str = ""
    points_deducted: float = 0.0
    count: int = 1
    severity: str = "minor"  # minor, moderate, major, critical, auto_fail
    frame_evidence: list[int] = Field(default_factory=list)
    confidence: float = 0.5
    rubric_reference: str = ""


class CriticalError(BaseModel):
    type: str
    present: bool
    reason: str = ""
    frame_evidence: list[int] = Field(default_factory=list)
    forces_zero_score: bool = False
    blocks_proficiency_claim: bool = True


class ScoringResult(BaseModel):
    ...
    task_name: str = ""
    max_time_seconds: float = 0
    max_score: float = 0
    score_components: Optional[ScoreComponents] = None
    penalties: list[PenaltyItem] = []
    critical_errors: list[CriticalError] = []
    cannot_determine: list[str] = []
    confidence_rationale: str = ""
    task_specific_assessments: dict = Field(default_factory=dict)
```

Acceptance criteria

- No scored output loses `max_score`, `time_used`, `total_penalties`, or `formula_applied`.
- `estimated_fls_score` must always equal `score_components.total_fls_score`.
- `estimated_penalties` must always equal `score_components.total_penalties`.
- Old score records still load through compatibility logic.

---

### Task 3 — Recompute score centrally, never trust model math blindly

File

`src/scoring/frontier_scorer.py`

Add a function:

```python
def recompute_score_from_components(payload: dict, task_id: str) -> dict:
    rubric = load_rubric(task_id)
    max_score = float(rubric["max_score"])
    completion_time = float(payload.get("completion_time_seconds", 0))
    penalties = payload.get("penalties", [])

    total_penalties = sum(
        float(p.get("points_deducted", p.get("value", 0)))
        for p in penalties
        if isinstance(p, dict)
    )

    if any(
        p.get("severity") == "auto_fail" or p.get("forces_zero_score") is True
        for p in penalties
        if isinstance(p, dict)
    ):
        total_score = 0.0
        formula = f"automatic zero due to auto-fail penalty"
    else:
        total_score = max(0.0, max_score - completion_time - total_penalties)
        formula = f"{max_score:g} - {completion_time:g} - {total_penalties:g} = {total_score:g}"

    payload["max_score"] = max_score
    payload["estimated_penalties"] = total_penalties
    payload["estimated_fls_score"] = total_score
    payload["score_components"] = {
        "max_score": max_score,
        "time_used": completion_time,
        "total_penalties": total_penalties,
        "total_fls_score": total_score,
        "formula_applied": formula,
    }
    return payload
```

Acceptance criteria

- Consensus score is never the average of teacher final scores.
- Consensus score is recomputed from max score, time, and penalties.
- If the model reports a formula that does not match recomputed math, overwrite it and log a warning.
- For Task 5 sample:
  `600 − 142 − 61 = 397`
  must be represented exactly.

---

### Task 4 — Replace hard-coded report generators with v003 report module

Existing files

```
src/feedback/generator.py
src/feedback/feedback_generator.py
```

New file

`src/reporting/report_v3.py`

The current feedback schema already frames feedback as a primary coaching target with phase coaching, priorities, progression insights, strengths, fatigue assessment, benchmarks, and next-session plan. Preserve that intent, but make it task-aware and safety-first.

Required public API

```python
def generate_report_v3(
    score: ScoringResult,
    previous_scores: list[ScoringResult] | None = None,
    profile: TraineeProfile | None = None,
    include_experimental_metrics: bool = True,
) -> dict:
    ...
```

Required report sections

```json
{
  "report_version": "v003",
  "disclaimer": "AI-assisted training feedback; not an official FLS certification score.",
  "task": {
    "task_id": "task5",
    "task_name": "Intracorporeal Suture with Knot Tying",
    "official_fls_task": true,
    "max_score": 600,
    "max_time_seconds": 600
  },
  "score_summary": {
    "training_score": 397.0,
    "max_score": 600,
    "completion_time_seconds": 142.0,
    "total_penalties": 61.0,
    "formula_applied": "600 - 142 - 61 = 397",
    "score_interpretation": "Below local Task 5 proficiency target; technical penalties dominate."
  },
  "readiness_status": {
    "label": "needs_focused_remediation",
    "proficiency_claim_allowed": false,
    "rationale": [
      "Score below configured Task 5 target.",
      "Major technical error: knot insecurity.",
      "Major technical error: incomplete slit closure."
    ]
  },
  "critical_findings": [],
  "strengths": [],
  "improvement_priorities": [],
  "phase_breakdown": [],
  "task_specific_feedback": {},
  "next_practice_plan": {},
  "experimental_metrics": {},
  "cannot_determine": [],
  "confidence": {}
}
```

Prohibited report behavior

Reports must not say:

```
proficient
pass likely
above average
excellent
no significant penalties
clinically competent
ready for formal FLS
```

unless all gating rules pass.

---

### Task 5 — Implement proficiency/readiness gating

New file

`src/reporting/readiness.py`

Implement:

```python
def determine_readiness(score: ScoringResult, rubric: dict) -> dict:
    ...
```

Labels

Use these exact labels:

```
unscorable
automatic_fail
needs_human_review
needs_focused_remediation
borderline
on_track_for_training
meets_local_training_target
```

Gating logic

Use this order:

```
if video_classification in ["instructional", "unusable"]:
    return "unscorable"

if any(error.forces_zero_score for error in critical_errors):
    return "automatic_fail"

if confidence_score < 0.60:
    return "needs_human_review"

if any(error.blocks_proficiency_claim for error in critical_errors):
    return "needs_focused_remediation"

if estimated_fls_score < local_target:
    return "needs_focused_remediation"

if estimated_fls_score within 5% of local_target:
    return "borderline"

return "meets_local_training_target"
```

Critical errors that block proficiency claims

Task 1
- incomplete task
- lost object outside field/unrecoverable
- wrong transfer sequence not corrected

Task 2
- gauze detachment
- incomplete cut
- large off-line deviation
- cannot assess final cut

Task 3
- loop not cinched
- loop grossly off mark
- appendage transection
- incomplete ligation

Task 4
- drain avulsion
- knot failure
- visible gap after knot
- gross mark miss
- knot pusher failure preventing secure knot

Task 5
- drain avulsion
- knot failure
- visible gap after final knot
- failure to complete required throws
- failure to switch hands when required
- gross mark miss

Task 6
- needle leaves field of view
- block dislodged
- failure to complete all required ring pairs if rubric requires completion for scoring

---

### Task 6 — Move z-scores to experimental appendix

Problem

The sample report says:

> The global z-score places you above average.

Do not allow that as primary assessment language unless the cohort is explicitly validated.

Required change

Add:

```json
"experimental_metrics": {
  "displayed": true,
  "title": "AI-derived process metrics",
  "disclaimer": "These are internal model-derived coaching features and are not part of official FLS scoring.",
  "reference_cohort": {
    "name": "",
    "n": null,
    "date_range": "",
    "validated": false
  },
  "metrics": [
    {
      "name": "bimanual_dexterity",
      "value": 0.71932,
      "scale": "z_score",
      "interpretation": "higher than this model's internal reference mean",
      "not_official_score": true
    }
  ]
}
```

Display rule

If `reference_cohort.validated` is false, do not use:

```
above average
excellent
poor
proficient
clinically competent
```

Use:

```
higher relative model-derived signal
lower relative model-derived signal
consistent with observed smoothness
may support coaching focus
```

---

### Task 7 — Rewrite task-specific coaching templates

Create:

`src/reporting/task_templates.py`

Each task gets:

```python
TASK_REPORT_TEMPLATES = {
    "task1": {...},
    "task2": {...},
    ...
}
```

Each template must define:

```json
{
  "strength_signals": [],
  "weakness_signals": [],
  "critical_errors": [],
  "recommended_drills": [],
  "phase_focus_rules": [],
  "plain_language_summary_rules": []
}
```

#### Task 1 — Peg Transfer

Strengths to identify
- Maintains object control through mid-air transfer
- Uses both hands appropriately
- Smooth transfer rhythm
- Minimal collisions with pegs/board
- Completes outbound and return transfers
- Recovers minor bobbles without losing object

Weaknesses to identify
- Object dropped or unrecoverable
- Board-assisted transfer
- Wrong peg/wrong side placement
- Excessive searching/regrasping
- Dominant hand over-reliance
- Poor handoff timing
- Inefficient return phase

Priority drills

```
1. Mid-air handoff drill:
   Transfer one object back and forth 20 times without setting it down.

2. Peg-target accuracy drill:
   Place each object on assigned peg with no board contact.

3. Six-object rhythm drill:
   Complete all six outbound transfers before return; focus on consistent cadence.

4. Recovery-control drill:
   Practice controlled regrasp after intentional minor bobble without object leaving field.
```

Report focus

Task 1 coaching should emphasize:

```
object control
bimanual coordination
handoff efficiency
unrecoverable drops
sequence completion
```

#### Task 2 — Pattern Cutting

Strengths to identify
- Maintains gauze tension
- Cuts with small controlled scissor bites
- Follows marked circle
- Uses grasper to rotate/reposition gauze safely
- Completes cut without detachment
- Avoids large line deviations

Weaknesses to identify
- Deviation outside target line
- Large irregular cuts
- Loss of tension
- Gauze tearing
- Gauze detached from clamp
- Incomplete cut
- Poor scissor angle
- Excessive repositioning

Priority drills

```
1. Cardinal-point accuracy drill:
   Cut short arcs at 12, 3, 6, and 9 o'clock while measuring line deviation.

2. Micro-bite scissor drill:
   Use 2–3 mm scissor bites along a printed circle.

3. Tension-control drill:
   Maintain consistent grasper tension without tearing or tenting gauze.

4. Half-circle segmentation drill:
   Cut clockwise half, reset, then counterclockwise half.
```

Report focus

Task 2 coaching should include:

```
line deviation map
which arc was worst
tension control
scissor bite size
completion status
```

If deviation cannot be measured from the video, report:

> Exact line deviation could not be determined from this camera angle.

The v002 scoring prompt already requires cardinal-point deviation reporting for Task 2 when visible.

#### Task 3 — Endoloop / Ligating Loop

Strengths to identify
- Opens loop cleanly
- Approaches appendage without snagging
- Places loop at target mark
- Cinches securely
- Cuts tail without destabilizing loop
- Maintains instrument control

Weaknesses to identify
- Loop off mark
- Loose cinch
- Loop catches on appendage or instrument
- Excessive manipulation
- Tail cut destabilizes loop
- Appendage transection
- Incomplete loop placement

Priority drills

```
1. Loop-opening drill:
   Open and orient the loop five times before target approach.

2. Mark-alignment drill:
   Place loop at the line without cinching, reset, repeat 10 times.

3. Cinch-control drill:
   Cinch gradually while keeping loop perpendicular to appendage.

4. Tail-cut stability drill:
   Cut tail while maintaining loop position and tension.
```

Report focus

Task 3 coaching should include:

```
loop location
cinch security
appendage integrity
tail management
```

#### Task 4 — Extracorporeal Suture with Knot Pusher

Strengths to identify
- Needle loaded correctly
- Needle passes through both marks
- Appropriate bite depth
- Smooth extracorporeal knot formation
- Knot pusher advances knot without loosening
- Slit approximates after tightening
- Safe tail cutting

Weaknesses to identify
- Suture deviation from marks
- Poor needle angle
- Excessive drain traction
- Knot pusher misalignment
- Knot loosens during advancement
- Visible gap after knot
- Drain avulsion
- Knot failure

Priority drills

```
1. Mark-to-mark needle driving drill:
   Ten passes through both marks with immediate deviation measurement.

2. Extracorporeal knot build drill:
   Tie knot outside box, inspect structure, then deliver with pusher.

3. Knot-pusher alignment drill:
   Advance pusher in line with suture without levering against drain.

4. Progressive tension drill:
   Tighten gradually while watching slit approximation.
```

Report focus

Task 4 coaching should include:

```
needle placement
knot pusher control
slit closure
knot security under tension
```

#### Task 5 — Intracorporeal Suture with Knot Tying

Strengths to identify
- Needle orientation
- Accurate entry/exit through marks
- Proper first surgeon's knot/double throw
- Hand switching between throws
- Smooth wrapping
- Maintains tension without avulsion
- Cuts both tails appropriately

Weaknesses to identify
- Mark deviation
- Gap visible after final knot
- Knot slips or comes apart
- First throw not double throw
- Missed hand switch
- Excessive tail length
- Drain trauma
- Cutting before knot secure

Priority drills

```
1. Needle-angle drill:
   Load needle at correct angle and drive through both marks without dragging.

2. First-throw surgeon's-knot drill:
   Ten double-throw first knots with deliberate tension control.

3. Alternating-hand square-knot drill:
   First throw, switch hands, second throw, switch hands, third throw.

4. Closure-before-cut drill:
   Pause before cutting tails and verify no visible slit gap.

5. Knot-security check drill:
   Apply gentle opposing tension after knot completion; knot must not slip.
```

Report focus

Task 5 coaching should prioritize:

```
closure quality
knot security
required throw sequence
hand switching
suture placement
```

Do not let speed praise outweigh knot failure or incomplete closure.

#### Task 6 — Rings of Rings Needle Manipulation

Strengths to identify
- Needle remains in view
- Smooth approach to ring pair
- Alternates central/peripheral passes as required
- Completes ring pairs sequentially
- Avoids block dislodgement
- Recovers needle orientation efficiently

Weaknesses to identify
- Missed inner ring
- Missed outer ring
- Incorrect pass type
- Needle exits field of view
- Block dislodged
- Excessive reorientation
- Incomplete ring sequence

Priority drills

```
1. Two-ring repeat drill:
   Repeat one ring pair until central and peripheral passes are clean.

2. Needle-orientation reset drill:
   After each pass, reset needle angle before moving to next pair.

3. Alternation callout drill:
   Verbally call central/peripheral before each pass.

4. No-exit field discipline drill:
   Keep needle tip visible continuously for the entire sequence.
```

Report focus

Task 6 coaching should include:

```
rings completed
rings missed
needle visibility
block stability
pass-type alternation
```

---

### Task 8 — Update markdown report renderer

Create:

`src/reporting/render_markdown_v3.py`

Required markdown structure

```markdown
# FLS Training Feedback Report

## 1. Task and Score Summary
- Task:
- Score:
- Formula:
- Completion time:
- Penalty total:
- Confidence:
- Training-readiness status:

## 2. Important Interpretation
This is AI-assisted training feedback, not an official FLS certification result.

## 3. Critical Findings
List critical errors first.

## 4. Strengths
Each strength must be evidence-linked.

## 5. Priority Improvements
Rank 1–3. Each priority must include:
- Observation
- Why it matters
- Practice target
- Recommended drill
- How to know it improved

## 6. Phase Breakdown
Table with phase, duration, benchmark, interpretation.

## 7. Task-Specific Coaching
Use task-specific template.

## 8. Next Practice Plan
Concrete next session plan.

## 9. Experimental AI Metrics
Only shown if enabled. Must include disclaimer.

## 10. Cannot Determine / Review Flags
List ambiguity and human-review triggers.
```

Required behavior

For every strength:

```
Bad: "Excellent tissue handling."
Good: "No drain avulsion was observed, and the instruments did not visibly tear the Penrose drain."
```

For every weakness:

```
Bad: "Improve knot tying."
Good: "The final knot appeared insecure after completion; practice alternating-hand square knots with deliberate tension equalization."
```

---

### Task 9 — Fix frontend upload/report behavior

Required frontend changes

- `task_id` must be required on video upload.
- The UI must show the task denominator from the rubric, not from a hard-coded 600.
- The UI must show:
  Training score, not official certification score
- The UI must render:
  Score = max score − completion time − penalties
- Z-scores must be collapsible under:
  Experimental AI-derived metrics
- If critical errors exist, show a warning banner:
  Major technical issue detected. Interpret score with caution; focus on correctness before speed.
- If confidence is low:
  Needs human review
- For Task 6:
  Custom training task, not official FLS manual skills task

Acceptance criteria

- Uploading a video without task selection is impossible.
- Task 1 never displays /600.
- Task 5 displays /600.
- Task 6 displays /315 and non-official label.
- Reports with knot failure cannot show "proficient."
- Reports with instructional/unusable video cannot show a score.

(Implementation note: branch this from `demo/fls_demo_v5.py` into `demo/fls_demo_v6.py`, read denominators from the rubric loader, and replace the `pct >= 75 → excellent / ≥55 → proficient` ladder with the readiness label produced by `determine_readiness`.)

---

### Task 10 — Update prompts to v003

Create:

```
prompts/v003_universal_scoring_system.md
prompts/v003_consensus_system.md
prompts/v003_report_generation_system.md
```

v003 scoring prompt requirements

Add explicit instructions:

```
1. Never call a trainee proficient solely because score is nonzero.
2. Never call performance proficient if critical errors exist.
3. Always output max_score and formula_applied.
4. Every penalty must include points_deducted, severity, confidence, and frame_evidence.
5. Separate official score facts from coaching observations.
6. Do not output z-scores unless supplied by a separate analytics module.
7. Do not invent mm deviations if not visible.
8. For Task 6, label as custom/non-official FLS extension.
```

v003 report generation prompt requirements

The report generation model receives:

```json
{
  "score_json": {},
  "rubric_json": {},
  "resident_level": "PGY3",
  "previous_attempts_summary": [],
  "experimental_metrics": {}
}
```

It outputs:

```json
{
  "report_version": "v003",
  "score_summary": {},
  "readiness_status": {},
  "critical_findings": [],
  "strengths": [],
  "improvement_priorities": [],
  "next_practice_plan": {},
  "experimental_metrics": {},
  "markdown": ""
}
```

Prohibited phrases unless gating allows

```
demonstrated proficiency
pass likely
clinically competent
above average
excellent tissue handling
no significant penalties
well on your way to becoming a competent surgeon
```

Replace with conditional, evidence-based language.

---

### Task 11 — Build validation tests

Create:

```
tests/test_report_v3_task_denominators.py
tests/test_report_v3_score_math.py
tests/test_report_v3_critical_error_gating.py
tests/test_report_v3_zscore_handling.py
tests/test_report_v3_task_templates.py
tests/test_report_v3_sample_task5.py
tests/test_frontend_task_selection.py
```

Specific tests

Test 1 — Denominator

```python
assert report(task1_score)["score_summary"]["max_score"] == 300
assert report(task5_score)["score_summary"]["max_score"] == 600
assert report(task6_score)["score_summary"]["max_score"] == 315
```

Test 2 — Formula

```python
score = make_task5_score(time=142, penalties=61)
report = generate_report_v3(score)
assert report["score_summary"]["training_score"] == 397
assert report["score_summary"]["formula_applied"] == "600 - 142 - 61 = 397"
```

Test 3 — No proficiency with knot failure

```python
score = make_task5_score(
    time=142,
    penalties=[{"type": "knot_failure", "points_deducted": 50, "severity": "major"}],
    estimated_fls_score=408,
)
report = generate_report_v3(score)
assert report["readiness_status"]["proficiency_claim_allowed"] is False
assert "proficient" not in report["markdown"].lower()
```

Test 4 — Z-score disclaimer

```python
report = generate_report_v3(score, include_experimental_metrics=True)
assert "not part of official FLS scoring" in report["markdown"]
assert "above average" not in report["markdown"].lower()
```

Test 5 — Critical errors outrank speed

```python
score = make_task5_score(
    time=120,
    critical_errors=["gap_visible", "knot_failure"],
)
report = generate_report_v3(score)
assert report["improvement_priorities"][0]["topic"] in ["knot_security", "slit_closure"]
```

Test 6 — Sample report contradiction prevention

Use the sample scenario:

```
Score 397 / 600
Time 142 s
Gap visible
Knot failure
Penalty burden 61
```

Assert:

```python
assert "no significant penalties" not in markdown.lower()
assert "demonstrated proficiency" not in markdown.lower()
assert "knot" in markdown.lower()
assert "slit" in markdown.lower() or "gap" in markdown.lower()
assert "600 - 142 - 61 = 397" in markdown
```

---

### Task 12 — Update LoRA / training-data pipeline

Do not train or continue training the LoRA on the old reports without relabeling. The current sample style includes contradictions and overconfident language, so it will teach the model the wrong reporting behavior.

New scripts

Create:

```
scripts/060_generate_report_v3_labels.py
scripts/061_validate_report_v3_labels.py
scripts/062_prepare_lora_report_v3_dataset.py
scripts/063_train_report_lora_v3.py
```

Label generation strategy

Use existing scored outputs where possible.

For each score record:

```
1. Load score JSON.
2. Load task rubric.
3. Normalize score fields into v003 schema.
4. Recompute score math.
5. Generate report_v3 JSON.
6. Validate with report validator.
7. Save as training target.
```

Only rescore the video if:

```
- task_id is missing
- task_specific_assessments are missing
- critical errors are unclear
- score math cannot be reconciled
- confidence < 0.60
```

Dataset format

Each LoRA example should be:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You generate rubric-faithful FLS training reports from structured scoring JSON. You do not invent observations."
    },
    {
      "role": "user",
      "content": {
        "score_json": {},
        "rubric_json": {},
        "resident_level": "PGY3",
        "previous_attempts_summary": [],
        "experimental_metrics": {}
      }
    },
    {
      "role": "assistant",
      "content": {
        "report_version": "v003",
        "score_summary": {},
        "readiness_status": {},
        "critical_findings": [],
        "strengths": [],
        "improvement_priorities": [],
        "next_practice_plan": {},
        "markdown": ""
      }
    }
  ]
}
```

LoRA training guidance

Train the LoRA to produce structured report JSON, not free-form prose only.

The markdown should be rendered from structured fields when possible. This reduces hallucination and prevents contradictions.

Validation before training

Reject any label if:

```
- formula does not match score
- wrong denominator
- says proficient despite critical error
- says no penalties despite penalties > 0
- z-score appears in primary assessment
- missing task-specific priority
- improvement advice is generic
- report contains unsupported observations
```

---

### Task 13 — Add a report validator

Create:

`src/reporting/validator.py`

Implement:

```python
def validate_report_v3(report: dict, score: ScoringResult, rubric: dict) -> list[str]:
    errors = []
    ...
    return errors
```

Required validation checks

```python
# denominator
assert report["score_summary"]["max_score"] == rubric["max_score"]

# score math
assert report["score_summary"]["training_score"] == score.estimated_fls_score

# no contradiction
if score.estimated_penalties > 0:
    assert "no significant penalties" not in markdown.lower()

# critical error gating
if critical_errors:
    assert "proficient" not in markdown.lower()
    assert report["readiness_status"]["proficiency_claim_allowed"] is False

# z-score placement
assert "global z-score" not in report["overall_assessment"].lower()

# task-specific specificity
assert len(report["improvement_priorities"]) >= 1
assert each priority has observation, practice_target, drill, success_metric
```

---

### Task 14 — Add before/after sample output

Create:

`docs/report_v3_examples/task5_pgy3_sample.md`

Use this corrected sample:

```markdown
# FLS Training Feedback Report

## Task and Score Summary

- Task: Intracorporeal Suture with Knot Tying
- Resident level: PGY3
- Training score: 397 / 600
- Completion time: 142 s
- Penalty burden: 61 points
- Formula: 600 − 142 − 61 = 397
- Training-readiness status: Needs focused remediation before formal FLS readiness
- Confidence: [insert model confidence]

This is an AI-assisted training report and not an official FLS certification result.

## Overall Assessment

You completed the task efficiently, and the time component suggests good basic task flow. However, the final score was substantially reduced by technical quality issues. The main concerns are incomplete slit closure and knot insecurity. These findings are higher priority than additional speed gains because the goal of the task is not only to finish quickly, but to create a secure closure.

## Critical Findings

1. Knot security concern
   - Observation: The final knot appeared to loosen or come apart after completion.
   - Impact: This is a major technical issue because the knot must maintain approximation under tension.
   - Coaching focus: Practice square knot construction and equal tension across all throws.

2. Incomplete slit closure
   - Observation: A visible gap remained after knot completion.
   - Impact: Persistent gap suggests insufficient approximation or tension control.
   - Coaching focus: Verify closure before cutting tails.

## Strengths

- Efficient completion time: 142 seconds is not the main limitation in this attempt.
- Task flow: The task was completed without timeout.
- Instrument control: No drain avulsion was reported.
- Bimanual workflow: Continue using deliberate hand transitions, but ensure the required alternating-hand knot sequence is fully maintained.

## Priority Improvements

### 1. Knot security

- Practice target: Secure square knot that does not slip under gentle tension.
- Drill: 10 intracorporeal knots using first double throw, then two alternating single throws.
- Success metric: Knot remains flat and secure after gentle tension; no loosening before tail cut.

### 2. Slit closure

- Practice target: No visible gap after the final throw.
- Drill: Mark-to-mark needle pass followed by closure verification before cutting.
- Success metric: Slit edges remain approximated after knot completion.

### 3. Penalty reduction before speed optimization

- Practice target: Reduce penalty burden from 61 points to under 20 while keeping time under 160 seconds.
- Drill: Slow deliberate repetitions with post-repetition video review.
- Success metric: Secure knot, no visible gap, and accurate mark placement on three consecutive attempts.

## Experimental AI-Derived Metrics

The following metrics are internal model-derived coaching signals and are not part of official FLS scoring.

- Bimanual dexterity: [value]
- Depth perception: [value]
- Efficiency of movement: [value]
- Tissue/material handling: [value]

Interpret these as coaching aids only, not validated pass/fail criteria.
```

---

### Task 15 — Update documentation

Create or update:

```
docs/REPORTING_V3.md
docs/SCORING_VS_COACHING.md
docs/LORA_REPORT_TRAINING_V3.md
```

`SCORING_VS_COACHING.md` must explain

```
Official/rubric-derived fields:
- task_id
- max_score
- completion time
- penalties
- final score
- critical errors
- confidence
- cannot determine

Coaching fields:
- strengths
- weaknesses
- recommended drills
- next practice plan
- longitudinal progress

Experimental fields:
- z-scores
- movement metrics
- cohort-relative analytics
```

Required policy statement

> The system must not represent AI-derived coaching analytics as official FLS certification scoring. Official FLS scoring uses normalized task scoring and certified procedures; this system provides AI-assisted training feedback unless explicitly validated and approved for certification use.

---

## Implementation priority order

Give the agent this exact order:

```
1. Add rubric loader.
2. Update scoring schema.
3. Add centralized score recomputation.
4. Add report_v3 generator.
5. Add readiness/proficiency gating.
6. Add task-specific templates.
7. Add markdown renderer.
8. Add report validator.
9. Add tests.
10. Update frontend task selection and denominator display.
11. Add v003 prompts.
12. Generate v003 labels from existing scores.
13. Validate labels.
14. Retrain LoRA only after labels pass validation.
15. Add documentation and examples.
```

---

## Final product behavior checklist

After the update, your system should be able to say:

```
Your strongest area was efficient task flow.
Your weakest area was knot security.
Your next practice focus should be alternating-hand square-knot construction and closure verification.
Your score was 397 / 600 because:
600 − 142 seconds − 61 penalty points = 397.
This is below the configured Task 5 training target, primarily due to technical penalties rather than time.
```

It should not say:

```
You demonstrated proficiency.
No significant penalties were applied.
Your z-score proves you are above average.
You are clinically competent.
```

That gives you the best of both worlds: rubric-faithful scoring plus specific, actionable coaching.

---

## ADDENDUM: Task 6 Rings of Rings Support

Important correction:
The system includes a sixth task called "Rings of Rings Needle Manipulation." It is not one of the five official FLS manual skills tasks, but it is part of our training platform and must remain fully supported in scoring, reporting, frontend display, LoRA labels, and longitudinal analytics.

Do not remove Task 6.
Do not treat Task 6 as an error.
Do not label Task 6 as official FLS certification scoring.

Task 6 should be labeled as:
"Custom FLS-adjacent training task: Rings of Rings Needle Manipulation."

Required Task 6 metadata:

```
- task_id: task6
- canonical_task_id: task6_rings_needle_manipulation
- task_name: Rings of Rings Needle Manipulation
- official_fls_task: false
- custom_training_task: true
- certification_eligible: false
- max_score: 315
- max_time_seconds: 315
- score_units: training points
- score_formula: 315 − completion_time_seconds − penalties, unless auto-fail condition is present
```

Also add this to the report rules:

```
For Task 6 reports:
- Use "training score," not "FLS certification score."
- Do not say "passed FLS" or "failed FLS."
- Use readiness language such as:
  - "meets custom Task 6 training target"
  - "needs focused remediation on ring traversal"
  - "automatic failure for this custom task due to needle leaving field of view"
- Display denominator as /315, never /600.
- Include a visible note:
  "Task 6 is a custom training task and is not part of official FLS certification scoring."
```

Task 6-specific report sections should include:

```
Task-specific findings:
- rings_completed: 0–8
- rings_missed: number of failed ring pairs
- needle_left_view: true/false
- block_dislodged: true/false
- central/peripheral alternation accuracy
- ring_traversal_notes: per-pair observations

Critical errors:
- needle exits field of view
- block dislodged
- failure to complete required ring sequence
- repeated missed inner/outer rings
- incorrect central/peripheral pass pattern

Strengths:
- maintained needle visibility
- completed sequential ring traversal
- clean pass through both inner and outer rings
- efficient reorientation between ring pairs
- stable pegboard/block handling
- correct alternation between central and peripheral passes

Improvement priorities:
1. Needle visibility discipline
2. Ring-pair targeting accuracy
3. Central/peripheral alternation consistency
4. Needle reorientation between passes
5. Smooth sequential traversal without block contact
```

I would also add one clarification task to the work order:

```
Resolve Task 6 scoring ambiguity.

Current Task 6 language may contain two interpretations:
A. score = 315 − completion_time − penalties, with 20 points/seconds per missed ring pair
B. must complete all 8 ring pairs within 315 seconds to receive any score

Decide and encode one rule explicitly in the rubric YAML and report generator.

Recommended rule:
- Auto-fail score = 0 if needle exits field of view or block is dislodged.
- Otherwise score = 315 − completion_time − 20 × failed_or_incomplete_ring_pairs.
- If the task is abandoned or fewer than a minimum required number of rings are attempted, mark as incomplete and require human review.
- Do not say official FLS pass/fail.
```

So the revised architecture is:

```
Official FLS manual skills tasks:
- Task 1 Peg Transfer
- Task 2 Pattern Cutting
- Task 3 Ligating Loop
- Task 4 Extracorporeal Suturing
- Task 5 Intracorporeal Suturing

Custom platform task:
- Task 6 Rings of Rings Needle Manipulation
```

The key is not to exclude Task 6, but to separate "official FLS" from "your platform's training curriculum."
