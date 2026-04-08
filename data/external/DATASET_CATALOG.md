# External Surgical Video Dataset Catalog

Curated list of public datasets that can feed the FLS-Training pipeline as
additional training data beyond the current YouTube harvest. Ordered by
estimated usefulness to this project.

Scope: datasets must contain (a) box-trainer / simulated laparoscopic video,
and (b) some form of skill label, performance metric, or expert demonstration
that the teacher-scoring pipeline can exploit. Real-OR videos and animal-lab
content are intentionally excluded.

## 1. LASANA — Laparoscopic Skill Analysis and Assessment (TU Dresden, 2026)

| Field | Value |
|---|---|
| URL | https://arxiv.org/abs/2602.09927 (paper) ; https://opara.zih.tu-dresden.de (host) |
| License | TBD — verify on download (likely CC BY / CC BY-NC) |
| Size | 1,270 stereo video clips, ~70 participants, 4 tasks |
| Task overlap with FLS | High — Peg Transfer, Pattern Cut, Suturing, Knot Tying (no Endoloop) |
| Annotations | GRS skill rating (3 raters), binary error labels, predefined splits |
| Ease of integration | **Medium**. Stereo format needs left-view crop; annotations are structured CSV-style; GRS labels remove the need for teacher scoring. |
| Estimated pair yield | **700–1,000 training examples** (single biggest source available). |
| Priority | **1 — highest leverage**. |
| Status | Metadata only; download blocked by Cowork egress proxy. See `LASANA_README.md`. |

## 2. SPD-FLS1 — Self-Practice Dataset, FLS Peg Transfer (Virginia Tech, 2025)

| Field | Value |
|---|---|
| URL | https://data.lib.vt.edu/articles/dataset/Self-practice_dataset_-_FLS_Peg_transfer_task_SPD-FLS1_/29421842 |
| License | Virginia Tech Figshare — check page (Figshare default is CC BY 4.0 unless flagged) |
| Size | 19 participants × up to 6 sessions × ~10 trials = hundreds of peg-transfer trials, each with video + wrist motion + tool-handle motion + eye trajectory |
| Format | `.avi` video plus `.txt` sensor logs, participant-session-trial file naming |
| Task overlap with FLS | **Task 1 only** (Peg Transfer), but deep longitudinal skill progression — same participants improving over sessions. |
| Annotations | Implicit skill progression (session number, trial time, proficiency criterion = 48s for 10 non-consecutive trials); no GRS rating. |
| Ease of integration | **Medium-Easy**. Video only works out-of-the-box; sensor streams are bonus multimodal signal we don't currently use. Longitudinal structure means we can auto-label skill via session index + time. |
| Estimated pair yield | **~150–250 peg-transfer pairs** (high-quality, covers novice→proficient within subject). |
| Priority | **2 — best Task 1 booster**. |
| Status | Not yet downloaded. |

## 3. PhysioNet — EEG and Eye-Gaze for FLS Tasks (NIBIB-RPCCC, 2024)

| Field | Value |
|---|---|
| URL | https://physionet.org/content/eeg-eye-gaze-for-fls-tasks/1.0.0/ |
| License | PhysioNet Credentialed Health Data License 1.5.0 (typical for PhysioNet; verify on page). Requires PhysioNet account. |
| Size | 25 participants × 3 FLS tasks × up to 5 attempts ≈ 250–375 trials |
| Format | CSVs for EEG (128ch, 500Hz) and eye-gaze (20 metrics, 50Hz). **Note: no video files.** The title suggests video; the asset listing on PhysioNet shows only sensor CSVs. |
| Task overlap with FLS | **High — 3 of 5 FLS tasks**: Peg Transfer, Pattern Cut, Intracorporeal Suturing. |
| Annotations | **Per-attempt performance scores from a human rater.** This is the gold nugget — scored FLS attempts. |
| Ease of integration | **Low as a video source** (no video), but the rating-per-trial metadata is extractable to use as cross-validation of our teacher scorers if we can ever correlate them to our clips. |
| Estimated pair yield | **~0 direct video training pairs**, but ~300 scored non-video trials that can be used for teacher-calibration or for scoring-function validation. |
| Priority | **4 — low priority for vision training, high priority for scorer calibration**. |
| Status | Registration required; not yet downloaded. |

## 4. JIGSAWS — JHU-ISI Gesture and Skill Assessment Working Set

| Field | Value |
|---|---|
| URL | https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/ |
| License | CIRL release license — free for research, redistribution restricted (check current terms). |
| Size | 3 tasks (Suturing 39, Needle Passing 26, Knot Tying 36) × 8 surgeons × ~5 reps = ~100 trials |
| Format | Synchronised stereo endoscopic video + da Vinci kinematics + manual gesture segment annotations |
| Task overlap with FLS | **Partial**. Suturing and Knot Tying are conceptually similar to FLS Tasks 4–5 but performed on the da Vinci robot, not on a manual laparoscopic box trainer. Motion is fundamentally different (wristed robotic instruments vs. rigid laparoscopic graspers). |
| Annotations | Global Rating Scores + expert/intermediate/novice class + gesture segments. |
| Ease of integration | **Low**. Robotic video distribution is visually distinct from box-trainer FLS video; including it risks domain confusion and hurting the student model. |
| Estimated pair yield | **~0–50 pairs** — only useful if we explicitly condition on "robotic" and keep a separate branch. **Not recommended for initial scaling runs.** |
| Priority | **5 — skip until we have a robotic-surgery use case**. |
| Status | Reference only. |

## Priority summary

| Rank | Dataset    | Est. pairs | Action |
|------|------------|------------|--------|
| 1    | LASANA     | 700–1,000  | Download ASAP (manual), integrate in Phase 3. |
| 2    | SPD-FLS1   | 150–250    | Download next; integrate in Phase 4. |
| 3    | YouTube expansion | see harvest_targets.csv | Phase 2, runs this week. |
| 4    | PhysioNet EEG-FLS | 0 video | Use for scorer calibration only. |
| 5    | JIGSAWS    | 0          | Skip. |

## References

- [LASANA paper (arXiv 2602.09927)](https://arxiv.org/abs/2602.09927)
- [SPD-FLS1 (VT Figshare)](https://data.lib.vt.edu/articles/dataset/Self-practice_dataset_-_FLS_Peg_transfer_task_SPD-FLS1_/29421842)
- [PhysioNet EEG+Eye-gaze for FLS](https://physionet.org/content/eeg-eye-gaze-for-fls-tasks/1.0.0/)
- [JIGSAWS (JHU CIRL)](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
