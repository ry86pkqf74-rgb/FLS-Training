# External Data Source Catalog — Verified April 8, 2026

## Current State
- **58 harvested videos** — ALL Task 5 (intracorporeal suturing/knot tying)
- **Zero coverage** of Tasks 1–4
- **78 total videos** after coworker's v3 expansion (still all Task 5)

---

## Tier 1: HIGH VALUE — Direct FLS task match with labels

### LASANA (Laparoscopic Skill Analysis and Assessment)
- **URL**: https://opara.zih.tu-dresden.de (search "LASANA" in TU Dresden collections)
- **Paper**: arXiv 2602.09927 (Feb 2026, Isabel Funke et al.)
- **Size**: 1,270 stereo video recordings
- **Tasks**: 4 FLS-like tasks — peg transfer, circle cutting, balloon resection, suture & knot
- **Per task**: ≥311 recordings
- **Labels**: Structured skill ratings (3 independent raters, aggregated) + binary task-specific error labels (e.g., "object dropped" for peg transfer, "balloon punctured" for balloon resection)
- **Participants**: 70 (from a laparoscopic training course — natural skill variation)
- **Data splits**: Pre-defined train/val/test per task
- **License**: Research use (check OPARA terms)
- **Integration effort**: MEDIUM — videos need frame extraction, labels need mapping to our JSON schema. Skill ratings are structured but may not map 1:1 to FLS point scores (they're ordinal ratings, not time-based). Error labels map well to our penalty system.
- **Estimated yield**: 800–1,000 training pairs (after excluding test split). **This is the single highest-ROI source.**
- **CRITICAL NOTE**: OPARA recently migrated platforms. Old data may be on the legacy site. May need to contact authors (TU Dresden / NCT) if download link isn't live yet — paper is from Feb 2026, dataset may still be in staged release.
- **Action**: Attempt download from OPARA. If not available, email authors (funke@nct-dresden.de likely, check paper).

### LASK (LAparoscopic Skill & Kinematics)
- **URL**: White Rose Research Online (University of Sheffield, 2025)
- **Paper**: MIUA 2025, Choudhry et al.
- **Size**: 114 trials (~3 hours), 324,101 frames
- **Tasks**: Peg transfer ONLY
- **Labels**: Low/medium/high skill (38/41/35 trials), 3,725 frames with bounding boxes, 7-DoF kinematics
- **Participants**: 114 trials from surgeons of varying experience
- **License**: Pending public release ("Once publicly released" — may not be available yet)
- **Integration effort**: LOW for video, HIGH for kinematics (different modality)
- **Estimated yield**: ~80 peg transfer training pairs
- **Action**: Check if released. Contact authors if not.

### PhysioNet NIBIB-RPCCC-FLS
- **URL**: https://physionet.org/content/eeg-eye-gaze-for-fls-tasks/1.0.0/
- **Size**: 315 recordings from 25 participants, 5 attempts each on 3 FLS tasks
- **Tasks**: 3 FLS tasks (specific tasks not detailed — likely peg transfer, pattern cut, suturing based on standard FLS curriculum)
- **Labels**: Performance scores assessed by expert rater after each attempt
- **Data**: EEG (.edf) + eye-gaze (.csv) + performance scores + demographics
- **License**: PhysioNet Credentialed Health Data License (free, requires account + data use agreement)
- **CRITICAL LIMITATION**: **No video files included** — only EEG and eye-gaze data. Performance scores exist but without corresponding video, this cannot be used for our VLM training pipeline.
- **Integration effort**: NOT VIABLE for video-based training. Scores could theoretically be used if we had the videos, but we don't.
- **Estimated yield**: 0 training pairs for VLM. Performance score distributions useful for validation/calibration only.
- **Action**: SKIP for training data. Optionally download score distributions to calibrate our rubric thresholds.

---

## Tier 2: MODERATE VALUE — Related but requires adaptation

### JIGSAWS (JHU-ISI Gesture and Skill Assessment)
- **URL**: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
- **Size**: 103 trials across 3 tasks
- **Tasks**: Suturing, knot tying, needle passing — but on **da Vinci ROBOTIC system**, not laparoscopic box trainer
- **Labels**: Gesture labels + OSATS-like skill scores (novice/intermediate/expert from surgeons with 10h to >100h experience)
- **License**: Research use
- **CRITICAL LIMITATION**: Robotic surgery (da Vinci), not manual laparoscopic. Visual domain is very different (robotic arms, different camera, different ergonomics). Would introduce significant domain shift.
- **Estimated yield**: Maybe 50–70 pairs if we accept domain shift, but risky
- **Action**: SKIP for now. Consider only after box-trainer data is exhausted.

### SLAM (Surgical LAparoscopic Motions)
- **URL**: Figshare (Nature Scientific Data, May 2025)
- **Size**: 4,097 video clips from real laparoscopic procedures
- **Tasks**: Cholecystectomy, appendectomy — real OR procedures, NOT box trainer
- **Labels**: 7 action categories (abdominal entry, clip use, hook cut, suturing, panoramic view, etc.)
- **License**: CC-BY
- **CRITICAL LIMITATION**: Real OR, not simulator/box trainer. Suturing clips exist but context is completely different (real tissue, bleeding, anatomical complexity). Would cause severe domain shift if mixed naively.
- **Estimated yield**: Suturing clips (~500) could be useful for a separate "OR transfer" model later
- **Action**: SKIP for initial training. Flag for Phase 5 (domain adaptation).

### Cholec80 / CholecT50
- **URL**: https://camma.unistra.fr/datasets/
- **Size**: 80 cholecystectomy videos (Cholec80), 50 with triplet annotations (CholecT50)
- **Labels**: Phase annotations, tool presence, action triplets
- **License**: Research use (form required)
- **CRITICAL LIMITATION**: Same as SLAM — real OR cholecystectomy, not FLS box trainer tasks. No skill scoring.
- **Action**: SKIP entirely for FLS scoring/coaching.

### LSPD (Laparoscopic Surgical Performance Dataset)
- **URL**: Described in Nature Scientific Reports (June 2025)
- **Size**: Custom dataset — Stack, Bands, Tower simulated skills
- **Tasks**: Simulated laparoscopic tasks (NOT standard FLS tasks)
- **Labels**: Novice/trainee/expert classification
- **License**: Unclear if public
- **CRITICAL LIMITATION**: Non-FLS tasks. Different simulator. May not be publicly released.
- **Action**: SKIP.

---

## Tier 3: YouTube Expansion — Untapped Task 1–4 Videos

Current harvest is 100% Task 5. The v002 prompts now support all 5 tasks. **YouTube is the fastest path to Task 1–4 coverage.**

### High-Priority Search Terms (NOT in current harvest)

**Task 1 — Peg Transfer:**
- "FLS peg transfer box trainer"
- "laparoscopic peg transfer practice"
- "FLS task 1 training"
- "peg transfer novice vs expert"
- "laparoscopic peg transfer timed"

**Task 2 — Pattern Cut:**
- "FLS pattern cut practice"
- "laparoscopic precision cutting circle"
- "FLS task 2 training"
- "laparoscopic cutting exercise box trainer"

**Task 3 — Endoloop:**
- "FLS endoloop task"
- "laparoscopic ligating loop box trainer"
- "FLS task 3 endoloop training"
- "endoloop placement practice"

**Task 4 — Extracorporeal Suture:**
- "FLS extracorporeal knot tying"
- "laparoscopic extracorporeal suture box trainer"
- "FLS task 4 training"
- "knot pusher technique laparoscopic"

**Multi-task / General:**
- "FLS all 5 tasks demonstration"
- "FLS exam practice full"
- "SAGES FLS tutorial complete"
- "FLS proficiency test preparation"
- "fundamentals of laparoscopic surgery complete training"

### Channels to Target (from existing harvest + new)
- sjhsurgery (FLS tutorials)
- SAGES (official society)
- Lapskills
- Laparoscopicboxx
- CeMSIM (simulation center)
- Inovus Medical
- University residency program channels

### Estimated Yield
Conservative estimate: 30–50 videos per task for Tasks 1–4, plus 30+ additional Task 5. Total: 150–250 new videos.

---

## Data Scaling Roadmap

| Phase | Source | New Videos | Cumulative | Timeline |
|-------|--------|-----------|------------|----------|
| Current | Personal + YT harvest | 78 | 78 | Done |
| 1 | Lower confidence filter + re-prep | 0 new videos, more pairs | ~100 pairs | Immediate |
| 2 | YouTube Tasks 1–4 harvest | 150–250 | 230–330 | This week |
| 3 | LASANA integration | 800–1,000 | 1,000–1,300 | 1–2 weeks (pending access) |
| 4 | LASK (if released) | 80 | 1,100–1,400 | TBD |
| 5 | Domain adaptation (SLAM/Cholec80) | 500 (suturing subset) | 1,600–1,900 | Future |

## Bottom Line

**Grok's LASANA recommendation is correct and high-value, but access may not be instant** — the paper is from Feb 2026 and the OPARA platform recently migrated. Need to verify the download is live.

**PhysioNet NIBIB-RPCCC-FLS is a dead end for us** — no video data, only EEG/eye-gaze + scores. Grok overstated its usefulness.

**YouTube Tasks 1–4 is the fastest, most reliable path to expanding coverage right now.** No access requests, no waiting for dataset releases. Just harvest and score.

**JIGSAWS, SLAM, Cholec80 are NOT suitable** for initial training — they're robotic or real OR, not box trainer. Mixing them in would introduce domain shift that hurts more than it helps at this stage.
