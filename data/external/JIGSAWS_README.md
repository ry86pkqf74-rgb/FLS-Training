# JIGSAWS — JHU-ISI Gesture and Skill Assessment Working Set

## ⚠️ MANDATORY CITATION

JHU-CIRL requires that any work using JIGSAWS cite the publication
listed on their release page. This is a **condition of access**, not
a courtesy. The full citation block is in
`data/external/CITATIONS.md` and is reproduced below — copy it into
every paper, blog post, and model card derived from this work.

> Gao, Y., Vedula, S.S., Reiley, C.E., Ahmidi, N., Varadarajan, B.,
> Lin, H.C., Tao, L., Zappella, L., Béjar, B., Yuh, D.D., Chen,
> C.C.G., Vidal, R., Khudanpur, S., Hager, G.D. *JHU-ISI Gesture and
> Skill Assessment Working Set (JIGSAWS): A Surgical Activity Dataset
> for Human Motion Modeling.* MICCAI Workshop: M2CAI, 2014.

If you use the gesture or skill labels (i.e., almost certainly), also
cite the 2017 benchmark paper (Ahmidi et al., IEEE TBME 64(9), PMC5559351).

---

> **Status (2026-04-08):** ✅ **All 4 JIGSAWS archives downloaded
> via Chrome and integrated as metadata-only / kinematics-only.**
>
> **Important:** JHU-CIRL only released kinematics + meta files +
> gesture transcriptions + experimental setup splits. **There is no
> video archive available** — the 3 short sample AVI clips in
> `samples/` are the only frames JHU exposes publicly. We're
> integrating JIGSAWS as a metadata + kinematics + GRS-label signal
> source, not as a visual training corpus.
>
> Total trial count: **103** (Suturing 39 + Knot_Tying 36 + Needle_Passing 28),
> exactly matching the published numbers.
>
> ⚠️ **License:** academic / non-commercial research only. The user has
> confirmed FLS-Training is being used as a research project, which is
> within the JIGSAWS DUA scope.

## Source

- **Official release:** https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
- **Maintainer email (for download issues):** jigsaws.cirl@gmail.com
- **Owners:** Johns Hopkins University CIRL + Intuitive Surgical Inc.
- **Paper:** Gao et al., "JHU-ISI Gesture and Skill Assessment Working
  Set (JIGSAWS): A Surgical Activity Dataset for Human Motion
  Modeling." MICCAI Workshop M2CAI 2014.
  Companion paper: PMC5559351 (2017 benchmark report).
- **License:** academic research only. ⚠️ Not commercial.

## What it contains (verified locally 2026-04-08)

**103 trials of basic surgical skills** performed by 8 right-handed
surgeons spanning 3 self-reported skill levels:

| Task          | Trials | Novice | Intermediate | Expert |
|---------------|-------:|-------:|-------------:|-------:|
| Suturing      |     39 |     19 |           10 |     10 |
| Knot_Tying    |     36 |     16 |           10 |     10 |
| Needle_Passing|     28 |     11 |            8 |      9 |
| **Total**     | **103**|   **46** |       **28** |   **29** |

Per trial **what's actually in the JIGSAWS public release**:
- **Kinematic data** at 30 Hz, 76 channels (positions, rotation
  matrices, linear/angular velocities, gripper angles for both Master
  Tool Manipulators and both Patient-Side Manipulators). See per-task
  `readme.txt` for the full channel layout.
- **Manual gesture annotation:** per-trial frame-range labels using
  the G1...G15 vocabulary (`transcriptions/<trial>.txt`, format:
  `<start_frame> <end_frame> <gesture_id>`).
- **Skill annotation:** GRS per trial (sum of 6 sub-scores, range
  6-30) plus self-reported skill level. Format in `meta_file_<task>.txt`:
  ```
  <trial_id>  <skill_level (E/I/N)>  <GRS>  <r1> <r2> <r3> <r4> <r5> <r6>
  ```
  Sub-scores in order: respect-for-tissue, suture/needle-handling,
  time-and-motion, flow-of-operation, overall-performance,
  quality-of-final-product (each 1-5).
- **Standardized cross-validation splits** in `Experimental_setup/`
  (LOSO / LOUO / OneTrialOut / UserOut variants for both Gesture
  Classification and Skill Detection tasks). 16,328 small Test.txt /
  Train.txt files arranged by `Task / Balanced or unBalanced /
  Classification or Detection / OneTrialOut or UserOut / N_Out / itr_M /`.

**NOT in the public release:**
- ❌ Stereo endoscope video (only 3 short sample clips total in `samples/`).

## Aggregate GRS statistics (computed locally)

Across all 103 trials:

| Task           | n  | mean | sd  | min | max |
|----------------|---:|-----:|----:|----:|----:|
| Suturing       | 39 | 19.13 | 5.40 |   8 |  30 |
| Knot_Tying     | 36 | 14.42 | 5.11 |   6 |  22 |
| Needle_Passing | 28 | 14.29 | 4.82 |   7 |  24 |
| **All**        | 103 | 16.17 | 5.60 |   6 |  30 |

Per-component (1-5 scale, all 103 trials pooled):

| Component             | mean | sd  |
|-----------------------|-----:|----:|
| Respect for tissue    | 2.58 | 1.11 |
| Suture/needle handling| 2.50 | 1.04 |
| Time and motion       | 2.24 | 1.04 |
| Flow of operation     | 3.17 | 0.98 |
| Overall performance   | 2.56 | 1.03 |
| Quality of final product | 3.12 | 1.05 |

## FLS task mapping

| JIGSAWS task   | FLS task # | FLS name                  | Notes |
|----------------|:----------:|---------------------------|-------|
| Suturing       | **5**      | Intracorporeal Suturing   | Direct skill match. |
| Knot Tying     | **4 / 5**  | Extracorporeal / Intra    | Knot family. |
| Needle Passing | adjunct    | (sub-skill of FLS 5)      | Useful as a finer-grained signal. |

**Domain caveat:** JIGSAWS is robotic (da Vinci with EndoWrist tools),
not laparoscopic with rigid sticks. The visual domain is meaningfully
different — robotic instrument tips look nothing like Maryland graspers.
But the **skill labels are gold** and the gesture taxonomy is the
gold-standard reference for surgical skill assessment ML. We treat
JIGSAWS as:

1. A **pre-training corpus** for the rating head (the model learns
   what "good knot tying" looks like, then we fine-tune on
   FLS-domain data).
2. A **held-out generalization probe** for the final scorer.

## Why it's worth pulling

1. **Best-curated surgical skill dataset in existence.** Used by
   essentially every published paper on surgical skill assessment.
2. **GRS labels by trained raters** — same scoring philosophy as
   ours, so no rubric translation needed.
3. **Gesture-level segmentation** — enables training a temporal
   action segmentation auxiliary head.
4. **Cross-validation splits already published** — directly
   comparable to literature numbers.
5. **Free** — only cost is filling out the access form.

## Access procedure

1. Visit https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
2. Fill in the request form (institution, name, intended use). Same-day
   approval has historically been the norm.
3. The reply email contains a download script + password.
4. If the script fails, email **jigsaws.cirl@gmail.com** for an
   alternative link.
5. Save under `data/external/jigsaws/` (gitignored).

## Expected on-disk layout

```
data/external/jigsaws/
├── _meta/
│   ├── README.txt
│   └── License.txt
├── Suturing/
│   ├── kinematics/          # per-trial *.txt (76 channels)
│   ├── transcriptions/      # gesture-segment labels
│   ├── meta_file_Suturing.txt  # GRS + self-reported skill per trial
│   └── video/               # *_capture1.avi (left), *_capture2.avi (right)
├── Knot_Tying/  (same structure)
└── Needle_Passing/  (same structure)
```

Approximate total size: ~12 GB (videos dominate).

## Integration TODO

- [ ] Submit access request form.
- [ ] On approval: run the official download script into
      `data/external/jigsaws/`.
- [ ] Verify the 103 trial count and the meta_file GRS distributions.
- [ ] Write `scripts/064_ingest_jigsaws.py`:
    - Read `meta_file_<Task>.txt` to get GRS + self-reported skill per
      trial.
    - Map trial → internal `video_id` (prefix `jigsaws_`).
    - Use only `_capture1.avi` (left camera) by default; treat
      `_capture2.avi` as optional stereo augmentation.
    - Honor LOSO/LOUO splits in the manifest.
- [ ] Tag records as `source: jigsaws`, `domain: robotic`,
      `task: 4` or `task: 5`.
- [ ] Add `jigsaws` row to `data/DATA_INVENTORY.md`.

## References

- Release page: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
- Benchmark paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC5559351/
- Original report PDF: https://cirl.lcsr.jhu.edu/wp-content/uploads/2015/11/JIGSAWS.pdf
