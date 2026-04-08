# JIGSAWS — JHU-ISI Gesture and Skill Assessment Working Set

> **Status (2026-04-08):** registration pending. README + download
> script stub committed; expected size ~12 GB so it can land on
> either RunPod alongside LASANA POD=B (SutureAndKnot) since it's
> the same surgical task family.

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

## Source

- **Official release:** https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
- **Maintainer email (for download issues):** jigsaws.cirl@gmail.com
- **Owners:** Johns Hopkins University CIRL + Intuitive Surgical Inc.
- **Paper:** Gao et al., "JHU-ISI Gesture and Skill Assessment Working
  Set (JIGSAWS): A Surgical Activity Dataset for Human Motion
  Modeling." MICCAI Workshop M2CAI 2014.
  Companion paper: PMC5559351 (2017 benchmark report).
- **License:** academic research only. ⚠️ Not commercial.

## What it contains

**103 trials of basic surgical skills** performed by 8 right-handed
surgeons spanning 3 self-reported skill levels (novice / intermediate /
expert), recorded on a da Vinci Surgical System:

| Task          | Trials |
|---------------|-------:|
| Suturing      |     39 |
| Knot Tying    |     36 |
| Needle Passing|     28 |
| **Total**     | **103** |

Per trial:
- **Stereo endoscope video** (left + right cameras, ~30 fps).
- **Kinematic data** at 30 Hz from both Master Tool Manipulators and
  both Patient-Side Manipulators (76 channels: positions, rotation
  matrices, linear/angular velocities, gripper angles).
- **Manual gesture annotation:** atomic surgical-gesture segments
  (G1...G15 vocabulary).
- **Skill annotation:** Modified Global Operative Assessment of
  Laparoscopic Skills (M-GOALS) Global Rating Score per trial, plus
  self-reported skill level of the operator.
- **Standardized cross-validation splits** (LOSO + LOUO).

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
