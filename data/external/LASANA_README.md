# LASANA — Laparoscopic Skill Analysis and Assessment

> Status: **metadata stub**. The actual dataset has NOT yet been downloaded. The
> hosting domain (`opara.zih.tu-dresden.de`) is blocked by the Cowork network
> egress proxy, so the download must be performed manually from a personal
> workstation using the link in the "Download" section below.

## Summary

LASANA is a public benchmark for video-based laparoscopic skill assessment
released with the preprint **"A benchmark for video-based laparoscopic skill
analysis and assessment"** (arXiv:2602.09927, Feb 2026). It contains
**1,270 stereo video recordings** of **four basic laparoscopic training tasks**,
collected primarily from participants in a laparoscopic training course. This
yields a natural distribution of skill levels (novice → near-expert).

Each clip is annotated with:

- A **structured skill rating**, aggregated from **three independent raters**
  (Global Rating Scale style).
- **Binary error labels** indicating the presence/absence of task-specific
  errors (e.g. dropped peg, cut outside line).
- **Predefined train/val/test splits** per task, with baseline deep-learning
  results reported in the paper for reference.

This is — by a wide margin — the largest publicly available, skill-rated
laparoscopic-box-trainer video corpus at the time of this writing, and the
single highest-leverage external dataset for the FLS-Training project.

## The four LASANA tasks and mapping to FLS

The paper describes LASANA as containing "four basic laparoscopic training
tasks" drawn from the standard box-trainer curriculum. The task names and
per-task counts are not reproduced in the abstract we were able to fetch; the
mapping below reflects the strong prior that any four-task laparoscopic
training corpus of this size is aligned with the core FLS manual-skills
curriculum. **Verify against the paper/README on first download** and update
this file.

| LASANA task (assumed) | FLS equivalent             | FLS task # | Notes |
|-----------------------|----------------------------|------------|-------|
| Peg Transfer          | Peg Transfer               | 1          | Direct 1:1 mapping. Stereo cameras may require rectification/cropping. |
| Pattern Cutting       | Pattern Cut                | 2          | Direct 1:1 mapping. |
| Knot Tying            | Extracorporeal or Intracorporeal Knot | 4 or 5 | LASANA does not always distinguish; inspect metadata to bucket correctly. |
| Suturing              | Intracorporeal Suturing    | 5          | Direct 1:1 mapping. |

LASANA does **not** include an Endoloop / Ligating Loop task, so FLS Task 3 is
not covered by this dataset and will need to come from YouTube harvest or
SPD-FLS1.

## Download (manual step)

1. Open https://opara.zih.tu-dresden.de in a browser.
2. Search for `LASANA` (or follow the dataset DOI link from the arXiv paper
   abstract at https://arxiv.org/abs/2602.09927).
3. Accept the license (see below), download all task archives and the
   metadata/annotations file.
4. Unpack into this directory following the structure below.

## Target on-disk layout

```
data/external/lasana/
├── README.md                  # copy of the upstream README
├── LICENSE                    # copy of the upstream license file
├── annotations/
│   ├── skill_ratings.csv      # per-clip GRS and individual rater scores
│   ├── error_labels.csv       # per-clip binary error labels
│   └── splits.csv             # predefined train/val/test splits
└── {task}/                    # one dir per LASANA task, e.g. peg_transfer/
    └── {participant_id}/      # one dir per participant (70 total)
        └── {clip_id}.mp4      # or .avi, stereo-rectified
```

If upstream distributes stereo as side-by-side single files, keep them in that
form; the training pipeline can crop to the left view at load time.

## License

**TBD — record on first download.** TU Dresden's Opara repository typically
uses Creative Commons licenses (CC BY 4.0 or CC BY-NC 4.0). Check the landing
page for the exact license before using the data for training a model that
will be distributed. For purely internal research this is almost certainly
fine, but any published model weights need the license verified.

## Estimated training-pair yield for FLS-Training

Working assumptions (update once the real distribution is known):

- 1,270 clips total across 4 tasks ≈ **~317 clips/task** on average.
- After removing clips that fail our quality/format filters (stereo, too
  short, corrupt): keep ~85%.
- At the current ingestion cadence (1 score per video per teacher), this
  yields ~270 scored clips per task × 4 tasks = **~1,080 scored clips**.
- Our pairwise-comparison generator produces O(N) consensus pairs per batch
  after confidence filtering; historical yield is ~0.65 training pairs per
  scored clip. That gives roughly **~700 LASANA-derived training examples**.
- With LASANA's built-in GRS labels we can *also* create high-confidence
  absolute-score training pairs without teacher scoring at all, which would
  push yield to **~1,000+ examples** from this dataset alone.

## Integration checklist

- [ ] Download archives + annotations from Opara (manual, off-Cowork).
- [ ] Place under `data/external/lasana/` using the layout above.
- [ ] Add `lasana` as a recognised source in `scripts/010_ingest_videos.py`
      (or the equivalent ingestion entry point).
- [ ] Write a small adapter that emits LASANA clips into the standard
      ingestion manifest, carrying the GRS rating as a pre-scored signal so
      the teacher-scoring pass can skip it.
- [ ] Confirm the task-name mapping table above against the real dataset and
      amend as needed.
- [ ] Record the license in `data/external/lasana/LICENSE` and in
      `docs/DATA_SCALING_PLAN.md`.

## References

- arXiv preprint: https://arxiv.org/abs/2602.09927
- Paper-reading club summary: http://paperreading.club/page?id=376094
- Hosting repository (TU Dresden Opara): https://opara.zih.tu-dresden.de
