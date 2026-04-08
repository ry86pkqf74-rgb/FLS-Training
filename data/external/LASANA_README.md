# LASANA — Laparoscopic Skill Analysis and Assessment

> **Status (2026-04-08):** metadata, annotations, calibration file, and the
> 4 example videos are downloaded and unpacked locally. **Full archive
> download (185.5 GB across all 12 bitstreams — both stereo cameras for
> all 4 tasks) is authorized and will be staged on dedicated RunPod
> servers** (no longer left-only / no task truncation). Per-server file
> distribution and the full bitstream UUID list are in `_meta/`. See
> *Download plan* below.

## Source

- **Item URL:** https://opara.zih.tu-dresden.de/handle/123456789/1907
- **DOI / handle:** 123456789/1907 (TU Dresden Opara)
- **Item UUID:** `629966ba-4d11-475c-a689-9959204f3be4`
- **Released:** 2025-12-26
- **Paper:** "A benchmark for video-based laparoscopic skill analysis and
  assessment" (arXiv 2602.09927, Feb 2026)
- **License: Creative Commons Attribution 4.0 International (CC BY 4.0).**
  ✅ Commercial use OK with attribution. Confirmed via the Opara API.

## What it actually contains

**1,270 trimmed and synchronized stereo video recordings** of four basic
laparoscopic training tasks. Stereo means each recording has a left-camera
and a right-camera file (separate `.mkv`), encoded H.264 in Matroska, at
**960 × 540 @ 20 fps**. Camera intrinsics + relative pose are in
`_meta/camera_calibration.yaml`.

### Per-task counts (verified against Annotation.zip)

| LASANA task          | n recordings | train | val | test | Mean duration | Max duration |
|----------------------|-------------:|------:|----:|-----:|--------------:|-------------:|
| PegTransfer          | 329          | 243   |  32 |  54  |  152 s (2.5m) | 357 s |
| CircleCutting        | 311          | 234   |  27 |  50  |  212 s (3.5m) | 592 s |
| BalloonResection     | 316          | 235   |  30 |  51  |  235 s (3.9m) | 579 s |
| SutureAndKnot        | 314          | 232   |  32 |  50  |  270 s (4.5m) | 601 s |
| **Total**            | **1,270**    | 944   | 121 | 205  |               |       |

### Annotations (per task, in `annotations/Annotation/`)

For each task `{T}`, the Annotation.zip provides:

- `{T}.csv` — main annotations: `id, duration, frame_count, GRS,
  bimanual_dexterity, depth_perception, efficiency, tissue_handling,` plus
  one boolean column per task-specific error.
- `{T}_split.csv` — predefined train/val/test split with column `split`.
- `{T}_rater0.csv` … `{T}_rater3.csv` — pre-normalisation per-rater scores
  on each of the four aspects + a per-rater GRS.

**Skill rating scheme.** Each recording is rated by 3 of 4 human raters on
four aspects (`depth_perception, efficiency, bimanual_dexterity,
tissue_handling`) on a 5-point Likert scale. The published `GRS` column is
the average of the per-rater sums after **z-score normalisation per rater**
(removes systematic rater bias). Two of the four raters scored every clip;
the other two scored complementary halves. Failed recordings are flagged
with normalised GRS = -2.50 (no per-aspect scores).

**Observed GRS distribution (computed locally):**

| Task              | n | mean | sd  | min  | max  |
|-------------------|---:|-----:|----:|-----:|-----:|
| PegTransfer       | 329 |  0.00 | 0.88 | -2.22 | 2.12 |
| CircleCutting     | 311 |  0.00 | 0.81 | -2.19 | 1.92 |
| BalloonResection  | 316 | -0.29 | 1.13 | -2.50 | 2.29 |
| SutureAndKnot     | 314 | -0.17 | 1.07 | -2.50 | 1.99 |

GRS is roughly N(0,1) by construction (z-scored), with a heavy left tail
from failed BalloonResection / SutureAndKnot attempts. Zero failed clips on
PegTransfer / CircleCutting — those tasks are fully scored.

### Task-specific error labels

PegTransfer: `object_dropped_within_fov`, `object_dropped_outside_of_fov`.

CircleCutting: `cutting_imprecise`, `gauze_detached`.

BalloonResection: `cutting_imprecise`, `cutting_incomplete`, `balloon_opened`,
`balloon_damaged`, `balloon_perforated`.

SutureAndKnot: `needle_dropped`, `suture_imprecise`, `fewer_than_three_throws`,
`slit_not_closed`, `knot_comes_apart`, `drain_detached`.

These boolean columns are gold for training a separate per-error classifier
or for stratifying the training set.

## Mapping to FLS tasks (verified — table updated)

| LASANA task        | FLS task # | FLS name                      | Notes |
|--------------------|:----------:|-------------------------------|-------|
| PegTransfer        | **1**      | Peg Transfer                  | Direct 1:1 mapping. |
| CircleCutting      | **2**      | Pattern Cut                   | Same task: cut around a pre-marked circle on gauze. |
| BalloonResection   |  —         | (not in FLS curriculum)       | Closest analog is none of the 5 FLS tasks; treat as a separate "domain transfer" task or skip. |
| SutureAndKnot      | **5**      | Intracorporeal Suturing       | Three-throw knot on a Penrose drain — exactly FLS Task 5. |

LASANA does **not** cover FLS Task 3 (Endoloop / Ligating Loop) or
FLS Task 4 (Extracorporeal Knot Tying). Those still need to come from
YouTube harvest or other sources.

## On-disk layout

```
data/external/lasana/
├── README.md                     # this file (was LASANA_README.md, see below)
├── Readme.md                     # original LASANA README from the dataset
├── _meta/
│   └── camera_calibration.yaml   # k1, k2, d1, d2, R, t (1606 fx, 825/1066 cx)
├── annotations/
│   ├── Annotation.zip            # 70 KB, source archive
│   └── Annotation/
│       ├── {Task}.csv            # GRS + per-aspect + error labels
│       ├── {Task}_split.csv      # train/val/test split
│       └── {Task}_rater[0-3].csv # raw per-rater scores
├── samples/
│   ├── example_videos.zip        # 143 MB, source archive
│   └── example_videos/
│       ├── PegTransfer.mkv
│       ├── CircleCutting.mkv
│       ├── BalloonResection.mkv
│       └── SutureAndKnot.mkv
└── videos/                       # staged on dedicated RunPod servers
    ├── PegTransfer_left.zip      # 16.21 GB
    ├── PegTransfer_right.zip     # 16.61 GB
    ├── CircleCutting_left.zip    # 23.16 GB
    ├── CircleCutting_right.zip   # 23.93 GB
    ├── BalloonResection_left.zip # 24.76 GB
    ├── BalloonResection_right.zip# 25.42 GB
    ├── SutureAndKnot_left.zip    # 27.32 GB
    └── SutureAndKnot_right.zip   # 27.97 GB
```

## Download plan (FULL — 185.51 GB across 12 bitstreams)

All 12 bitstreams are downloaded in full. Stereo (left + right) for all 4
tasks, plus annotations / samples / metadata. No task is dropped:
**BalloonResection is included** even though it doesn't map to an FLS task —
it's still a high-quality stereo skill-rating signal we can use for
pre-training, domain adaptation, or as a held-out generalization probe.

| File                       | Size     | Source UUID |
|----------------------------|---------:|-------------|
| Readme.md                  |   6.8 KB | 93d6cd40… |
| Annotation.zip             |    70 KB | aad0a7eb… |
| camera_calibration.yaml    |    0.9 KB | c527ce4a… |
| example_videos.zip         |  143 MB  | b58fc1a1… |
| PegTransfer_left.zip       | 16.21 GB | f577d564… |
| PegTransfer_right.zip      | 16.61 GB | e9b58d48… |
| CircleCutting_left.zip     | 23.16 GB | da15f935… |
| CircleCutting_right.zip    | 23.93 GB | 71209477… |
| BalloonResection_left.zip  | 24.76 GB | 08bc523e… |
| BalloonResection_right.zip | 25.42 GB | 7f522ef6… |
| SutureAndKnot_left.zip     | 27.32 GB | 005bdc22… |
| SutureAndKnot_right.zip    | 27.97 GB | cc8e197b… |
| **TOTAL**                  | **185.51 GB** | |

Full URL list with checksums is in `_meta/bitstreams.json` (auto-fetched
from the Opara API). Each `content` link follows the pattern:
`https://opara.zih.tu-dresden.de/server/api/core/bitstreams/<uuid>/content`

### Distribution across RunPod servers

The dataset is staged on dedicated RunPod servers — disk is not a
constraint. The download is split so two large files run in parallel per
server (avoid hammering Opara) and so left/right stereo pairs land on the
same machine (lets the ingestion script open both cameras without a
cross-host hop). Suggested split:

| RunPod node | Files                                                                 | Total    |
|-------------|-----------------------------------------------------------------------|---------:|
| pod-A       | PegTransfer L+R, CircleCutting L+R                                    | 79.91 GB |
| pod-B       | BalloonResection L+R, SutureAndKnot L+R                               | 105.47 GB |
| pod-A *or* B | annotations + metadata + example_videos (small, replicate to both)   |  144 MB  |

Per-pod download script template lives at
`scripts/061_lasana_download_runpod.sh` (added in the same PR — uses
`curl -fSL --retry 5 -C - -o NAME.tmp` with atomic rename, runs 2 files in
parallel, resumable). Drop the script onto each pod and run; both pods can
go in parallel.

## Estimated training-pair yield (revised with real numbers)

Now that we know the per-task counts and that GRS is already a clean
continuous label, the yield calculus is much better than the prior
estimate:

- **PegTransfer:** 329 clips × 1 GRS-derived absolute-score example
  = 329 examples *without any teacher scoring*, plus ~3-5 pairs per
  clip ≈ 1,500 high-confidence pairs.
- **CircleCutting:** ~311 absolute + ~1,400 pairs.
- **BalloonResection:** ~316 absolute + ~1,400 pairs (domain-transfer
  / pre-training, no FLS task mapping).
- **SutureAndKnot:** ~314 absolute + ~1,400 pairs.
- **Total LASANA-only training signal:** ~1,270 absolute-score
  examples and up to ~5,700 pairwise examples — without spending a
  cent on teacher scoring. With both stereo cameras you can also build
  a stereo-consistency auxiliary loss or simply double the example
  count by treating each camera as an independent view.

This single dataset alone is enough to blow past the 500-example target
in `docs/DATA_SCALING_PLAN.md` and make Phase 3 the dominant
contributor.

## Integration checklist

- [x] Download metadata + annotations + samples.
- [ ] Wait for the three left-camera archives to finish (background).
- [ ] Unpack into `videos/{task}_left/` and verify file count matches
      annotations.
- [ ] Write `scripts/060_ingest_lasana.py`:
    - Read `{task}.csv`, emit one ingestion record per recording with the
      GRS as the pre-scored signal.
    - Map LASANA recording id → internal `video_id` (prefix with `lasana_`).
    - Honour the predefined `{task}_split.csv` train/val/test assignment so
      we don't accidentally leak.
- [ ] Update the dataset prep script (`scripts/040_prepare_training_data.py`)
      to recognise `source: lasana` and to skip teacher scoring when the
      record already carries a GRS.
- [ ] Confirm task name → FLS task mapping in `data/training/.../manifest.yaml`.
- [ ] Update `data/DATA_INVENTORY.md` after the first LASANA-augmented
      training run.

## References

- Item page: https://opara.zih.tu-dresden.de/handle/123456789/1907
- arXiv: https://arxiv.org/abs/2602.09927
- Original dataset Readme: see `Readme.md` next to this file.
- License: https://creativecommons.org/licenses/by/4.0/
