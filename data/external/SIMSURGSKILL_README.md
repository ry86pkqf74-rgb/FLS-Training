# SimSurgSkill 2021 — MICCAI EndoVis Sub-Challenge

> **Status (2026-04-08):** ✅ **publicly downloadable, no
> registration**. Single 4.61 GB ZIP at
> `gs://isi-simsurgskill-2021/simsurgskill_2021_dataset.zip`.
> Download script `scripts/065_simsurgskill_download.sh` added in
> this PR; can run on Mac or any RunPod with no auth.

## Source

- **GCS bucket (public):** `gs://isi-simsurgskill-2021/`
  - File: `simsurgskill_2021_dataset.zip` — **4,613,059,266 bytes (4.61 GB)**
  - Direct HTTPS: https://storage.googleapis.com/isi-simsurgskill-2021/simsurgskill_2021_dataset.zip
- **Bucket listing:** https://storage.googleapis.com/storage/v1/b/isi-simsurgskill-2021/o
- **Challenge home:** https://www.synapse.org/SimSurgSkill (2021 EndoVis sub-challenge)
- **Paper:** Funke et al., "Objective Surgical Skills Assessment and
  Tool Localization: Results from the MICCAI 2021 SimSurgSkill
  Challenge." arXiv:2212.04448 (2022).
- **License:** non-commercial research only. ⚠️

## What it contains

VR exercises performed on a **da Vinci simulator** by participants of
varying skill levels. Each video is a single endoscope channel,
**1280 × 720 @ 30 fps**, plus per-trial **objective performance
indicators (OPIs)** and per-frame tool bounding boxes.

OPIs published per trial include (verbatim from the challenge):

- **Needle drop count** (suturing-style tasks)
- **Instrument out-of-view count**
- **Economy of motion** (path length)
- **Excessive instrument force events**
- **Total task time**

These OPIs are objective skill proxies — usable directly as
absolute-score targets without any teacher scoring. Each one maps
cleanly to a sub-component of FLS GRS (efficiency, instrument
handling, safety).

## FLS task mapping

SimSurgSkill is **VR da Vinci simulator** content, so the visual
domain is again robotic, not laparoscopic. The tasks aren't a 1:1
match to FLS, but the **skill signal generalizes**:

- **Needle drop count → FLS Task 5** (intracorporeal suturing) — same
  failure mode.
- **Economy of motion** → applicable to all 5 FLS tasks.
- **Out-of-view count** → applicable to all 5 FLS tasks.
- **Total time** → applicable to all 5 FLS tasks (though FLS uses a
  task-specific cutoff, not raw time).

We use SimSurgSkill the same way as JIGSAWS: as **pre-training data
for the rating head**, plus as a held-out generalization probe for
non-FLS-specific motion-economy signals.

## Why it's worth pulling

1. **Zero friction.** No registration, no DUA click-through, no
   email dance. `curl` it and unzip.
2. **Small** — 4.6 GB compressed, probably ~10-15 GB uncompressed.
   Fits anywhere.
3. **Objective performance indicators** are pre-computed and rater
   independent — adds a different *kind* of training signal than
   the GRS-based labels in LASANA / JIGSAWS / PETRAW.
4. **Tool bounding boxes** — bonus signal for any future
   tool-tracking auxiliary task.

## Download (live)

```bash
bash scripts/065_simsurgskill_download.sh        # default: ./data/external/simsurgskill
DEST=/workspace/simsurgskill bash scripts/065_simsurgskill_download.sh   # RunPod
```

The script:
- Uses `curl -fSL --retry 5 -C - -o NAME.tmp` with atomic rename, so
  it's safely resumable.
- Verifies the downloaded size against the bucket-published byte count
  (4,613,059,266) before unzipping.
- Unpacks into `${DEST}/simsurgskill_2021_dataset/`.

## Expected on-disk layout (post-unzip)

```
data/external/simsurgskill/
├── _meta/
│   └── README.* / LICENSE.*    # bundled with the zip
├── simsurgskill_2021_dataset/
│   ├── train/
│   │   ├── videos/             # *.mp4
│   │   ├── annotations/        # OPI CSVs + tool bboxes
│   │   └── ...
│   └── test/
│       ├── videos/
│       ├── annotations/
│       └── ...
```

(Exact internal layout will be confirmed once the zip is unpacked.)

## Integration TODO

- [ ] Run `065_simsurgskill_download.sh` on the target host.
- [ ] Unzip and document the actual internal layout in this README.
- [ ] Write `scripts/066_ingest_simsurgskill.py`:
    - Read the OPI CSV per trial, emit one record per OPI as a
      separate scalar absolute-score target.
    - Map trial → internal `video_id` (prefix `simsurg_`).
    - Tag records with `source: simsurgskill`, `domain: vr_robotic`,
      `task: <closest FLS task>`.
- [ ] Add `simsurgskill` row to `data/DATA_INVENTORY.md`.

## References

- Paper: https://arxiv.org/abs/2212.04448
- Bucket: https://storage.googleapis.com/storage/v1/b/isi-simsurgskill-2021/o
- Challenge page: https://www.synapse.org/SimSurgSkill
