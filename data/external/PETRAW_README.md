# PETRAW — PEg TRAnsfer Workflow Recognition (EndoVis 2021)

> **Status (2026-04-08):** registration pending. README + download
> script stub committed; videos will land on the same RunPod that
> hosts LASANA POD=A (PegTransfer + CircleCutting), since this is
> another peg-transfer corpus.

## Source

- **Challenge home:** https://www.synapse.org/PETRAW
- **Dataset platform:** Synapse (Sage Bionetworks). Free account, click-through
  data use agreement (non-commercial research).
- **Paper:** "PEg TRAnsfer Workflow recognition challenge report" (Huaulmé
  et al., MedIA 2023). DOI: 10.1016/j.media.2023.102844 — arXiv:2202.05821.
- **EndoVis page:** https://opencas.dkfz.de/endovis/challenges/2021/
- **License:** non-commercial research. ⚠️ Confirm before training a
  model we plan to release commercially.

## What it contains

**150 peg-transfer training sequences** captured on a Virtual Reality
(VR) peg-transfer simulator, split 90 train / 60 test. For every case
the dataset provides:

- A **video** of the simulator screen (RGB, full sequence).
- Synchronized **kinematics** of both instrument tips (positions +
  velocities, 30 Hz).
- **Semantic segmentation** of every frame (peg, block, instrument
  left, instrument right, etc.).
- **Workflow annotation:** phase, step, and atomic-action labels frame
  by frame, plus a sequence-level skill label.

Estimated total size: ~25-40 GB (no exact figure published; will be
recorded once Synapse access lands).

## FLS task mapping

- **FLS Task 1 (Peg Transfer):** direct 1:1. PETRAW gives us a
  second independent peg-transfer corpus on top of LASANA's 329
  PegTransfer clips, but on a *different visual domain* (VR
  simulator vs. dry-lab box trainer with real instruments).

This is exactly the cross-domain signal we want for FLS Task 1, which
currently has zero training examples in the production set.

## Why it's worth pulling

1. **Direct FLS Task 1 coverage** — our weakest task right now.
2. **VR ≠ dry-lab visual domain** — gives the model a chance to learn
   peg-transfer skill features that generalize across rendering
   styles, which is critical because user submissions to
   FLS-Training span both VR and physical box trainers.
3. **Frame-level workflow + skill labels** — usable directly as
   absolute-score training targets without any teacher scoring spend.
4. **Kinematics** — we don't currently use kinematic features, but
   they're a future option for distillation experiments.

## Access procedure

1. Create a free Synapse account: https://www.synapse.org/Profile:create
2. Visit https://www.synapse.org/PETRAW
3. Accept the data-use agreement (non-commercial research click-through).
4. From the Files tab, copy the Synapse IDs of the bitstreams.
5. `pip install synapseclient --break-system-packages`
6. `synapse login -u <user> -p <pat>`
7. `synapse get -r syn<id>` for each archive into
   `data/external/petraw/` (gitignored).

## Download script

`scripts/062_petraw_download.sh` (added in this PR) is a stub — it
will be filled in once we have the Synapse file IDs. The expected
on-disk layout mirrors LASANA:

```
data/external/petraw/
├── _meta/             # synapse manifest, license, etc.
├── annotations/       # phase/step/action CSVs + skill labels
├── kinematics/        # per-trial CSVs
├── segmentation/      # per-frame masks (zipped)
└── videos/            # 150 .mp4 sequences
```

## Integration TODO

- [ ] Get Synapse account + accept DUA.
- [ ] Pull file manifest, fill in `062_petraw_download.sh` with IDs.
- [ ] Verify checksums against Synapse-published MD5s.
- [ ] Write `scripts/063_ingest_petraw.py` mirroring the LASANA
      ingestion script — emit one record per trial with the published
      sequence-level skill label as the absolute-score target,
      `source: petraw`, `task: 1` (peg transfer).
- [ ] Honor the official 90/60 train/test split as `split: train` /
      `split: test`. Carve a val set from the train half (random 10%).
- [ ] Add `petraw` row to `data/DATA_INVENTORY.md`.

## References

- arXiv: https://arxiv.org/abs/2202.05821
- Synapse: https://www.synapse.org/PETRAW
- EndoVis 2021: https://opencas.dkfz.de/endovis/challenges/2021/
