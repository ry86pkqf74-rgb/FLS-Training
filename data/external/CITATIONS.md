# Required Citations & Attribution for External Datasets

> **Read this before publishing any paper, blog post, model card, or
> public artifact derived from FLS-Training.** Every external dataset
> we use carries an attribution requirement under its license or DUA.
> Failing to cite is a license violation, not a courtesy. The
> ingestion scripts also tag every record with its `source` and
> `license` fields so we can audit downstream.

## LASANA — TU Dresden Opara

- **License:** CC BY 4.0 (commercial use OK with attribution)
- **Required citation:**

  > Carstens, M., Rinner, F., Bodenstedt, S., et al. *A benchmark for
  > video-based laparoscopic skill analysis and assessment.* arXiv
  > preprint arXiv:2602.09927, 2026.
  > https://arxiv.org/abs/2602.09927

- **Dataset DOI / handle:** TU Dresden Opara handle 123456789/1907 —
  https://opara.zih.tu-dresden.de/handle/123456789/1907
- **Attribution string** (for model cards, paper acknowledgments,
  product about-pages):

  > "This work uses the LASANA dataset (Carstens et al., 2026), released
  > by TU Dresden under CC BY 4.0."

## JIGSAWS — JHU-ISI Gesture and Skill Assessment Working Set

- **License:** Academic / non-commercial research only
- **Required citation** (per JHU-CIRL release page — citation is a
  condition of access):

  > Gao, Y., Vedula, S.S., Reiley, C.E., Ahmidi, N., Varadarajan, B.,
  > Lin, H.C., Tao, L., Zappella, L., Béjar, B., Yuh, D.D., Chen,
  > C.C.G., Vidal, R., Khudanpur, S., Hager, G.D. *JHU-ISI Gesture and
  > Skill Assessment Working Set (JIGSAWS): A Surgical Activity Dataset
  > for Human Motion Modeling.* MICCAI Workshop: M2CAI, 2014.

- **Companion benchmark paper (cite if using gesture or skill labels):**

  > Ahmidi, N., Tao, L., Sefati, S., Gao, Y., Lea, C., Bejar Haro, B.,
  > Zappella, L., Khudanpur, S., Vidal, R., Hager, G.D. *A Dataset and
  > Benchmarks for Segmentation and Recognition of Gestures in Robotic
  > Surgery.* IEEE Transactions on Biomedical Engineering, 64(9):
  > 2025-2041, 2017. PMC5559351.

- **Owners:** Johns Hopkins University CIRL + Intuitive Surgical Inc.
- **Release page:** https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
- **Attribution string:**

  > "This work uses the JIGSAWS dataset (Gao et al., 2014), released by
  > Johns Hopkins University CIRL and Intuitive Surgical Inc. for
  > academic research."

## PETRAW — PEg TRAnsfer Workflow Recognition (EndoVis 2021)

- **License:** Synapse non-commercial research DUA
- **Required citation:**

  > Huaulmé, A., Sarikaya, D., Le Mut, K., Despinoy, F., Long, Y., Dou,
  > Q., et al. *PEg TRAnsfer Workflow recognition challenge report:
  > Does multi-modal data improve recognition?* Medical Image Analysis,
  > 87: 102844, 2023. arXiv:2202.05821.

- **Synapse project:** syn25147789 — https://www.synapse.org/PETRAW
- **EndoVis page:** https://opencas.dkfz.de/endovis/challenges/2021/
- **Attribution string:**

  > "This work uses the PETRAW dataset (Huaulmé et al., 2023), released
  > under the EndoVis 2021 challenge for non-commercial research."

## SimSurgSkill 2021 — MICCAI EndoVis Sub-Challenge

- **License:** Non-commercial research only
- **Required citation:**

  > Funke, I., Mees, S.T., Weitz, J., Speidel, S., et al. *Objective
  > Surgical Skills Assessment and Tool Localization: Results from the
  > MICCAI 2021 SimSurgSkill Challenge.* arXiv preprint
  > arXiv:2212.04448, 2022.

- **Public bucket:** `gs://isi-simsurgskill-2021/`
- **Attribution string:**

  > "This work uses the SimSurgSkill 2021 dataset, released as part of
  > the MICCAI 2021 EndoVis sub-challenge for non-commercial research."

## YouTube harvest

- **License:** mixed, per video. YouTube Terms of Service plus the
  individual uploader's stated license (most are All Rights Reserved;
  some are CC BY).
- **Attribution requirement:** keep `harvest_log.jsonl` and
  `harvest_targets.csv` as the source of truth for every clip's URL,
  uploader, and stated license. When publishing, reproduce a sample of
  these in an appendix and link the original videos.
- ⚠️ **Commercial deployment of YouTube-derived training data needs a
  separate legal review** — independent of any of the research
  datasets above. See the legal note at the bottom of this file.

---

## Summary table for paper / model-card boilerplate

| Dataset | License | Citation key | OK for commercial product? |
|---|---|---|---|
| LASANA | CC BY 4.0 | Carstens 2026 | ✅ Yes (with attribution) |
| JIGSAWS | Academic only | Gao 2014 + Ahmidi 2017 | ❌ No |
| PETRAW | Non-commercial research | Huaulmé 2023 | ❌ No |
| SimSurgSkill | Non-commercial research | Funke 2022 | ❌ No |
| YouTube harvest | Mixed | per-video | ⚠️ Legal review required |

## Legal note

This file is **not** legal advice. The license summary above reflects
the public license text for each dataset as of 2026-04-08. Before
shipping any commercial product derived from this work, have a lawyer
review the actual DUA you signed plus the YouTube TOS. The "research
checkpoint" trained on the union of all five sources and the
"commercial checkpoint" trained only on LASANA + cleared YouTube data
should remain separate model artifacts with separate model cards.
