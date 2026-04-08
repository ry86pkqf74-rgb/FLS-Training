# Honest baseline MAE — 2026-04-08

Computed via `src.training.schema_adapter.normalize_score` on the full
`memory/scores/` tree (256 score files, latest per `(video_id, source)`
pair).

**This is the first time the number has been computed through the same
code path training will use**, so it is the number that the first
fine-tuned student run on v4 data has to beat to justify any further GPU
spend.

## Corpus

| metric | value |
| --- | --- |
| unique videos | 78 |
| consensus scores | 37 |
| teacher_claude scores | 78 |
| teacher_gpt4o scores | 49 |
| consensus score range | 0 – 600 |
| consensus score mean | 459.2 |
| consensus score stdev | 130.2 |

Only 37 of 78 videos have a consensus record; the other 41 have only
teacher scores. This alone is a reason to be skeptical of any "val MAE"
number quoted from a model trained on <50 supervised examples.

## Pairwise MAE

| pair | n | MAE |
| --- | ---: | ---: |
| Consensus vs Claude | 35 | **12.84** |
| Consensus vs GPT-4o | 16 | 25.83 |
| Claude vs GPT-4o | 48 | **21.61** |

## Interpretation

**Teacher-vs-teacher MAE is ≈21.6.** That is the ceiling on how tight any
student's MAE-against-consensus number can realistically be before we
are measuring inter-teacher noise rather than student quality.

**Consensus-vs-Claude MAE is ≈12.8.** That's because consensus is a
reconciliation of Claude + GPT, and Claude drives the reconciliation most
of the time when GPT is missing. It should not be interpreted as "Claude
is close to ground truth" — it is a self-similarity floor.

## Deployment-plan implications

1. The abort criterion in `docs/LASANA_DEPLOYMENT_PLAN.md` currently reads
   *"MAE > 12 after 2 runs = stop GPU spending, harvest more data."* That
   threshold was set blind. Revise to:
   - **MAE > 22** on the held-out faculty-rated gold set → abort, data
     problem, not a training problem.
   - **MAE 15–22** → training is working but inside teacher noise. One
     more run at most; do not scale.
   - **MAE < 15** → only then is further GPU spend justified.

2. We cannot train and grade on consensus alone — the n=37 corpus is too
   small and too noisy for a VLM fine-tune to produce a trustworthy
   generalization signal. The v4 run MUST wait until the lasana + petraw
   + simsurg records are ingested and (at minimum) teacher-scored so the
   held-out split has ≥100 videos with at least two teacher scores each.

3. The fact that `teacher_gpt4o` only covers 49 videos means half of our
   training records are effectively single-teacher. That inflates the
   "consensus" number for those rows to whatever Claude said. Before the
   next training run, re-score the 29 missing videos with GPT-4o.

## Reproducing this report

```bash
python3 -c "
import json, glob, statistics
from collections import defaultdict
from src.training.schema_adapter import normalize_score

files = glob.glob('memory/scores/**/*.json', recursive=True)
by_video = defaultdict(dict)
for f in files:
    try: d=json.load(open(f))
    except: continue
    n = normalize_score(d)
    src = n['source']
    if 'gpt' in src.lower(): src='teacher_gpt'
    elif 'claude' in src.lower(): src='teacher_claude'
    elif 'consensus' in src.lower(): src='consensus'
    else: continue
    prev = by_video[n['video_id']].get(src)
    if prev is None or n['raw'].get('scored_at','') > prev['raw'].get('scored_at',''):
        by_video[n['video_id']][src] = n

def mae(pairs): return round(sum(abs(a-b) for a,b in pairs)/max(1,len(pairs)),2)
for label, picks in [
    ('consensus/claude', ('consensus','teacher_claude')),
    ('consensus/gpt4o',  ('consensus','teacher_gpt')),
    ('claude/gpt4o',     ('teacher_claude','teacher_gpt')),
]:
    a,b = picks
    pairs = [(s[a]['total_fls_score'], s[b]['total_fls_score'])
             for s in by_video.values() if a in s and b in s]
    print(f'{label:20s} n={len(pairs):3d} MAE={mae(pairs)}')
"
```
