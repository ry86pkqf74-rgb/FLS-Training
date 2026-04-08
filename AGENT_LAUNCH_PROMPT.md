RunPod Launch Prompt - FLS-Training v2

Copy-paste this whole document as the prompt for the RunPod-side agent (Cursor / VSC / Claude Code) once the pod is up. It encodes the full prelaunch state, the run config, the success criteria, and the post-run handoff. Nothing in here should require the agent to guess.

Context

You are launching the current supported v2 student-model training run for the FLS-Training project (ry86pkqf74-rgb/FLS-Training). The repo is a teacher-critique-student architecture for FLS Task 5 (intracorporeal suture) scoring. Your job is to clone, prepare the v2 dataset layout, train on RunPod with the `finetune_vlm` path, evaluate against the held-out trainee gate, and push the artifacts back to GitHub. Then stop the pod.

A prelaunch fix commit (fix(data): rglob memory, dedupe by video, stratified val split, tighter trainer defaults) has already landed on main. It fixed five real bugs in the data and training pipeline, materialized training/data/v2/, and tightened the trainer defaults. You must train against --ver v2, not v1. v1 was built from a broken pipeline that silently dropped 65 trainee video scores and never filtered the superseded V18 record.

Pod requirements

Hardware: single A100 80GB (RunPod Community Cloud is fine). Do not request a multi-GPU pod. Qwen2.5-VL-7B + LoRA fits comfortably on the supported single-GPU path; multi-GPU adds overhead with no benefit at this corpus size.
Image: RunPod's PyTorch 2.x CUDA 12.x template (e.g. runpod/pytorch:2.1.0-py3.10-cuda12.1.1).
Disk: >=30 GB persistent volume.
Network: allow outbound to huggingface.co, github.com, wandb.ai (if you use W&B).

Step-by-step launch

1. Clone and install

```bash
cd /workspace
git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git
cd FLS-Training
git log -1 --oneline   # MUST show the "fix(data): rglob memory..." commit at HEAD
pip install -e '.[training]'
```

If .[training] isn't defined in pyproject.toml, fall back to:

```bash
pip install -e .
pip install torch transformers peft datasets accelerate bitsandbytes wandb rich
```

2. Sanity check the data and the GPU

```bash
bash scripts/runpod_setup.sh
```

This script verifies CUDA + VRAM and counts files in training/data/. It should print at minimum:

- A100 detected with >=40 GB VRAM
- Training data files for v2 present (scoring_train_v2.jsonl, scoring_val_v2.jsonl, coaching_train_v2.jsonl, coaching_val_v2.jsonl)

Then verify the v2 corpus is what we expect:

```bash
PYTHONPATH=. python scripts/090_status.py
cat training/data/meta_v2.json
```

You should see, in meta_v2.json:

- raw_score_count: 192 (not 128 - if it says 128, the rglob fix did not land; abort and investigate)
- after_video_dedupe: 66
- scoring_train_count: 56, scoring_val_count: 10
- seed: 42
- split_strategy: "stratified_by_video_id"
- val_video_ids should include V13_video, V22_video, V31_video (real trainee held-outs)

If any of these are wrong, stop. Re-run `python scripts/040_prepare_training_data.py --ver v2 --seed 42` and inspect the output before training.

3. Prepare the trainer-compatible v2 directory

```bash
bash scripts/045_prep_v2_training.sh
```

4. Launch training

```bash
python -m src.training.finetune_vlm --config src/configs/finetune_task5_v2.yaml
```

The supported v2 config already encodes the current conservative settings and output path.

Expected behavior during training:

- Qwen2.5-VL-7B loads from the base model specified in `src/configs/finetune_task5_v2.yaml`.
- The v2 config should start fresh from the base model, not resume from the overfit v1 checkpoint.
- Eval is step-based in the v2 config so val loss is available during training.
- If you are using watchdog resume mode, make sure it resumes only within the v2 run directory and not from any v1-era checkpoint tree.

5. Generate held-out predictions

```bash
python scripts/055_generate_predictions.py \
   --model memory/model_checkpoints/v2_diverse/merged_16bit \
   --data training/data/scoring_val_v2.jsonl \
   --output memory/predictions/v2_on_val
```

6. Evaluate against the Phase 3 gate

```bash
python scripts/060_evaluate_student.py --student-scores memory/predictions/v2_on_val
```

The Phase 3 gate from `docs/EXECUTION_PLAN.md` is MAE <= 12 FLS points on a held-out trainee. The val set (`val_video_ids` in `meta_v2.json`) contains three real trainee videos: `V13_video`, `V22_video`, `V31_video`. The eval script should report aggregate val MAE plus per-video MAE.

Decision matrix:

- Aggregate val MAE <= 12 AND each held-out trainee video MAE <= 15 -> promote. Proceed to step 5.
- Aggregate <= 12 but one trainee video > 20 -> likely overfitting / leakage on that trainee. Do NOT promote. Capture the per-video numbers, file an issue, and stop.
- Aggregate > 12 -> underfit. Try one rerun with `--epochs 5 --lr 5e-5`. If still failing, the corpus is the problem (66 videos may simply be insufficient); report and stop.

7. Commit and push

```bash
git config user.email "<your email>"
git config user.name "<your name>"
git add memory/model_checkpoints/v2_diverse/ memory/predictions/v2_on_val/ training/runs/ src/configs/finetune_task5_v2.yaml
git commit -m "feat: student model v2 trained on 56-video corpus

Run config: epochs=3 lr=1e-4 batch=2 grad_accum=8 lora_r=16 seed=42
Val MAE (held-out trainee): <FILL IN from 060 output>
Per-video val MAE: V13=<...> V22=<...> V31=<...>"
git push origin main
```

8. Stop the pod

```bash
runpodctl stop pod $RUNPOD_POD_ID    # or use the web UI
```

Do not skip this. Idle A100 time is the largest cost in this project.

What "ready" looks like for the next training cycle

After this run, the user will start collecting more trainee videos and YouTube harvest. Cycle N+1 should:

- Re-run `scripts/040_prepare_training_data.py --ver vN+1 --seed 42` locally
- Re-run drift check (`scripts/075_check_drift.py`) against the new student
- Only train again if drift > threshold OR >=20 new ACCEPTED videos have landed since v2
- Use the same `--ver vN+1 --epochs 3 --lr 1e-4 --batch-size 2 --grad-accum 8 --seed 42` invocation

Things you should NOT do

- Do not run `--ver v1`. v1 is from the broken pipeline.
- Do not raise `--epochs` above 5 without a sweep. With 56 train examples you will overfit.
- Do not request multi-GPU. Florence-2-large LoRA does not need it; you'll pay 5x for nothing.
- Do not delete `memory/scores/2026-04-07/V18_video_claude-sonnet-4_20260407_140000.json`. It is the superseded V18 record; it's retained intentionally as audit history. The data pipeline filters it via the `superseded: true` flag.
- Do not push without W&B URLs (or no-wandb confirmation) in the commit body. Future you will want to find the run.
- Do not leave the pod running after git push.

Known caveats / TODOs (don't fix as part of this run)

- Coaching dataset is small (27 train / 7 val) because feedback only exists for the YouTube subset. Trainee video coaching data is the main backlog item - flag this in the post-run summary if val MAE on coaching looks weak.
- `MemoryStore.get_stats()` now reports active vs superseded; the old DuckDB-based stats query in `_load_from_disk` is fine as-is because it now skips superseded too.
- `pyproject.toml` should pin `transformers`, `peft`, `accelerate`, `torch` versions before the next run. Not blocking for v2 but will bite you in 6 weeks.

Success summary template

When you're done, post this back:

```text
FLS-Training v2 student trained ✓
   Run name:           <run_name>
   W&B URL:            <link or "no-wandb">
   Wall clock:         <NN min>  (scoring <NN>m + coaching <NN>m)
   Train loss:         scoring=<x.xxx>  coaching=<x.xxx>
   Val loss:           scoring=<x.xxx>  coaching=<x.xxx>
   Val MAE (held-out trainee, scoring head):
      aggregate:        <NN.N> FLS pts
      V13_video:        <NN.N>
      V22_video:        <NN.N>
      V31_video:        <NN.N>
   Phase 3 gate (<=12 MAE):  PASS / FAIL
   Promoted to production:  YES / NO
   Pod stopped:             YES
   Commit:                  <sha>

   API key for RunPod and SSH public key available locally
```
