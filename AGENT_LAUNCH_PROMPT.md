# FLS-Training: Autonomous RunPod Training Launch

## Context

You are operating on the FLS-Training repo: https://github.com/ry86pkqf74-rgb/FLS-Training

This is an AI surgical skills training system for FLS Task 5 (intracorporeal suturing). It has a teacher-critique-student architecture where Claude + GPT-4o score videos as teachers, and we're fine-tuning a student model to take over scoring AND generate coaching feedback.

The repo already contains:
- 31 scored videos (V1-V31) with both Claude and GPT-4o scores in `memory/scores/`
- 31 consensus comparisons in `memory/comparisons/`
- Pre-built training data in `data/training/` (train/val/test JSONL)
- A fine-tuning config at `src/configs/finetune_task5_v1.yaml` targeting Qwen2.5-VL-7B-Instruct with Unsloth LoRA
- A Dockerfile at `deploy/Dockerfile.trainer`
- Launch scripts at `deploy/runpod_launch.sh` and `scripts/runpod_setup.sh`
- A launch guide at `deploy/LAUNCH_GUIDE.md`
- Dual training objectives: scoring accuracy (frames → FLS score JSON) AND coaching feedback (frames + history → FeedbackReport with progression insights, drill recommendations, fatigue detection)
- The feedback schema is at `src/feedback/schema.py` — the student must learn to produce FeedbackReport objects, not just scores

## Your Mission

Autonomously launch a full training cycle on RunPod. Execute every step — do not ask me to do anything manually. Here's what to do:

### Phase 1: Select GPU and Create Pod

Use the RunPod API or CLI to provision the BEST GPU for this workload:

**Optimal choice (ranked):**
1. **H200 SXM 141GB** — if available, ideal. Qwen2.5-VL-7B fits in full bf16 with room for large batch. ~$3-4/hr.
2. **H100 SXM 80GB** — excellent. Full bf16 LoRA fits. ~$2.69/hr on-demand, ~$1.99 community.
3. **A100 SXM 80GB** — very good. bf16 LoRA fits comfortably. ~$1.39/hr secure, ~$0.89 community.
4. **A100 PCIe 80GB** — good fallback. ~$1.19/hr.
5. **L40S 48GB** — minimum viable. May need 4-bit quantization. ~$0.79/hr.

Selection criteria: Pick the best available GPU that's actually in stock. Prefer community cloud for cost. The training run will take 30-90 minutes depending on GPU, so total cost should be $1.50-$5.00.

**Pod configuration:**
- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Container disk: 20GB
- Volume disk: 50GB (persistent, mount at /workspace)
- Expose HTTP port 8888 (Jupyter) and SSH port 22
- Set env var: `GITHUB_TOKEN=<your_token>` (for git push back)

If you don't have RunPod API access, generate the exact `runpodctl` CLI commands or the curl API calls I can paste. If you can access RunPod directly through a browser tool, do it.

### Phase 2: Setup on the Pod

Once the pod is running, SSH in and execute:

```bash
cd /workspace
git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git
cd FLS-Training
bash scripts/runpod_setup.sh
```

Then verify:
- GPU is detected and has sufficient VRAM
- Training data exists in `data/training/`
- The manifest.yaml confirms the dataset (train/val/test splits)
- All Python dependencies installed (torch, transformers, peft, unsloth, datasets, accelerate)

### Phase 3: Prepare Enhanced Training Data

The existing training data in `data/training/` may only cover scoring. We need BOTH heads:

1. **Check if coaching training data exists.** Look for files matching `coaching_*.jsonl` in `data/training/` or `training/data/`.

2. **If coaching data is missing, generate it:**
   ```bash
   python scripts/040_prepare_training_data.py --ver v2
   ```
   This should produce both `scoring_train_v2.jsonl` AND `coaching_train_v2.jsonl`.

3. **If the 040 script doesn't produce coaching data**, run the feedback generator across all scored videos first:
   ```bash
   python scripts/080_generate_feedback_report.py --video-id <each_video>
   ```
   Or write a quick batch script to iterate over all video IDs in `memory/scores/`.

4. **Verify the training data has both objectives:**
   - Scoring examples: input = frame descriptions → output = ScoringResult JSON
   - Coaching examples: input = frame descriptions + trainee history → output = FeedbackReport JSON
   - Minimum 25+ training examples per head

### Phase 4: Run Training

Execute the training using the repo's config:

```bash
# Option A: Use the existing launch script
bash deploy/runpod_launch.sh data/training/2026-04-07_v1 src/configs/finetune_task5_v1.yaml

# Option B: Use the Python trainer directly
python -m src.training.runpod_trainer

# Option C: If Unsloth is the configured backend, use its workflow
python src/training/finetune_vlm.py --config src/configs/finetune_task5_v1.yaml
```

Pick whichever script actually exists and works. Check `deploy/LAUNCH_GUIDE.md` for the recommended approach.

**Training config overrides if needed:**
- If VRAM < 80GB: set `quantization: "4bit"` in the config
- If VRAM >= 80GB: keep `quantization: "none"` for best quality
- LoRA r=16, alpha=32, dropout=0.05
- Learning rate: 2e-4 with cosine schedule
- Epochs: 3-5 (with 31 videos / ~30 examples, 5 epochs is fine)
- Batch size: 2 with gradient_accumulation=4 (effective batch 8)
- Enable gradient checkpointing
- Log to wandb if WANDB_API_KEY is set

**Monitor training:**
- Watch loss curves — training loss should decrease smoothly
- Val loss should decrease then flatten (stop if it increases for 2+ eval steps)
- Expected: ~20-60 min on H100/H200, ~60-90 min on A100

### Phase 5: Evaluate

After training completes:

```bash
python scripts/060_evaluate_student.py
# or
python scripts/050_evaluate.py
```

Check:
- Time estimation agreement: >85% within 10s of teacher
- FLS score agreement: >85% within 20 points of teacher
- Phase detection accuracy
- If coaching head was trained: verify feedback quality on 2-3 held-out videos

### Phase 6: Push Results to GitHub

This is critical — GitHub is the persistent brain. Push everything back:

```bash
cd /workspace/FLS-Training
git config user.email "logan.glosser@gmail.com"
git config user.name "FLS Training Agent"
git add models/ training/runs/ memory/ data/
git commit -m "training: student v1 — Qwen2.5-VL-7B scoring+coaching heads

- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
- Training time: $(cat training/runs/*/run_summary.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"{d.get(\"scoring\",{}).get(\"elapsed_seconds\",0)/60:.0f}min scoring + {d.get(\"coaching\",{}).get(\"elapsed_seconds\",0)/60:.0f}min coaching")' 2>/dev/null || echo 'see logs')
- Scoring agreement: $(cat training/runs/*/eval_*.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"{d.get(\"fls_agreement_pct\",\"?\")}")' 2>/dev/null || echo 'pending')%
- LoRA adapters saved to models/"

git push origin main
```

If git push fails (large files), use Git LFS:
```bash
git lfs install
git lfs track "models/**/*.safetensors" "models/**/*.bin"
git add .gitattributes
git add -A
git commit --amend --no-edit
git push origin main
```

If push still fails due to sandbox/size limits, use the GitHub Git Data API pattern (see `scripts/` for examples or the repo's existing push patterns).

### Phase 7: Shutdown

After confirming the push succeeded:
1. Verify on GitHub that `models/` contains the LoRA adapters
2. Verify `training/runs/` contains the run summary
3. Stop the RunPod pod (don't delete the volume — keep for next iteration)

## Key Constraints

- **Total budget: $5 max.** Pick community cloud GPUs. Training should take 30-90 min.
- **GitHub is the brain.** Every artifact (adapters, logs, eval results) MUST be pushed back.
- **Dual objectives are non-negotiable.** The student must learn BOTH scoring AND coaching. If you can only train one, train scoring first, then coaching as a second LoRA adapter on the same base.
- **Don't modify scored data.** The files in `memory/scores/` and `memory/comparisons/` are ground truth — read-only.
- **Use what exists.** The repo has deploy scripts, configs, and training code. Read them first. Only write new code if the existing scripts don't cover the need.

## Repo Structure Reference

```
FLS-Training/
├── data/training/           # Pre-built JSONL datasets
├── deploy/                  # Dockerfile, launch scripts, guide
├── memory/scores/           # 62 scored video JSONs (Claude+GPT)
├── memory/comparisons/      # 31 consensus comparisons
├── memory/feedback/         # Coaching reports (may need generating)
├── models/                  # LoRA adapters (output target)
├── prompts/                 # System prompts for scoring + coaching
├── rubrics/                 # FLS Task 5 rubric
├── scripts/                 # CLI scripts (010-090)
├── src/configs/             # finetune_task5_v1.yaml
├── src/feedback/schema.py   # FeedbackReport, PhaseCoaching, TraineeProfile
├── src/scoring/schema.py    # ScoringResult, VideoRecord
├── src/training/            # runpod_trainer.py, data_prep.py, evaluator.py
└── training/runs/           # Training run logs (output target)
```

## Start Now

Read `deploy/LAUNCH_GUIDE.md` and `src/configs/finetune_task5_v1.yaml` first. Then provision the GPU and execute. Report back with: GPU selected, training time, final loss, evaluation metrics, and the git commit SHA of the pushed results.
