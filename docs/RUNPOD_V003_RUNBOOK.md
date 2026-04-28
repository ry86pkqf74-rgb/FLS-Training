# RunPod v003 Training Runbook

End-to-end commands for the v003 training iteration: v17 multimodal scoring
LoRA followed by the report-generation LoRA, both on a fresh RunPod pod.

This runbook assumes the `v003-scoring-report` branch has been pushed to
GitHub and contains:

- `src/rubrics/loader.py`
- `src/reporting/{report_v3,readiness,task_templates,render_markdown_v3,validator}.py`
- `src/training/v003_target.py`
- `scripts/030c_prep_v003_multimodal.py`
- `scripts/060_generate_report_v3_labels.py`
- `scripts/061_validate_report_v3_labels.py`
- `scripts/062_prepare_lora_report_v3_dataset.py`
- `scripts/063_train_report_lora_v3.py`
- `scripts/070_train_qwen_vl_v17_v003.py`

Run every step inside `tmux` so you can detach safely.

---

## 0. Pre-flight (local laptop)

```bash
cd ~/FLS_Training/FLS-Training
git checkout v003-scoring-report
git pull
source .venv/bin/activate
python -m pytest tests/test_report_v3_*.py tests/test_frontend_task_selection.py tests/test_frontier_scorer.py -q
# Expected: 17 passed
```

If any of those fail, do not launch the pod.

---

## 1. Spin up the pod (RunPod web UI)

| Setting           | Value                                             |
|-------------------|---------------------------------------------------|
| Template          | RunPod PyTorch 2.4 (CUDA 12.4) or newer           |
| GPU               | **H100 SXM 80 GB** (preferred) or H200 141 GB     |
| CPU               | 16+ vCPU                                          |
| RAM               | 100+ GB                                           |
| Container disk    | 80 GB                                             |
| Persistent volume | 200 GB mounted at `/workspace`                    |
| Expose            | SSH (port 22), Jupyter (8888) optional            |
| Region            | Pick the cheapest region with H100 SXM in stock   |

After it boots, copy the SSH command from the RunPod UI and connect:

```bash
ssh root@<pod_ip> -p <port> -i ~/.ssh/<your_key>
```

---

## 2. Bootstrap the pod

```bash
tmux new -s fls
cd /workspace
git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git
cd FLS-Training
git checkout v003-scoring-report

# System deps + Python deps
bash scripts/runpod_setup.sh
# If runpod_setup.sh is missing on this branch, fall back to:
# pip install --break-system-packages -e ".[training]"
# pip install --break-system-packages \
#   "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Sanity: GPU + key Python imports
nvidia-smi
python -c "import torch, transformers, peft; print(torch.cuda.is_available(), transformers.__version__, peft.__version__)"
```

Set the API keys you'll need (only required if you intend to (re)score videos
or push artifacts back to GitHub):

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
export OPENAI_API_KEY='sk-...'
export GITHUB_TOKEN='github_pat_...'
```

---

## 3. Sync data onto the pod

You need three directories on `/workspace`:

| Dir                          | Contents                                                                |
|------------------------------|-------------------------------------------------------------------------|
| `/workspace/v003_multimodal` | `train.jsonl` / `val.jsonl` / `test.jsonl` from `030c_prep_v003_multimodal.py` |
| `/workspace/v003_frames`     | extracted JPEGs `<video_id>/frame_<NNN>.jpg`                            |
| `/workspace/v16_lora_output/final_adapter` | the existing v16 adapter to resume from                  |

### 3a. Build v003 multimodal JSONL

If `data/training/youtube_sft_v1/{train,val,test}.jsonl` is already in the
repo (it is, on this branch), run the v003 enrichment in place:

```bash
python scripts/030c_prep_v003_multimodal.py
mkdir -p /workspace/v003_multimodal
cp data/training/youtube_sft_v003/{train,val,test}.jsonl /workspace/v003_multimodal/
```

For a richer dataset (full memory/scores corpus) use the integrated path:

```bash
python scripts/040_prepare_training_data.py \
    --ver v003 \
    --frames-dir /workspace/v003_frames \
    --max-frames 24 \
    --group-by trainee \
    --min-confidence 0.5 \
    --include-coach-feedback \
    --v003-target-schema
```

That writes `data/training/<DATE>_v003/{train,val,test}.jsonl`. Move them
into `/workspace/v003_multimodal/`.

### 3b. Frames

Frames are not in the GitHub repo. Either:

- Pull them from B2: `bash scripts/067_b2_sync.sh /workspace/v003_frames`
- Or rsync from a previous pod / Hetzner box

Verify:

```bash
find /workspace/v003_frames -name '*.jpg' | head
find /workspace/v003_frames -name '*.jpg' | wc -l
```

### 3c. v16 adapter

```bash
mkdir -p /workspace/v16_lora_output/final_adapter
# Pull from B2 (preferred):
b2 sync b2://fls-adapters/v16/final_adapter /workspace/v16_lora_output/final_adapter
# OR copy via scp from your laptop:
# scp -P <port> -r /opt/fls/adapters/v16/* root@<pod_ip>:/workspace/v16_lora_output/final_adapter/
```

If the v16 adapter is unavailable, train v17 with `--no-resume` (slower, but
deterministic).

---

## 4. Train v17 multimodal scoring LoRA

```bash
cd /workspace/FLS-Training
python scripts/070_train_qwen_vl_v17_v003.py \
    --data-dir /workspace/v003_multimodal \
    --frames-dir /workspace/v003_frames \
    --base-adapter /workspace/v16_lora_output/final_adapter \
    --output-dir /workspace/v17_lora_output \
    --epochs 2 \
    --batch-size 1 \
    --grad-accum 16 \
    --learning-rate 5e-5 \
    2>&1 | tee /workspace/v17_train.log
```

Expected wall time: 1.5–4 h on H100 SXM, depending on dataset size.

Watch the loss with:

```bash
tail -f /workspace/v17_train.log
# in another shell:
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5
```

**Stop conditions:**

- `eval_loss` plateaus or rises for 3 consecutive eval steps → kill, take last best checkpoint.
- OOM → reduce `--grad-accum` to 8 (effective batch becomes 8) or `IMAGE_MAX_PIXELS` in `scripts/059_train_qwen_vl_v16_multimodal.py`.

When done you have `/workspace/v17_lora_output/final_adapter/`.

---

## 5. Generate + validate v003 report labels

These run on CPU and take a few minutes:

```bash
python scripts/060_generate_report_v3_labels.py \
    memory/scores \
    /workspace/v003_report_labels

python scripts/061_validate_report_v3_labels.py /workspace/v003_report_labels
# Expected: "All v003 labels passed validation."
# If failures: read JSON output, fix scoring inputs, re-run.

python scripts/062_prepare_lora_report_v3_dataset.py \
    /workspace/v003_report_labels \
    /workspace/v003_report_lora/all.jsonl

# 90/10 split for the report LoRA
mkdir -p /workspace/v003_report_lora
python - <<'PY'
import json, random
from pathlib import Path
src = Path("/workspace/v003_report_lora/all.jsonl")
rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
random.Random(42).shuffle(rows)
cut = max(1, len(rows) // 10)
val, train = rows[:cut], rows[cut:]
Path("/workspace/v003_report_lora/train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
Path("/workspace/v003_report_lora/val.jsonl").write_text("\n".join(json.dumps(r) for r in val) + "\n")
print(f"train={len(train)} val={len(val)}")
PY
```

Spot-check a handful:

```bash
shuf -n 3 /workspace/v003_report_lora/train.jsonl | python -m json.tool | less
```

You're looking for: `formula_applied` matches `max - time - penalties`, no
`"proficient"` when `critical_findings` is non-empty, z-scores are inside
`experimental_metrics` only.

---

## 6. Train report LoRA v003

```bash
python scripts/063_train_report_lora_v3.py \
    --dataset-jsonl /workspace/v003_report_lora/train.jsonl \
    --val-jsonl /workspace/v003_report_lora/val.jsonl \
    --output-dir /workspace/report_lora_v003 \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --learning-rate 1e-4 \
    --human-reviewed \
    2>&1 | tee /workspace/report_lora_train.log
```

Expected wall time: 30–60 min on H100. The `--human-reviewed` flag is a
deliberate gate — only pass it after step 5's spot-check.

Output: `/workspace/report_lora_v003/final_adapter/`.

---

## 7. Quick eval on the held-out set

```bash
# v17 multimodal eval (loss-only smoke test):
python scripts/060_evaluate_student_v2.py \
    --adapter /workspace/v17_lora_output/final_adapter \
    --data /workspace/v003_multimodal/test.jsonl \
    --frames-dir /workspace/v003_frames \
    --output /workspace/v17_eval.json 2>&1 | tee /workspace/v17_eval.log

# Report LoRA eval (text-only):
python - <<'PY'
import json
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter = "/workspace/report_lora_v003/final_adapter"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)

sample = json.loads(Path("/workspace/v003_report_lora/val.jsonl").read_text().splitlines()[0])
prompt = tok.apply_chat_template(sample["messages"][:-1], tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device), max_new_tokens=1500)
print(tok.decode(out[0], skip_special_tokens=True))
PY
```

---

## 8. Persist artifacts

```bash
# Push adapters back to B2
b2 sync /workspace/v17_lora_output/final_adapter   b2://fls-adapters/v17_v003/final_adapter
b2 sync /workspace/report_lora_v003/final_adapter  b2://fls-adapters/report_v003/final_adapter

# Push training metrics back to GitHub for the run record
cd /workspace/FLS-Training
mkdir -p memory/training_runs/v003
cp /workspace/v17_lora_output/final_adapter/training_metrics.json   memory/training_runs/v003/v17_metrics.json
cp /workspace/report_lora_v003/final_adapter/training_metrics.json  memory/training_runs/v003/report_lora_metrics.json
cp /workspace/v17_train.log         memory/training_runs/v003/v17_train.log
cp /workspace/report_lora_train.log memory/training_runs/v003/report_lora_train.log
git add memory/training_runs/v003
git commit -m "training: v003 v17 + report LoRA metrics from RunPod"
git push origin v003-scoring-report
```

---

## 9. Tear down

Confirm artifacts are off the pod (B2 + GitHub) before terminating. Then:

- RunPod web UI → Stop → Terminate.
- Persistent volume can be kept (cheap) if you expect to iterate again
  within a week; otherwise delete it.

---

## Troubleshooting

| Symptom                                  | Fix                                                                                        |
|------------------------------------------|--------------------------------------------------------------------------------------------|
| `OOM` mid-train                          | `--grad-accum 8` (cuts effective batch in half) or lower `IMAGE_MAX_PIXELS`.               |
| `Unsloth` won't import                   | Fall back to `framework: hf_trainer` — already what scripts/070 + scripts/063 use.         |
| Frames not found                         | `python -c "from src.training.prepare_dataset import _sample_frame_paths; ..."` to debug.  |
| `eval_loss` immediately spikes           | LR too high for resume → drop `--learning-rate` to `2e-5`.                                  |
| Validator reports formula mismatches     | Original score had bad math; rerun `recompute_score_from_components` over `memory/scores`. |
