#!/usr/bin/env bash
# FLS Round 2 runbook — execute AFTER Hetzner batch 2 shows [432/432].
# Runs from the user's Mac (/Users/ros/Downloads/FLS-Training). Routes everything
# through ssh/scp since the Cowork sandbox has no outbound network.
set -euo pipefail

# Required env vars (populate before running):
#   HETZNER_PASS   — Hetzner S5 root password
#   GH_TOKEN       — GitHub PAT (decode from: security find-generic-password -s 'gh:github.com' -w)
#   RUNPOD_KEY     — RunPod API key (from ~/.zsh_history: grep RUNPOD_API_KEY)
: "${HETZNER_PASS:?set HETZNER_PASS}"
: "${GH_TOKEN:?set GH_TOKEN}"
: "${RUNPOD_KEY:?set RUNPOD_KEY}"

HETZNER="root@77.42.85.109"
GH_USER="ry86pkqf74-rgb"
POD_ID="n4w9mpzc82mry8"
# RunPod SSH endpoint — confirm port after resume (may change on restart):
RUNPOD_HOST="root@216.243.220.224"
RUNPOD_PORT=14152

ssh_h() { sshpass -p "$HETZNER_PASS" ssh -o StrictHostKeyChecking=no "$HETZNER" "$@"; }
scp_h() { sshpass -p "$HETZNER_PASS" scp -o StrictHostKeyChecking=no "$@"; }
ssh_r() { ssh -o StrictHostKeyChecking=no -p "$RUNPOD_PORT" "$RUNPOD_HOST" "$@"; }
scp_r() { scp -o StrictHostKeyChecking=no -P "$RUNPOD_PORT" "$@"; }

echo "=== [1/7] Verify batch 2 complete ==="
ssh_h "tmux capture-pane -t fls -pS -4 | tail -4 | grep -E '^\\[' || true; find /opt/fls-training/memory/scores/2026-04-13 -name '*.json' | wc -l"
read -p "Batch 2 complete? (y/N) " ok; [[ "$ok" == "y" ]] || exit 1

echo "=== [2/7] Push scores from Hetzner -> GitHub ==="
ssh_h "cd /opt/fls-training && git add memory/scores/2026-04-13/ && \
  git -c user.email=fls-scorer@researchflow.ai -c user.name='FLS Scorer' \
  commit -m 'data: batch 2 complete (432/432)' && \
  git push https://${GH_USER}:${GH_TOKEN}@github.com/${GH_USER}/FLS-Training.git main"

echo "=== [3/7] Re-run SFT prep ==="
ssh_h "cd /opt/fls-training && python3 scripts/030_prep_sft_data.py"
ssh_h "cd /opt/fls-training && wc -l data/training/youtube_sft_v1/{train,val}.jsonl && \
  git add data/training/youtube_sft_v1/ data/training/youtube_sft_v1_manifest.json && \
  git -c user.email=fls-scorer@researchflow.ai -c user.name='FLS Scorer' \
  commit -m 'data: youtube_sft_v1 full batch' && \
  git push https://${GH_USER}:${GH_TOKEN}@github.com/${GH_USER}/FLS-Training.git main"

echo "=== [4/7] Resume RunPod pod ==="
curl -s -H "Authorization: Bearer $RUNPOD_KEY" -H "Content-Type: application/json" \
  -X POST "https://api.runpod.io/graphql" \
  -d "{\"query\":\"mutation { podResume(input: {podId: \\\"$POD_ID\\\", gpuCount: 1}) { id desiredStatus } }\"}"
echo
echo "Waiting 60s for SSH to come up... Confirm SSH port in RunPod UI (may differ from $RUNPOD_PORT)."
sleep 60

echo "=== [5/7] Stage data + adapter on RunPod ==="
TMP=$(mktemp -d)
scp_h "${HETZNER}:/opt/fls-training/data/training/2026-04-09_lasana_v1/train.jsonl" "$TMP/lasana_train.jsonl"
scp_h "${HETZNER}:/opt/fls-training/data/training/2026-04-09_lasana_v1/val.jsonl"   "$TMP/lasana_val.jsonl"
scp_h "${HETZNER}:/opt/fls-training/data/training/youtube_sft_v1/train.jsonl"       "$TMP/yt_train.jsonl"
scp_h "${HETZNER}:/opt/fls-training/data/training/youtube_sft_v1/val.jsonl"         "$TMP/yt_val.jsonl"
scp_r "$TMP/lasana_train.jsonl" "$TMP/lasana_val.jsonl" "$TMP/yt_train.jsonl" "$TMP/yt_val.jsonl" "${RUNPOD_HOST}:/workspace/"

# Copy the round 1 adapter into /workspace/adapter_init for warm-start
ssh_r "mkdir -p /workspace/adapter_init && cp /workspace/checkpoints/final/adapter_model.safetensors \
  /workspace/checkpoints/final/adapter_config.json /workspace/adapter_init/ && ls -la /workspace/adapter_init/"

# Copy the training script
scp_r scripts/033_train_lora_round2.py "${RUNPOD_HOST}:/workspace/"

echo "=== [6/7] Launch training in tmux on RunPod ==="
ssh_r "tmux new-session -d -s train 'cd /workspace && python3 033_train_lora_round2.py 2>&1 | tee /workspace/round2_train.log'"
echo "Attach: ssh -p $RUNPOD_PORT $RUNPOD_HOST -t 'tmux attach -t train'"
echo "Tail:   ssh -p $RUNPOD_PORT $RUNPOD_HOST 'tail -f /workspace/round2_train.log'"

echo "=== [7/7] After training finishes: eval + push + stop pod ==="
cat <<'POST'
# Eval (from user Mac):
ssh -p $RUNPOD_PORT $RUNPOD_HOST \
  "cd /workspace && python3 /workspace/FLS-Training/scripts/032_eval_adapter.py \
     --adapter /workspace/checkpoints_r2/final \
     --test data/training/2026-04-09_lasana_v1/test.jsonl \
     --out /workspace/checkpoints_r2/eval_results.json"

# Push adapter to GitHub from RunPod:
ssh -p $RUNPOD_PORT $RUNPOD_HOST "cd /workspace/FLS-Training && \
  mkdir -p memory/model_checkpoints/lasana_plus_youtube_v1 && \
  cp /workspace/checkpoints_r2/final/adapter_model.safetensors \
     /workspace/checkpoints_r2/final/adapter_config.json \
     /workspace/checkpoints_r2/run_manifest.json \
     /workspace/checkpoints_r2/eval_results.json \
     memory/model_checkpoints/lasana_plus_youtube_v1/ && \
  git add -f memory/model_checkpoints/lasana_plus_youtube_v1/ && \
  git -c user.email=fls@researchflow.ai -c user.name='FLS Trainer' \
    commit -m 'model: round 2 adapter (LASANA + YouTube v1)' && \
  git push https://\${GH_USER}:\${GH_TOKEN}@github.com/ry86pkqf74-rgb/FLS-Training.git main"

# Stop pod:
curl -s -H "Authorization: Bearer $RUNPOD_KEY" -H "Content-Type: application/json" \
  -X POST "https://api.runpod.io/graphql" \
  -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}"
POST
