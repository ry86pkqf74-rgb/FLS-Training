#!/usr/bin/env bash
# sync_to_gpu.sh — Push training data to a RunPod/GPU node via S1 (217.77.2.114) jump host
# Installed at /data/fls/sync_to_gpu.sh on the storage node (207.244.235.10)
#
# Usage: ./sync_to_gpu.sh <GPU_HOST> [GPU_PORT] [--dry-run]
#   GPU_HOST  — IP of the GPU node (e.g. RunPod pod IP)
#   GPU_PORT  — SSH port on GPU (default 22)
#
# Environment overrides:
#   SSH_KEY        — private key path (default /root/.ssh/id_ed25519)
#   GPU_USER       — remote user (default root)
#   GPU_DEST_BASE  — destination root on GPU (default /workspace/FLS-Training)
set -euo pipefail

GPU_HOST="${1:-}"
GPU_PORT="${2:-22}"
[[ -z "${GPU_HOST}" ]] && { echo "Usage: $0 <GPU_HOST> [GPU_PORT] [--dry-run]"; exit 1; }

S1_HOST="217.77.2.114"
S1_USER="root"
SSH_KEY="${SSH_KEY:-/root/.ssh/id_ed25519}"
GPU_USER="${GPU_USER:-root}"
GPU_DEST_BASE="${GPU_DEST_BASE:-/workspace/FLS-Training}"
LOCAL_BASE="/data/fls"
LOG_FILE="${LOCAL_BASE}/logs/sync_to_gpu_$(date +%Y%m%d_%H%M%S).log"

RSYNC_OPTS="-avz --partial --progress"
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && RSYNC_OPTS="${RSYNC_OPTS} --dry-run"; done

mkdir -p "${LOCAL_BASE}/logs"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date -Iseconds)] Pushing training data to GPU ${GPU_HOST}:${GPU_PORT} via jump ${S1_HOST}"

# Route: storage node -> S1 (ProxyJump) -> GPU
SSH_CMD="ssh -i ${SSH_KEY} \
  -o StrictHostKeyChecking=no \
  -o ConnectTimeout=20 \
  -o ProxyJump=${S1_USER}@${S1_HOST} \
  -p ${GPU_PORT}"

for DIR in datasets-jsonl frames; do
    echo "--- Pushing ${DIR} -> ${GPU_DEST_BASE}/data/${DIR} ---"
    rsync ${RSYNC_OPTS} \
        -e "${SSH_CMD}" \
        "${LOCAL_BASE}/${DIR}/" \
        "${GPU_USER}@${GPU_HOST}:${GPU_DEST_BASE}/data/${DIR}/"
done

echo "[$(date -Iseconds)] Sync to GPU complete."
