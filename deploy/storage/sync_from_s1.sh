#!/usr/bin/env bash
# sync_from_s1.sh — Pull scored data and datasets from S1 (217.77.2.114)
# Installed at /data/fls/sync_from_s1.sh on the storage node (207.244.235.10)
#
# Usage: ./sync_from_s1.sh [--dry-run]
#
# Environment overrides:
#   SSH_KEY   — path to private key (default /root/.ssh/id_ed25519)
set -euo pipefail

S1_HOST="217.77.2.114"
S1_USER="root"
S1_BASE="/data/fls"
LOCAL_BASE="/data/fls"
SSH_KEY="${SSH_KEY:-/root/.ssh/id_ed25519}"
RSYNC_OPTS="-avz --partial --progress"
LOG_FILE="${LOCAL_BASE}/logs/sync_from_s1_$(date +%Y%m%d_%H%M%S).log"

[[ "${1:-}" == "--dry-run" ]] && RSYNC_OPTS="${RSYNC_OPTS} --dry-run"

mkdir -p "${LOCAL_BASE}/logs"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date -Iseconds)] Starting sync FROM S1 (${S1_HOST})"

for DIR in scored datasets-jsonl; do
    echo "--- Syncing ${DIR} ---"
    rsync ${RSYNC_OPTS} \
        -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
        "${S1_USER}@${S1_HOST}:${S1_BASE}/${DIR}/" \
        "${LOCAL_BASE}/${DIR}/"
done

echo "[$(date -Iseconds)] Sync from S1 complete."
