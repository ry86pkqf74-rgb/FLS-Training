#!/usr/bin/env bash
# scripts/095_contabo_sync.sh — Sync large artifacts to/from Contabo S8
#
# Contabo S8 is the durable storage tier for checkpoints, frames, and
# raw videos until the corpus exceeds 500 videos (then graduate to B2/R2).
#
# Server: 207.244.235.10 (m66910.contaboserver.net)
#   - 24 vCPU, 61 GiB RAM, ~838 GiB disk
#   - Ubuntu 22.04, always-on ($99/mo)
#   - SSH: ssh s8-other-project  (see ~/.ssh/config)
#
# Remote layout:
#   /srv/fls-training/
#     checkpoints/    ← model checkpoints from training runs
#     frames/         ← extracted video frames
#     videos/         ← raw harvested videos
#     logs/           ← large training logs
#     run_manifests/  ← copy of memory/training_runs/ for quick lookup
#
# Usage:
#   bash scripts/095_contabo_sync.sh push-checkpoints
#   bash scripts/095_contabo_sync.sh push-frames
#   bash scripts/095_contabo_sync.sh push-videos <local_video_dir>
#   bash scripts/095_contabo_sync.sh pull-checkpoints [run_id]
#   bash scripts/095_contabo_sync.sh pull-frames
#   bash scripts/095_contabo_sync.sh status
#   bash scripts/095_contabo_sync.sh init          # first-time remote setup

set -euo pipefail

REMOTE_HOST="${CONTABO_HOST:-s8-other-project}"
REMOTE_BASE="${CONTABO_BASE:-/srv/fls-training}"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  init                 Create remote directory layout"
    echo "  push-checkpoints     Sync memory/model_checkpoints/ → remote"
    echo "  push-frames          Sync data/frames/ → remote"
    echo "  push-videos <dir>    Sync a local video directory → remote"
    echo "  push-manifests       Sync memory/training_runs/ → remote"
    echo "  pull-checkpoints [x] Pull checkpoints (optionally filter by run_id prefix)"
    echo "  pull-frames          Pull frames from remote"
    echo "  status               Show remote disk usage"
    exit 1
}

[ $# -ge 1 ] || usage

CMD="$1"
shift

case "$CMD" in
    init)
        echo "Creating remote directory layout on $REMOTE_HOST..."
        ssh "$REMOTE_HOST" "mkdir -p $REMOTE_BASE/{checkpoints,frames,videos,logs,run_manifests}"
        echo "Done. Remote layout:"
        ssh "$REMOTE_HOST" "ls -la $REMOTE_BASE/"
        ;;

    push-checkpoints)
        echo "Syncing checkpoints → $REMOTE_HOST..."
        rsync -avz --progress \
            --include='*/' \
            --include='*.json' \
            --include='*.safetensors' \
            --include='*.bin' \
            --include='*.yaml' \
            --include='*.md' \
            --exclude='*.pyc' \
            "$LOCAL_REPO/memory/model_checkpoints/" \
            "$REMOTE_HOST:$REMOTE_BASE/checkpoints/"
        echo "Done."
        ;;

    push-frames)
        SRC="${LOCAL_REPO}/data/frames"
        if [ ! -d "$SRC" ]; then
            SRC="${LOCAL_REPO}/memory/frames"
        fi
        if [ ! -d "$SRC" ]; then
            echo "No frames directory found at data/frames/ or memory/frames/"
            exit 1
        fi
        echo "Syncing frames from $SRC → $REMOTE_HOST..."
        rsync -avz --progress "$SRC/" "$REMOTE_HOST:$REMOTE_BASE/frames/"
        echo "Done."
        ;;

    push-videos)
        VIDEO_DIR="${1:?Usage: $0 push-videos <local_video_dir>}"
        echo "Syncing videos from $VIDEO_DIR → $REMOTE_HOST..."
        rsync -avz --progress "$VIDEO_DIR/" "$REMOTE_HOST:$REMOTE_BASE/videos/"
        echo "Done."
        ;;

    push-manifests)
        echo "Syncing run manifests → $REMOTE_HOST..."
        rsync -avz --progress \
            "$LOCAL_REPO/memory/training_runs/" \
            "$REMOTE_HOST:$REMOTE_BASE/run_manifests/"
        echo "Done."
        ;;

    pull-checkpoints)
        FILTER="${1:-}"
        if [ -n "$FILTER" ]; then
            echo "Pulling checkpoints matching '$FILTER'..."
            rsync -avz --progress \
                --include="$FILTER*/" --include="$FILTER*/**" --exclude='*' \
                "$REMOTE_HOST:$REMOTE_BASE/checkpoints/" \
                "$LOCAL_REPO/memory/model_checkpoints/"
        else
            echo "Pulling all checkpoints..."
            rsync -avz --progress \
                "$REMOTE_HOST:$REMOTE_BASE/checkpoints/" \
                "$LOCAL_REPO/memory/model_checkpoints/"
        fi
        echo "Done."
        ;;

    pull-frames)
        echo "Pulling frames from $REMOTE_HOST..."
        mkdir -p "$LOCAL_REPO/data/frames"
        rsync -avz --progress \
            "$REMOTE_HOST:$REMOTE_BASE/frames/" \
            "$LOCAL_REPO/data/frames/"
        echo "Done."
        ;;

    status)
        echo "Remote storage status ($REMOTE_HOST:$REMOTE_BASE):"
        ssh "$REMOTE_HOST" "du -sh $REMOTE_BASE/*/ 2>/dev/null; echo '---'; df -h $REMOTE_BASE"
        ;;

    *)
        usage
        ;;
esac
