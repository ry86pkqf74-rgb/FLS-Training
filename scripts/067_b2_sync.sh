#!/usr/bin/env bash
# scripts/067_b2_sync.sh — Backblaze B2 sync helper for FLS-Training
#
# Wraps the `b2` CLI for the four artifacts that move between pods:
#   1. LASANA frames (phase 1 → bucket)
#   2. LASANA features (phase 2 → bucket and bucket → phase 3)
#   3. Training data versions (laptop → bucket and bucket → phase 3)
#   4. Model checkpoints (phase 3 → bucket, ALWAYS run before pod shutdown)
#
# Prerequisites:
#   1. b2 CLI installed: pip install b2
#   2. Account authorized once on this box:
#        b2 account authorize <KEY_ID> <APP_KEY>
#      (or set B2_APPLICATION_KEY_ID + B2_APPLICATION_KEY env vars)
#   3. Bucket exists: b2 bucket create fls-checkpoints allPrivate
#
# Usage:
#   bash scripts/067_b2_sync.sh push-frames    [LOCAL_DIR]
#   bash scripts/067_b2_sync.sh pull-frames    [LOCAL_DIR]
#   bash scripts/067_b2_sync.sh push-features  [LOCAL_DIR]
#   bash scripts/067_b2_sync.sh pull-features  [LOCAL_DIR]
#   bash scripts/067_b2_sync.sh push-dataset   <VERSION> [LOCAL_DIR]
#   bash scripts/067_b2_sync.sh pull-dataset   <VERSION> [LOCAL_DIR]
#   bash scripts/067_b2_sync.sh push-checkpoint <RUN_NAME> [LOCAL_DIR]
#   bash scripts/067_b2_sync.sh pull-checkpoint <RUN_NAME> [LOCAL_DIR]
#   bash scripts/067_b2_sync.sh status

set -euo pipefail

BUCKET="${B2_BUCKET:-fls-checkpoints}"
B2_PREFIX_FRAMES="b2://${BUCKET}/lasana_frames/"
B2_PREFIX_FEATS="b2://${BUCKET}/lasana_features/"
B2_PREFIX_DATA="b2://${BUCKET}/training_data/"
B2_PREFIX_CKPT="b2://${BUCKET}/checkpoints/"

if ! command -v b2 >/dev/null 2>&1; then
    echo "ERROR: b2 CLI not installed. Run: pip install b2" >&2
    exit 2
fi

# Verify auth
if ! b2 account get >/dev/null 2>&1; then
    echo "ERROR: b2 not authorized. Run one of:" >&2
    echo "  b2 account authorize <KEY_ID> <APP_KEY>" >&2
    echo "  export B2_APPLICATION_KEY_ID=... B2_APPLICATION_KEY=..." >&2
    exit 2
fi

cmd="${1:-}"
shift || true

case "$cmd" in
    push-frames)
        DIR="${1:-./data/external/lasana_processed/frames}"
        echo "PUSH frames: $DIR -> ${B2_PREFIX_FRAMES}"
        b2 sync --delete "$DIR" "$B2_PREFIX_FRAMES"
        ;;
    pull-frames)
        DIR="${1:-./data/external/lasana_processed/frames}"
        mkdir -p "$DIR"
        echo "PULL frames: ${B2_PREFIX_FRAMES} -> $DIR"
        b2 sync "$B2_PREFIX_FRAMES" "$DIR"
        ;;
    push-features)
        DIR="${1:-./data/external/lasana_processed/features}"
        echo "PUSH features: $DIR -> ${B2_PREFIX_FEATS}"
        b2 sync --delete "$DIR" "$B2_PREFIX_FEATS"
        ;;
    pull-features)
        DIR="${1:-./data/external/lasana_processed/features}"
        mkdir -p "$DIR"
        echo "PULL features: ${B2_PREFIX_FEATS} -> $DIR"
        b2 sync "$B2_PREFIX_FEATS" "$DIR"
        ;;
    push-dataset)
        VER="${1:-}"
        DIR="${2:-./data/training/$VER}"
        if [ -z "$VER" ]; then echo "VERSION required" >&2; exit 2; fi
        echo "PUSH dataset $VER: $DIR -> ${B2_PREFIX_DATA}${VER}/"
        b2 sync --delete "$DIR" "${B2_PREFIX_DATA}${VER}/"
        ;;
    pull-dataset)
        VER="${1:-}"
        DIR="${2:-./data/training/$VER}"
        if [ -z "$VER" ]; then echo "VERSION required" >&2; exit 2; fi
        mkdir -p "$DIR"
        echo "PULL dataset $VER: ${B2_PREFIX_DATA}${VER}/ -> $DIR"
        b2 sync "${B2_PREFIX_DATA}${VER}/" "$DIR"
        ;;
    push-checkpoint)
        RUN="${1:-}"
        DIR="${2:-./memory/model_checkpoints/$RUN}"
        if [ -z "$RUN" ]; then echo "RUN_NAME required" >&2; exit 2; fi
        echo "PUSH checkpoint $RUN: $DIR -> ${B2_PREFIX_CKPT}${RUN}/"
        b2 sync --delete "$DIR" "${B2_PREFIX_CKPT}${RUN}/"
        echo ""
        echo "✓ Checkpoint persisted to B2. Safe to stop the pod."
        ;;
    pull-checkpoint)
        RUN="${1:-}"
        DIR="${2:-./memory/model_checkpoints/$RUN}"
        if [ -z "$RUN" ]; then echo "RUN_NAME required" >&2; exit 2; fi
        mkdir -p "$DIR"
        echo "PULL checkpoint $RUN: ${B2_PREFIX_CKPT}${RUN}/ -> $DIR"
        b2 sync "${B2_PREFIX_CKPT}${RUN}/" "$DIR"
        ;;
    status)
        echo "Bucket: $BUCKET"
        b2 ls "b2://${BUCKET}/" 2>/dev/null | head -50
        ;;
    *)
        cat <<USAGE
Usage: $0 <command> [args]

Commands:
  push-frames     [LOCAL_DIR]                 Phase 1 → B2
  pull-frames     [LOCAL_DIR]                 B2 → Phase 2
  push-features   [LOCAL_DIR]                 Phase 2 → B2
  pull-features   [LOCAL_DIR]                 B2 → Phase 3
  push-dataset    <VERSION> [LOCAL_DIR]       Local → B2
  pull-dataset    <VERSION> [LOCAL_DIR]       B2 → Phase 3
  push-checkpoint <RUN_NAME> [LOCAL_DIR]      Phase 3 → B2 (RUN BEFORE SHUTDOWN)
  pull-checkpoint <RUN_NAME> [LOCAL_DIR]      B2 → Local for eval
  status                                      List bucket contents

Environment:
  B2_BUCKET                Override default 'fls-checkpoints'
  B2_APPLICATION_KEY_ID    For non-interactive auth
  B2_APPLICATION_KEY
USAGE
        exit 1
        ;;
esac
