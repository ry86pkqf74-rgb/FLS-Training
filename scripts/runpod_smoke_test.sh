#!/usr/bin/env bash
# scripts/runpod_smoke_test.sh — sub-60-second pre-launch smoke test
#
# Run this the moment SSH lands on a fresh RunPod or Vast.ai pod.
# Designed to fail loudly and immediately if the host is broken, so
# you can destroy it and pick a different one BEFORE wasting GPU hours
# debugging dependency hell.
#
# Cost: <60 seconds of pod time = <$0.05 even on H100 SXM.
#
# Usage:
#   bash scripts/runpod_smoke_test.sh           # GPU pod (default)
#   PHASE=cpu bash scripts/runpod_smoke_test.sh # CPU-only pod (phase 1)
#   PHASE=gpu bash scripts/runpod_smoke_test.sh # GPU pod (phases 2-3)
#
# Exit codes:
#   0   all checks pass; safe to start training
#   1   one or more checks failed; DESTROY this pod and pick another

set -uo pipefail

PHASE="${PHASE:-gpu}"
PASS=0
FAIL=0

ok()   { printf "  \033[32mok\033[0m   %s\n" "$1"; PASS=$((PASS+1)); }
fail() { printf "  \033[31mFAIL\033[0m %s\n" "$1"; FAIL=$((FAIL+1)); }
info() { printf "  ..   %s\n" "$1"; }

echo "==============================================="
echo "FLS-Training pre-launch smoke test (phase: $PHASE)"
echo "==============================================="
echo ""

# --- Host basics -----------------------------------------------------------
echo "[1] Host basics"

if command -v lsb_release >/dev/null 2>&1; then
    OS=$(lsb_release -ds 2>/dev/null || echo "unknown")
    info "OS: $OS"
fi

PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VER" | awk -F. '{print $1"."$2}')
case "$PY_MAJOR" in
    3.10|3.11|3.12) ok "python $PY_VER (supported)" ;;
    3.9|3.13)       fail "python $PY_VER (PyTorch wheels often missing)" ;;
    *)              fail "python $PY_VER (unsupported)" ;;
esac

DISK_AVAIL=$(df -BG /workspace 2>/dev/null || df -BG /)
WS=$(echo "$DISK_AVAIL" | awk 'NR==2 {print $4}' | tr -d 'G')
if [ -z "$WS" ]; then WS=0; fi
if [ "$WS" -ge 200 ]; then
    ok "/workspace free: ${WS}G"
elif [ "$WS" -ge 50 ]; then
    info "/workspace free: ${WS}G (ok for phase 3, NOT for phase 1)"
else
    fail "/workspace free: ${WS}G (too small)"
fi

if free -g >/dev/null 2>&1; then
    RAM=$(free -g | awk '/^Mem:/ {print $2}')
    if [ "$RAM" -ge 24 ]; then ok "RAM ${RAM}G"; else fail "RAM ${RAM}G (need ≥24)"; fi
fi

echo ""

# --- GPU + driver ----------------------------------------------------------
if [ "$PHASE" = "gpu" ]; then
    echo "[2] GPU + driver"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        fail "nvidia-smi not found (no GPU on this pod)"
    else
        GPU_LINE=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -1)
        info "GPU: $GPU_LINE"
        DRV=$(echo "$GPU_LINE" | awk -F, '{print $2}' | tr -d ' ' | cut -d. -f1)
        if [ -n "$DRV" ] && [ "$DRV" -ge 535 ]; then
            ok "driver $DRV (>= 535, supports CUDA 12.x)"
        else
            fail "driver $DRV (< 535, will break torch 2.10 cu128)"
        fi
        VRAM=$(echo "$GPU_LINE" | awk -F, '{print $3}' | tr -d ' MiB')
        if [ -n "$VRAM" ] && [ "$VRAM" -ge 70000 ]; then
            ok "VRAM ${VRAM} MiB (>= 70G, fits Qwen2.5-VL-7B bf16)"
        elif [ "$VRAM" -ge 40000 ]; then
            info "VRAM ${VRAM} MiB (40-70G — needs 4-bit quant)"
        else
            fail "VRAM ${VRAM} MiB (too small for 7B bf16)"
        fi
    fi
    echo ""
fi

# --- Python stack ----------------------------------------------------------
echo "[3] Python stack"

check_import() {
    local mod=$1
    local label=$2
    if python3 -c "import $mod" 2>/dev/null; then
        VER=$(python3 -c "import $mod; print(getattr($mod,'__version__','?'))" 2>/dev/null)
        ok "$label $VER"
    else
        if [ "${3:-required}" = "required" ]; then
            fail "$label not installed"
        else
            info "$label not installed (will install during setup)"
        fi
    fi
}

if [ "$PHASE" = "gpu" ]; then
    check_import torch "torch" optional
    check_import torchvision "torchvision" optional
    check_import transformers "transformers" optional
    check_import peft "peft" optional
    check_import accelerate "accelerate" optional
    check_import bitsandbytes "bitsandbytes" optional
    # Vision-specific dependencies for the v4 path
    check_import PIL "Pillow" optional
    check_import qwen_vl_utils "qwen-vl-utils" optional
    check_import unsloth "unsloth" optional
    check_import trl "trl" optional
    check_import datasets "datasets" optional

    # CUDA reachability via torch
    if python3 -c "import torch" 2>/dev/null; then
        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
            ok "torch.cuda.is_available() (cuda=$CUDA)"
        else
            fail "torch installed but cuda not available"
        fi
    fi

    # UnslothVisionDataCollator presence — the lever finetune_vlm.py pulls
    # when the first training example has image blocks. Missing collator =
    # broken vision path = we will abort before training.
    if python3 -c "import unsloth" 2>/dev/null; then
        if python3 -c "from unsloth.trainer import UnslothVisionDataCollator" 2>/dev/null; then
            ok "UnslothVisionDataCollator importable"
        else
            fail "unsloth installed but UnslothVisionDataCollator missing — upgrade unsloth"
        fi
    fi
fi

if [ "$PHASE" = "cpu" ]; then
    check_import numpy "numpy" optional
    check_import PIL "Pillow" optional
    check_import b2sdk "b2sdk" optional
fi

# Hardening-sprint project sanity: the quarantined modules must raise,
# and schema_adapter must import. This catches a revert before it ships.
echo ""
echo "[3a] Project sanity (hardening sprint)"
if [ -d "src" ]; then
    if python3 -c "from src.training import schema_adapter; schema_adapter.get_total_score({'estimated_fls_score': 450})" 2>/dev/null; then
        ok "schema_adapter importable + functional"
    else
        fail "schema_adapter broken — cannot read score records"
    fi

    # Quarantined modules should raise on first call, not silently work.
    if python3 -c "
import sys
try:
    from src.training.data_prep import prepare_training_data
    try:
        prepare_training_data()
    except RuntimeError as e:
        if 'quarantined' in str(e).lower():
            sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        ok "data_prep quarantine stub raises as expected"
    else
        fail "data_prep stub does not guard against the dead code path"
    fi
fi

echo ""

# --- ffmpeg / HEVC ---------------------------------------------------------
echo "[4] ffmpeg + HEVC"

if ! command -v ffmpeg >/dev/null 2>&1; then
    fail "ffmpeg not installed (apt-get install -y ffmpeg)"
else
    FF_VER=$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')
    info "ffmpeg $FF_VER"
    if ffmpeg -decoders 2>/dev/null | grep -qE '^V.*hevc'; then
        ok "HEVC decoder present"
    else
        fail "HEVC decoder MISSING (apt-get install -y libavcodec-extra)"
    fi
fi

echo ""

# --- Network ---------------------------------------------------------------
echo "[5] Network"

# Test HF + Github + Backblaze + TU Dresden in parallel via curl
test_host() {
    local host=$1
    local label=$2
    if curl -fsS --max-time 10 -o /dev/null "$host" 2>/dev/null; then
        ok "$label reachable"
    else
        fail "$label NOT reachable ($host)"
    fi
}

test_host "https://huggingface.co/api/models/Qwen/Qwen2.5-VL-7B-Instruct" "huggingface.co"
test_host "https://github.com" "github.com"
test_host "https://api.backblazeb2.com/b2api/v3/" "backblazeb2.com"
if [ "$PHASE" = "cpu" ]; then
    test_host "https://opara.zih.tu-dresden.de" "opara.zih.tu-dresden.de (LASANA)"
fi

echo ""

# --- Env vars --------------------------------------------------------------
echo "[6] Env vars"

if [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    ok "HF_TOKEN set"
else
    info "HF_TOKEN not set (only required for gated models — Qwen2.5-VL-7B is open)"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    ok "WANDB_API_KEY set"
else
    info "WANDB_API_KEY not set (set report_to: none in config or login)"
fi

if [ -n "${B2_APPLICATION_KEY_ID:-}" ] && [ -n "${B2_APPLICATION_KEY:-}" ]; then
    ok "B2_APPLICATION_KEY_ID + KEY set"
else
    info "B2 keys not set in env (run 'b2 account authorize' or set both vars)"
fi

echo ""
echo "==============================================="
echo "RESULT: $PASS passed, $FAIL failed"
echo "==============================================="

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "❌ FAILURES DETECTED — destroy this pod and pick a different host."
    echo "   Do NOT try to fix a broken host; the second-try cost is lower"
    echo "   than the debug cost on this one."
    exit 1
fi

echo ""
echo "✓ All checks passed. Safe to start training."
exit 0
