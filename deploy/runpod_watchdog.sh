#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${1:-training/data/v2}"
CONFIG_PATH="${2:-src/configs/finetune_task5_v2.yaml}"
RUN_DIR_OVERRIDE="${RUN_DIR_OVERRIDE:-}"
BASE_MODEL_OVERRIDE="${BASE_MODEL_OVERRIDE:-}"
CONTINUOUS_HOURS="${CONTINUOUS_HOURS:-0}"
SECONDS_PER_STEP_ESTIMATE="${SECONDS_PER_STEP_ESTIMATE:-7}"
WATCHDOG_POLL_SECONDS="${WATCHDOG_POLL_SECONDS:-15}"
WATCHDOG_MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS:-10}"
TRAIN_LOG="${TRAIN_LOG:-/workspace/fls_train_continuous.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-/workspace/fls_watchdog.log}"

cd /workspace/FLS-Training

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$WATCHDOG_LOG"
}

trainer_running() {
    pgrep -f "finetune_vlm.*--config /tmp/finetune_runtime.yaml" >/dev/null 2>&1 || \
        pgrep -f "finetune_vlm.*--config ${CONFIG_PATH}" >/dev/null 2>&1
}

launcher_running() {
    pgrep -f "bash /workspace/FLS-Training/deploy/runpod_launch.sh ${DATASET_PATH} ${CONFIG_PATH}" >/dev/null 2>&1
}

training_complete() {
    grep -q "Training complete!" "$TRAIN_LOG" 2>/dev/null
}

configured_run_dir() {
    python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
if not config_path.exists():
    raise SystemExit(0)

with config_path.open() as handle:
    config = yaml.safe_load(handle) or {}

output_dir = str(config.get("output_dir") or "").strip()
if output_dir:
    print(output_dir)
PY
}

latest_run_dir() {
    if [[ -n "$RUN_DIR_OVERRIDE" ]]; then
        printf '%s\n' "$RUN_DIR_OVERRIDE"
        return 0
    fi

    local configured_dir
    configured_dir="$(configured_run_dir || true)"
    if [[ -n "$configured_dir" ]]; then
        printf '%s\n' "$configured_dir"
        return 0
    fi

    find /workspace/FLS-Training/memory/model_checkpoints -maxdepth 1 -mindepth 1 -type d -name '*_unsloth' | sort | tail -n 1
}

latest_checkpoint_dir() {
    local run_dir="$1"
    find "$run_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

start_or_resume_training() {
    local run_dir="$1"
    local checkpoint_dir=""
    local launch_env=()

    if [[ -n "$run_dir" ]]; then
        launch_env+=("OUTPUT_DIR_OVERRIDE=$run_dir")
    fi

    checkpoint_dir="$(latest_checkpoint_dir "$run_dir" || true)"
    if [[ -n "$checkpoint_dir" ]]; then
        launch_env+=("RESUME_FROM_CHECKPOINT=$checkpoint_dir")
        log "Resuming training from $checkpoint_dir"
    else
        log "No checkpoint found yet for ${run_dir:-<new run>}; launching from base model"
    fi

    if [[ -n "$BASE_MODEL_OVERRIDE" ]]; then
        launch_env+=("BASE_MODEL_OVERRIDE=$BASE_MODEL_OVERRIDE")
    fi
    launch_env+=("SKIP_DEP_INSTALL=1")
    launch_env+=("CONTINUOUS_HOURS=$CONTINUOUS_HOURS")
    launch_env+=("SECONDS_PER_STEP_ESTIMATE=$SECONDS_PER_STEP_ESTIMATE")

    nohup env PYTHONUNBUFFERED=1 "${launch_env[@]}" \
        bash /workspace/FLS-Training/deploy/runpod_launch.sh "$DATASET_PATH" "$CONFIG_PATH" \
        >> "$TRAIN_LOG" 2>&1 < /dev/null &
    log "Started training launcher with PID $!"
}

log "Watchdog started for dataset=$DATASET_PATH config=$CONFIG_PATH"
restarts=0

while true; do
    if training_complete; then
        log "Training log reports completion. Watchdog exiting."
        exit 0
    fi

    if trainer_running || launcher_running; then
        sleep "$WATCHDOG_POLL_SECONDS"
        continue
    fi

    if (( restarts >= WATCHDOG_MAX_RESTARTS )); then
        log "Reached restart limit ($WATCHDOG_MAX_RESTARTS). Watchdog exiting with failure."
        exit 1
    fi

    run_dir="$(latest_run_dir || true)"
    if [[ -z "$run_dir" ]]; then
        run_dir="/workspace/FLS-Training/memory/model_checkpoints/$(date -u +%Y%m%d_%H%M)_unsloth"
        mkdir -p "$run_dir"
    fi

    restarts=$((restarts + 1))
    log "Trainer not running. Restart attempt $restarts using run dir $run_dir"
    start_or_resume_training "$run_dir"
    sleep "$WATCHDOG_POLL_SECONDS"
done
