#!/usr/bin/env bash

set -euo pipefail

REPO_SLUG="${REPO_SLUG:-ry86pkqf74-rgb/FLS-Training}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORK_DIR="${WORK_DIR:-/opt/FLS-Training}"
VENV_DIR="${VENV_DIR:-/opt/fls-training-venv}"
PROMPT_VERSION="${PROMPT_VERSION:-v002}"
HARVEST_MAX="${HARVEST_MAX:-100000}"
HARVEST_PROBE_FIRST="${HARVEST_PROBE_FIRST:-1}"
HARVEST_PROBE_MAX="${HARVEST_PROBE_MAX:-25}"
SCORE_MAX="${SCORE_MAX:-100000}"
SCORER_DELAY="${SCORER_DELAY:-5}"
RUN_CONSENSUS="${RUN_CONSENSUS:-1}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
PUSH_RESULTS="${PUSH_RESULTS:-1}"
HARVEST_INCLUDE_UNCLASSIFIED="${HARVEST_INCLUDE_UNCLASSIFIED:-0}"
LOG_ROOT="${LOG_ROOT:-/var/log/fls-training}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/s1_harvest_score_$TIMESTAMP.log"

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: required environment variable is not set: $name" >&2
        exit 1
    fi
}

configure_repo_url() {
    if [[ -n "${FLS_REPO_URL:-}" ]]; then
        printf '%s\n' "$FLS_REPO_URL"
        return
    fi
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        printf 'https://x-access-token:%s@github.com/%s.git\n' "$GITHUB_TOKEN" "$REPO_SLUG"
        return
    fi
    printf 'https://github.com/%s.git\n' "$REPO_SLUG"
}

log_step() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

ensure_system_packages() {
    log_step "Installing system dependencies"

    local packages=(ffmpeg git nodejs python3 python3-pip python3-venv tmux)
    local missing=()
    local package
    for package in "${packages[@]}"; do
        if ! dpkg -s "$package" >/dev/null 2>&1; then
            missing+=("$package")
        fi
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        echo "All required system packages are already installed."
        return
    fi

    apt-get update -qq
    apt-get install -y -qq "${missing[@]}"
}

sync_repo() {
    local repo_url
    repo_url="$(configure_repo_url)"

    log_step "Syncing repository into $WORK_DIR"
    if [[ -d "$WORK_DIR/.git" ]]; then
        cd "$WORK_DIR"
        if [[ -n "$(git status --porcelain)" ]]; then
            echo "ERROR: $WORK_DIR has uncommitted changes; refusing to overwrite." >&2
            exit 1
        fi
        git remote set-url origin "$repo_url"
        git fetch origin "$REPO_BRANCH"
        git checkout "$REPO_BRANCH"
        git pull --ff-only origin "$REPO_BRANCH"
    else
        mkdir -p "$(dirname "$WORK_DIR")"
        git clone --branch "$REPO_BRANCH" "$repo_url" "$WORK_DIR"
        cd "$WORK_DIR"
    fi
}

setup_python() {
    log_step "Installing Python dependencies"
    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -e . yt-dlp
}

run_harvest() {
    log_step "Harvesting videos from data/harvest_targets.csv"
    if [[ "$HARVEST_PROBE_FIRST" == "1" ]]; then
        log_step "Probing harvest target accessibility before downloading"
        local probe_cmd=(python scripts/011c_harvest_from_csv.py --probe-only --max "$HARVEST_PROBE_MAX")
        if [[ "$HARVEST_INCLUDE_UNCLASSIFIED" == "1" ]]; then
            probe_cmd+=(--include-unclassified)
        fi
        "${probe_cmd[@]}"
    fi

    local cmd=(python scripts/011c_harvest_from_csv.py --max "$HARVEST_MAX")
    if [[ "$HARVEST_INCLUDE_UNCLASSIFIED" == "1" ]]; then
        cmd+=(--include-unclassified)
    fi
    "${cmd[@]}"
}

run_scoring() {
    log_step "Scoring harvested videos with Claude Sonnet and GPT-4o"
    python scripts/021_batch_score.py \
        --max "$SCORE_MAX" \
        --prompt-version "$PROMPT_VERSION" \
        --delay "$SCORER_DELAY"
}

run_consensus() {
    if [[ "$RUN_CONSENSUS" != "1" ]]; then
        return
    fi
    log_step "Running consensus and coach feedback"
    python scripts/030_run_consensus.py --prompt-version "$PROMPT_VERSION" --with-coach-feedback
}

run_validation() {
    if [[ "$RUN_VALIDATION" != "1" ]]; then
        return
    fi
    log_step "Running auto-validation over raw score JSONs"
    python scripts/026_auto_validate.py \
        --scores-dir memory/scores \
        --output-jsonl memory/validation_results.jsonl
}

push_results() {
    if [[ "$PUSH_RESULTS" != "1" ]]; then
        log_step "Skipping git push because PUSH_RESULTS=$PUSH_RESULTS"
        return
    fi

    require_env GITHUB_TOKEN

    log_step "Committing and pushing generated score artifacts"
    git remote set-url origin "$(configure_repo_url)"
    git add harvest_log.jsonl memory/scores memory/comparisons memory/feedback memory/validation_results.jsonl

    if git diff --cached --quiet; then
        echo "No new score artifacts to push."
        return
    fi

    git config user.name "${GIT_AUTHOR_NAME:-FLS Automation}"
    git config user.email "${GIT_AUTHOR_EMAIL:-fls-automation@localhost}"
    git commit -m "data: S1 harvest + score run ($TIMESTAMP)"
    git pull --rebase origin "$REPO_BRANCH"
    git push origin "HEAD:$REPO_BRANCH"
}

main() {
    require_env ANTHROPIC_API_KEY
    require_env OPENAI_API_KEY

    export PYTHONUNBUFFERED=1
    export PIP_DISABLE_PIP_VERSION_CHECK=1

    mkdir -p "$LOG_ROOT"
    exec > >(tee -a "$LOG_FILE") 2>&1

    echo "=== FLS S1 harvest + score pipeline started at $(date -Is) ==="
    echo "Work dir: $WORK_DIR"
    echo "Branch: $REPO_BRANCH"
    echo "Log file: $LOG_FILE"

    ensure_system_packages
    sync_repo
    setup_python
    run_harvest
    run_scoring
    run_consensus
    run_validation
    push_results

    echo "=== Pipeline complete at $(date -Is) ==="
}

main "$@"