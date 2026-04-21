#!/bin/bash
cd /opt/fls-training
echo "=== Watchdog: waiting for batch 1 to finish ==="

# Poll tmux session until it dies (batch 1 finished)
while tmux has-session -t fls 2>/dev/null; do
    SCORES=$(ls memory/scores/2026-04-13/*.json 2>/dev/null | wc -l)
    echo "$(date +%H:%M:%S) — batch 1 running, $SCORES score files"
    sleep 60
done

echo "=== Batch 1 finished at $(date) ==="
echo "Score files: $(ls memory/scores/2026-04-13/*.json 2>/dev/null | wc -l)"
echo "Videos on server: $(ls harvested_videos/*.mp4 2>/dev/null | wc -l)"

# Pull latest repo (gets validator fix)
cd /opt/fls-training
git fetch origin main 2>&1 | tail -2
git reset --hard origin/main 2>&1 | tail -1

# Restore .env (git reset might overwrite)
if [ ! -f .env ]; then
    echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}" > .env
    echo "OPENAI_API_KEY=${OPENAI_API_KEY}" >> .env
    chmod 600 .env
fi

# Launch v2 for remaining videos
echo "=== Launching batch 2 (v2 scorer) ==="
tmux new-session -d -s fls 'bash /opt/fls-training/run_batch_score_v3.sh'
sleep 5
tmux capture-pane -t fls -p -S -5 | grep -v "^$" | tail -5
echo "=== Batch 2 launched ==="
