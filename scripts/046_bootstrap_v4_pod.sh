#!/usr/bin/env bash
set -euo pipefail
S8=root@207.244.235.10
WS=/workspace
BRANCH=round2-vl-pipeline-fixes
echo "=== v4 pod bootstrap $(date) ==="
cd $WS
if [ ! -d FLS-Training ]; then git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git; fi
cd FLS-Training
git fetch origin $BRANCH; git checkout $BRANCH; git pull --ff-only origin $BRANCH
cp data/training/youtube_sft_v2/yt_train.jsonl $WS/yt_train.jsonl
cp data/training/youtube_sft_v2/yt_val.jsonl   $WS/yt_val.jsonl
cp data/training/youtube_sft_v2/yt_test.jsonl  $WS/yt_test.jsonl
wc -l $WS/yt_*.jsonl
if [ ! -d $WS/lasana_frames ] || [ $(ls $WS/lasana_frames 2>/dev/null | wc -l) -lt 900 ]; then
  echo "Fetching LASANA frames tarball from S8..."
  ssh -o StrictHostKeyChecking=no $S8 "cat /data/fls/lasana_frames_v4.tar" | tar -xf - -C $WS/
fi
echo "YT frames:     $(ls $WS/frames 2>/dev/null | wc -l) dirs"
echo "LASANA frames: $(ls $WS/lasana_frames 2>/dev/null | wc -l) dirs"
echo "Train rows:    $(wc -l < $WS/yt_train.jsonl)"
nvidia-smi | head -15
SESSION=trainv4
if tmux has-session -t $SESSION 2>/dev/null; then exit 0; fi
tmux new-session -d -s $SESSION "cd $WS/FLS-Training && python3 scripts/045_train_qwen_vl_v4.py 2>&1 | tee $WS/trainv4.log"
echo "Launched v4 training in tmux session '$SESSION'"
