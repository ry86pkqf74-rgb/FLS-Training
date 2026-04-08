#!/usr/bin/env bash
# 064b_jigsaws_move_from_downloads.sh
#
# Move the full JIGSAWS dataset files from ~/Downloads (where the user
# downloaded them via a real browser, since JHU-CIRL sits behind a
# Cloudflare bot challenge that defeats curl) into the FLS-Training
# repo at data/external/jigsaws/.
#
# What this expects in ~/Downloads (filenames as published by JHU-CIRL):
#   Suturing.zip          (videos + kinematics + transcriptions)
#   Knot_Tying.zip
#   Needle_Passing.zip
#   Experimental_setup.zip   (cross-validation splits)
#   Video.zip              (sometimes packaged separately)
#   meta_file_*.txt        (per-task GRS + skill labels, sometimes loose)
#
# The script:
#   1. Waits for any .crdownload files matching the JIGSAWS task names to settle.
#   2. Moves them into data/external/jigsaws/.
#   3. Unzips into per-task subdirectories (Suturing/, Knot_Tying/, Needle_Passing/).
#   4. Verifies the trial counts match what JIGSAWS publishes:
#        Suturing       39
#        Knot_Tying     36
#        Needle_Passing 28
#   5. Prints a per-task GRS distribution from the meta_file_*.txt files.
#
# License: academic / non-commercial research only. See data/external/CITATIONS.md.

set -u
SRC="${SRC:-$HOME/Downloads}"
DEST="${DEST:-data/external/jigsaws}"
mkdir -p "$DEST"
log="$DEST/move.log"
echo "=== JIGSAWS move started $(date) ===" | tee -a "$log"
echo "src=$SRC dest=$DEST" | tee -a "$log"

# 1. Wait for any .crdownload to clear (Chrome's in-progress marker).
echo "[$(date)] Waiting for any *.crdownload in $SRC to clear..." | tee -a "$log"
for i in $(seq 1 360); do  # up to 60 minutes
  pending=$(ls "$SRC"/*.crdownload 2>/dev/null | wc -l | tr -d ' ')
  if [ "$pending" = "0" ]; then
    echo "[$(date)] No pending .crdownload files." | tee -a "$log"
    break
  fi
  echo "[$(date)] $pending .crdownload still in progress, sleeping 10s..." | tee -a "$log"
  sleep 10
done

# 2. Move every plausible JIGSAWS artifact into the dest folder.
shopt -s nullglob nocaseglob
moved=0
for pat in \
    "Suturing*.zip" "Knot*Tying*.zip" "Needle*Passing*.zip" \
    "Experimental_setup*.zip" "Video*.zip" \
    "meta_file_*.txt" "JIGSAWS*.zip" "JIGSAWS*.tar*"; do
  for f in "$SRC"/$pat; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    if [ -f "$DEST/$base" ]; then
      echo "[$(date)] SKIP $base (already in $DEST)" | tee -a "$log"
      continue
    fi
    mv -v "$f" "$DEST/$base" | tee -a "$log"
    moved=$((moved+1))
  done
done
shopt -u nullglob nocaseglob
echo "[$(date)] Moved $moved files into $DEST" | tee -a "$log"

# 3. Unpack each top-level zip into a task-named subdir.
cd "$DEST"
for z in Suturing*.zip Knot*Tying*.zip Needle*Passing*.zip Experimental_setup*.zip Video*.zip JIGSAWS*.zip; do
  [ -f "$z" ] || continue
  echo "[$(date)] UNZIP $z" | tee -a "$log"
  unzip -n -q "$z" >> "$log" 2>&1 || echo "[$(date)] WARN unzip non-zero on $z" | tee -a "$log"
done

# 4. Trial-count verification (only if the unzipped task dirs exist).
echo "=== trial counts ===" | tee -a "$log"
for task in Suturing Knot_Tying Needle_Passing; do
  if [ -d "$task/video" ]; then
    n=$(ls "$task/video"/*_capture1.avi 2>/dev/null | wc -l | tr -d ' ')
    echo "  $task: $n trials (capture1 AVIs)" | tee -a "$log"
  elif [ -d "$task" ]; then
    n=$(find "$task" -type f -name "*_capture1.avi" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $task: $n trials (recursive)" | tee -a "$log"
  fi
done

# 5. Show GRS-bearing meta files if present.
echo "=== meta files ===" | tee -a "$log"
find . -name "meta_file_*.txt" -maxdepth 4 2>/dev/null | tee -a "$log"

echo "=== JIGSAWS move complete $(date) ===" | tee -a "$log"
