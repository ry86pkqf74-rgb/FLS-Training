#!/usr/bin/env bash
# 061_lasana_download_runpod.sh
#
# Download LASANA stereo video archives from TU Dresden Opara onto a
# RunPod (or any) server. Resumable, atomic, parallelizes 2 files at a
# time per pod. Usage:
#
#   POD=A bash 061_lasana_download_runpod.sh   # PegTransfer + CircleCutting
#   POD=B bash 061_lasana_download_runpod.sh   # BalloonResection + SutureAndKnot
#   POD=ALL bash 061_lasana_download_runpod.sh # all 8 video archives
#
# Output goes to ${LASANA_DIR:-/workspace/lasana}/videos. The annotations,
# samples, and metadata are also fetched (small).
#
# Source: https://opara.zih.tu-dresden.de/handle/123456789/1907  (CC BY 4.0)

set -u
POD="${POD:-ALL}"
LASANA_DIR="${LASANA_DIR:-/workspace/lasana}"
mkdir -p "$LASANA_DIR/videos" "$LASANA_DIR/_meta" "$LASANA_DIR/annotations" "$LASANA_DIR/samples"
log="$LASANA_DIR/download.log"
echo "=== LASANA download POD=$POD started $(date) on $(hostname) ===" | tee -a "$log"

base="https://opara.zih.tu-dresden.de/server/api/core/bitstreams"

# UUID -> filename
declare -A SMALL=(
  ["aad0a7eb-fdc3-421e-9f03-2fa4eb211e2c"]="annotations/Annotation.zip"
  ["c527ce4a-c79f-4950-9af2-b1e87020195f"]="_meta/camera_calibration.yaml"
  ["b58fc1a1-91ea-4934-bb61-16a8f1674d4e"]="samples/example_videos.zip"
  ["93d6cd40-fefa-49f3-8dbc-69e1b6f27508"]="Readme.md"
)

declare -A POD_A=(
  ["f577d564-0745-45b9-ac88-f82e0b7fdbee"]="videos/PegTransfer_left.zip"
  ["e9b58d48-2465-4ee2-a3ca-1bb8259b81da"]="videos/PegTransfer_right.zip"
  ["da15f935-03bf-4483-aab2-10315f53cdc8"]="videos/CircleCutting_left.zip"
  ["71209477-b015-442c-a3e9-b35957f37203"]="videos/CircleCutting_right.zip"
)

declare -A POD_B=(
  ["08bc523e-b123-47d2-82b6-564e6b6551b1"]="videos/BalloonResection_left.zip"
  ["7f522ef6-dafc-4761-b614-1ef3d405f853"]="videos/BalloonResection_right.zip"
  ["005bdc22-ca88-47d7-b998-22c70dc73f80"]="videos/SutureAndKnot_left.zip"
  ["cc8e197b-c96e-440b-b813-9a6e39e738ab"]="videos/SutureAndKnot_right.zip"
)

dl_one() {
  local uuid="$1" dest="$2"
  local out="$LASANA_DIR/$dest"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ ! -f "${out}.tmp" ]; then
    echo "[$(date)] SKIP $dest (exists)" >> "$log"; return 0
  fi
  echo "[$(date)] START $dest" >> "$log"
  curl -fSL --retry 5 --retry-delay 30 --max-time 14400 -C - \
    -o "${out}.tmp" "$base/$uuid/content" >> "$log" 2>&1 \
    && mv "${out}.tmp" "$out" \
    && echo "[$(date)] DONE  $dest $(du -h "$out" | cut -f1)" >> "$log" \
    || { echo "[$(date)] FAIL  $dest" >> "$log"; return 1; }
}

# Always fetch the small bits (annotations, calibration, samples, readme)
for uuid in "${!SMALL[@]}"; do dl_one "$uuid" "${SMALL[$uuid]}"; done

run_pair_parallel() {
  # $1, $2 are uuid:dest:uuid:dest pairs (4 args)
  dl_one "$1" "$2" &
  dl_one "$3" "$4" &
  wait
}

case "$POD" in
  A|a)
    run_pair_parallel \
      "f577d564-0745-45b9-ac88-f82e0b7fdbee" "videos/PegTransfer_left.zip" \
      "da15f935-03bf-4483-aab2-10315f53cdc8" "videos/CircleCutting_left.zip"
    run_pair_parallel \
      "e9b58d48-2465-4ee2-a3ca-1bb8259b81da" "videos/PegTransfer_right.zip" \
      "71209477-b015-442c-a3e9-b35957f37203" "videos/CircleCutting_right.zip"
    ;;
  B|b)
    run_pair_parallel \
      "08bc523e-b123-47d2-82b6-564e6b6551b1" "videos/BalloonResection_left.zip" \
      "005bdc22-ca88-47d7-b998-22c70dc73f80" "videos/SutureAndKnot_left.zip"
    run_pair_parallel \
      "7f522ef6-dafc-4761-b614-1ef3d405f853" "videos/BalloonResection_right.zip" \
      "cc8e197b-c96e-440b-b813-9a6e39e738ab" "videos/SutureAndKnot_right.zip"
    ;;
  ALL|all)
    run_pair_parallel \
      "f577d564-0745-45b9-ac88-f82e0b7fdbee" "videos/PegTransfer_left.zip" \
      "da15f935-03bf-4483-aab2-10315f53cdc8" "videos/CircleCutting_left.zip"
    run_pair_parallel \
      "e9b58d48-2465-4ee2-a3ca-1bb8259b81da" "videos/PegTransfer_right.zip" \
      "71209477-b015-442c-a3e9-b35957f37203" "videos/CircleCutting_right.zip"
    run_pair_parallel \
      "08bc523e-b123-47d2-82b6-564e6b6551b1" "videos/BalloonResection_left.zip" \
      "005bdc22-ca88-47d7-b998-22c70dc73f80" "videos/SutureAndKnot_left.zip"
    run_pair_parallel \
      "7f522ef6-dafc-4761-b614-1ef3d405f853" "videos/BalloonResection_right.zip" \
      "cc8e197b-c96e-440b-b813-9a6e39e738ab" "videos/SutureAndKnot_right.zip"
    ;;
  *) echo "POD must be A, B, or ALL"; exit 2 ;;
esac

echo "=== LASANA download POD=$POD complete $(date) ===" | tee -a "$log"
ls -lh "$LASANA_DIR/videos" | tee -a "$log"
