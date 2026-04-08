#!/usr/bin/env bash
# 064_jigsaws_download.sh
#
# Two modes:
#
#   MODE=samples bash scripts/064_jigsaws_download.sh
#       Pull the three publicly-available sample AVI clips (Suturing,
#       Knot-Tying, Needle-Passing) from the JHU-CIRL WordPress download
#       manager. Total ~3.4 MB. No email / registration required.
#       JHU's CDN sits behind Cloudflare bot protection, so we route
#       through web.archive.org's cached copies.
#
#   MODE=full bash scripts/064_jigsaws_download.sh   (DEFAULT)
#       Pull the full dataset (~12 GB) using a URL list emailed by
#       JHU-CIRL after the access form is approved. Set
#       JIGSAWS_URLS_FILE and (optionally) JIGSAWS_PW first.
#
# Access procedure for the FULL dataset:
#   1. Submit the request form at:
#      https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
#   2. Wait for approval email (usually same-day) — it contains a download
#      script + password, OR a direct URL list.
#   3. Save the password to env var JIGSAWS_PW and the URL list to
#      JIGSAWS_URLS_FILE (one URL per line), then run this script with
#      MODE=full (or just unset MODE).
#
# Usage examples:
#   MODE=samples bash scripts/064_jigsaws_download.sh
#   export JIGSAWS_PW=<password-from-email>
#   export JIGSAWS_URLS_FILE=/tmp/jigsaws_urls.txt
#   DEST=/workspace/jigsaws bash scripts/064_jigsaws_download.sh
#
# License: academic research only.

set -u
DEST="${DEST:-data/external/jigsaws}"
MODE="${MODE:-full}"
mkdir -p "$DEST"
log="$DEST/download.log"
echo "=== JIGSAWS download (mode=$MODE) started $(date) on $(hostname) ===" | tee -a "$log"

if [ "$MODE" = "samples" ]; then
  mkdir -p "$DEST/samples"
  cd "$DEST/samples"
  for entry in "Suturing.avi:1470" "Knot_Tying.avi:1475" "Needle_Passing.avi:1481"; do
    name="${entry%%:*}"
    id="${entry##*:}"
    if [ -f "$name" ]; then
      echo "[$(date)] SKIP $name" | tee -a "$log"; continue
    fi
    echo "[$(date)] START $name (sample, via wayback)" | tee -a "$log"
    curl -fsSL -A "Mozilla/5.0" \
      -o "${name}.tmp" \
      "https://web.archive.org/web/2024/https://cirl.lcsr.jhu.edu/download/$id/" \
      && mv "${name}.tmp" "$name" \
      && echo "[$(date)] DONE $name $(du -h "$name" | cut -f1)" | tee -a "$log"
  done
  echo "=== JIGSAWS samples complete $(date) ===" | tee -a "$log"
  exit 0
fi

URLS_FILE="${JIGSAWS_URLS_FILE:-}"
if [ -z "$URLS_FILE" ] || [ ! -f "$URLS_FILE" ]; then
  echo "ERROR: set JIGSAWS_URLS_FILE to a file containing one URL per line." | tee -a "$log"
  echo "If JHU-CIRL emailed you a download script, dump its URL list to that file." | tee -a "$log"
  exit 2
fi

cd "$DEST"
while IFS= read -r url; do
  [ -z "$url" ] && continue
  name="$(basename "${url%%\?*}")"
  if [ -f "$name" ]; then
    echo "[$(date)] SKIP $name" | tee -a "$log"; continue
  fi
  echo "[$(date)] START $name" | tee -a "$log"
  if [ -n "${JIGSAWS_PW:-}" ]; then
    curl -fSL --retry 5 --retry-delay 30 -u "jigsaws:${JIGSAWS_PW}" \
      -o "${name}.tmp" "$url" >> "$log" 2>&1
  else
    curl -fSL --retry 5 --retry-delay 30 -o "${name}.tmp" "$url" >> "$log" 2>&1
  fi
  mv "${name}.tmp" "$name"
  echo "[$(date)] DONE $name $(du -h "$name" | cut -f1)" | tee -a "$log"
done < "$URLS_FILE"

echo "=== JIGSAWS download complete $(date) ===" | tee -a "$log"
ls -lh "$DEST" | tee -a "$log"
