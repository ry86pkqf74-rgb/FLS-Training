#!/usr/bin/env bash
# 064_jigsaws_download.sh
#
# Download the JIGSAWS dataset from JHU-CIRL after access is approved.
# Access procedure:
#   1. Submit the request form at:
#      https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
#   2. Wait for approval email (usually same-day) — it contains a download
#      script + password, OR a direct URL list.
#   3. Save the password to env var JIGSAWS_PW and the URL list to
#      JIGSAWS_URLS_FILE (one URL per line), then run this script.
#
# Usage:
#   export JIGSAWS_PW=<password-from-email>
#   export JIGSAWS_URLS_FILE=/tmp/jigsaws_urls.txt
#   DEST=/workspace/jigsaws bash scripts/064_jigsaws_download.sh
#
# License: academic research only.

set -u
DEST="${DEST:-data/external/jigsaws}"
mkdir -p "$DEST"
log="$DEST/download.log"
echo "=== JIGSAWS download started $(date) on $(hostname) ===" | tee -a "$log"

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
