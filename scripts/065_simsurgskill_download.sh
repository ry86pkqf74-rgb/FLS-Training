#!/usr/bin/env bash
# 065_simsurgskill_download.sh
#
# Download the public SimSurgSkill 2021 dataset (MICCAI EndoVis sub-challenge)
# from the ISI public GCS bucket. No registration required.
#
# Usage:
#   bash scripts/065_simsurgskill_download.sh
#   DEST=/workspace/simsurgskill bash scripts/065_simsurgskill_download.sh
#
# License: non-commercial research only. See arXiv:2212.04448.

set -u
DEST="${DEST:-data/external/simsurgskill}"
URL="https://storage.googleapis.com/isi-simsurgskill-2021/simsurgskill_2021_dataset.zip"
EXPECTED_BYTES=4613059266
mkdir -p "$DEST"
cd "$DEST"
log="$DEST/download.log"
echo "=== SimSurgSkill download started $(date) on $(hostname) ===" | tee -a "$log"

ZIP="simsurgskill_2021_dataset.zip"
if [ -f "$ZIP" ]; then
  size=$(stat -f%z "$ZIP" 2>/dev/null || stat -c%s "$ZIP")
  if [ "$size" = "$EXPECTED_BYTES" ]; then
    echo "[$(date)] SKIP $ZIP (already complete: $size bytes)" | tee -a "$log"
  else
    echo "[$(date)] RESUME $ZIP (have $size of $EXPECTED_BYTES)" | tee -a "$log"
    curl -fSL --retry 5 --retry-delay 30 --max-time 7200 -C - -o "${ZIP}.tmp" "$URL" 2>&1 | tee -a "$log"
    mv "${ZIP}.tmp" "$ZIP"
  fi
else
  echo "[$(date)] START $ZIP" | tee -a "$log"
  curl -fSL --retry 5 --retry-delay 30 --max-time 7200 -o "${ZIP}.tmp" "$URL" 2>&1 | tee -a "$log"
  mv "${ZIP}.tmp" "$ZIP"
fi

# Verify size
size=$(stat -f%z "$ZIP" 2>/dev/null || stat -c%s "$ZIP")
if [ "$size" != "$EXPECTED_BYTES" ]; then
  echo "[$(date)] FAIL size mismatch: have $size, expected $EXPECTED_BYTES" | tee -a "$log"
  exit 1
fi
echo "[$(date)] SIZE OK ($size bytes)" | tee -a "$log"

# Unpack
if [ ! -d "simsurgskill_2021_dataset" ]; then
  echo "[$(date)] UNZIP $ZIP" | tee -a "$log"
  unzip -q "$ZIP" 2>&1 | tee -a "$log" || true
fi

echo "=== SimSurgSkill download complete $(date) ===" | tee -a "$log"
ls -la "$DEST" | tee -a "$log"
