#!/usr/bin/env bash
# 062_petraw_download.sh
#
# Download the PETRAW (PEg TRAnsfer Workflow Recognition) dataset from
# Synapse. Synapse access requires a free account + click-through DUA.
# After accepting the DUA at https://www.synapse.org/PETRAW, fill in
# the SYN_IDS array below with the file/folder Synapse IDs and run.
#
# Usage:
#   pip install synapseclient --break-system-packages
#   export SYNAPSE_AUTH_TOKEN=<your-PAT>
#   DEST=/workspace/petraw bash scripts/062_petraw_download.sh
#
# License: non-commercial research only (Synapse DUA).

set -u
DEST="${DEST:-data/external/petraw}"
mkdir -p "$DEST"
log="$DEST/download.log"
echo "=== PETRAW download started $(date) on $(hostname) ===" | tee -a "$log"

if [ -z "${SYNAPSE_AUTH_TOKEN:-}" ]; then
  echo "ERROR: set SYNAPSE_AUTH_TOKEN to your Synapse personal access token." | tee -a "$log"
  echo "Generate one at https://www.synapse.org/Profile:v/settings under 'Personal Access Tokens'." | tee -a "$log"
  exit 2
fi

# TODO: fill in after accepting DUA at https://www.synapse.org/PETRAW
# Inspect the project Files tab and copy the syn IDs of the parent folders
# (training data, test data, kinematics, segmentation, annotations).
SYN_IDS=(
  # "syn26134741"   # PETRAW training (placeholder — verify)
  # "syn26134742"   # PETRAW test     (placeholder — verify)
)

if [ ${#SYN_IDS[@]} -eq 0 ]; then
  echo "ERROR: SYN_IDS is empty. Edit this script with the actual Synapse IDs." | tee -a "$log"
  exit 3
fi

cd "$DEST"
for sid in "${SYN_IDS[@]}"; do
  echo "[$(date)] synapse get -r $sid" | tee -a "$log"
  synapse get -r "$sid" 2>&1 | tee -a "$log"
done

echo "=== PETRAW download complete $(date) ===" | tee -a "$log"
ls -lh "$DEST" | tee -a "$log"
