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

# Real Synapse IDs walked from project syn25147789 on 2026-04-08.
# Total payload: 24.13 GB across 5 files. License: non-commercial research.
SYN_IDS=(
  "syn25871182"   # data/Training.zip          14.40 GB  (kinematics+video+segmentation+workflow, 90 cases)
  "syn27021898"   # data/Test.zip               9.72 GB  (60 cases)
  "syn27026293"   # PETRAW_detailed_results.zip 0.68 MB  (per-team scoring details)
  "syn27026295"   # Evaluation/Segmentation_metrics.py
  "syn27026296"   # Evaluation/Workflow_metrics.py
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
