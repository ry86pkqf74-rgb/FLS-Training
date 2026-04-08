#!/usr/bin/env bash
# daily_backup.sh — Archive scored/ and datasets-jsonl/ with a date-stamped tarball
# Installed at /data/fls/daily_backup.sh on the storage node (207.244.235.10)
# Runs daily at 02:00 UTC via cron.
set -euo pipefail

BASE="/data/fls"
ARCHIVE_DIR="${BASE}/backups"
DATE=$(date +%Y%m%d)
LOG="${BASE}/logs/daily_backup_${DATE}.log"

mkdir -p "${ARCHIVE_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -Iseconds)] Daily backup starting"

for DIR in scored datasets-jsonl; do
    DEST="${ARCHIVE_DIR}/${DIR}_${DATE}.tar.gz"
    echo "Archiving ${BASE}/${DIR} -> ${DEST}"
    tar -czf "${DEST}" -C "${BASE}" "${DIR}"
    echo "  size: $(du -sh "${DEST}" | cut -f1)"
done

# Prune archives older than 30 days
find "${ARCHIVE_DIR}" -name "*.tar.gz" -mtime +30 -delete && \
    echo "Pruned backups older than 30 days"

echo "[$(date -Iseconds)] Daily backup complete"
