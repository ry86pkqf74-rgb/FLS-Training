# Storage Node Setup Guide

**Host:** `207.244.235.10` (Contabo VPS — Ubuntu 22.04, AMD Ryzen 9 7900, 928 GB NVMe)  
**Role:** Always-on durable artifact store for videos, frames, checkpoints, scored outputs, and training datasets.  
**Cost:** ~$99/mo. Replaces GitHub as the large-file store.

---

## Directory Layout (`/data/fls/`)

```
/data/fls/
  raw-videos/       — original .mov/.mp4 files from harvesting
  frames/           — extracted frames, organised by video_id/
  datasets-jsonl/   — training manifests (v2/v3 JSONL)
  checkpoints/      — adapter checkpoints from training runs
  eval/             — evaluation outputs and comparison reports
  logs/             — training logs, sync logs, cron output
  scored/           — scoring JSON outputs from frontier models
  backups/          — daily .tar.gz archives (auto-pruned after 30 days)
```

**What lives here (NOT in git):**
- Videos and extracted frames
- Model checkpoints (large binary files)
- Raw harvest/scoring logs
- Training JSONL datasets before they are versioned

---

## Installed Software

| Package | Version | Notes |
|---|---|---|
| Python | 3.11.0rc1 | System package |
| pip | bundled | `python3.11 -m pip` |
| duckdb | 1.5.1 | `python3.11 -m pip install duckdb` |
| rsync | 3.2.7 | System package |

---

## Scripts

### `sync_from_s1.sh`
Pulls `scored/` and `datasets-jsonl/` from S1 (217.77.2.114).

```bash
/data/fls/sync_from_s1.sh [--dry-run]
```

Override the SSH key with `SSH_KEY=/path/to/key ./sync_from_s1.sh`.

### `sync_to_gpu.sh`
Pushes `datasets-jsonl/` and `frames/` to a RunPod GPU node, routing through S1 as a ProxyJump.

```bash
/data/fls/sync_to_gpu.sh <GPU_HOST> [GPU_PORT] [--dry-run]
```

Environment overrides: `SSH_KEY`, `GPU_USER`, `GPU_DEST_BASE` (default `/workspace/FLS-Training`).

### `daily_backup.sh`
Date-stamped `.tar.gz` of `scored/` and `datasets-jsonl/`. Archives older than 30 days are pruned automatically.

---

## Cron Schedule

```
0 2 * * *   /data/fls/daily_backup.sh >> /data/fls/logs/cron.log 2>&1
```

Runs at **02:00 UTC** daily. Output at `/data/fls/logs/cron.log` and per-run log files in `/data/fls/logs/`.

---

## SSH Access

```bash
ssh root@207.244.235.10
```

No special port; uses default port 22. Add your public key to `/root/.ssh/authorized_keys` to allow passwordless access from new machines (required for rsync sync scripts).

### Allow storage node to pull from S1

On S1 (217.77.2.114):
```bash
cat /data/fls-storage-node.pub >> /root/.ssh/authorized_keys
```
Generate a key on the storage node first if needed:
```bash
ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N ""
cat /root/.ssh/id_ed25519.pub   # copy this to S1 and GPU nodes
```

---

## Typical Workflows

### After a scoring run (on your Mac or S1)
```bash
# push scored outputs to storage node directly
rsync -avz scored/ root@207.244.235.10:/data/fls/scored/
rsync -avz datasets-jsonl/ root@207.244.235.10:/data/fls/datasets-jsonl/
```

### Before a training run (from storage node)
```bash
ssh root@207.244.235.10
/data/fls/sync_to_gpu.sh <RUNPOD_IP> <RUNPOD_PORT>
```

### Pull latest scored data from S1
```bash
ssh root@207.244.235.10 /data/fls/sync_from_s1.sh
```
