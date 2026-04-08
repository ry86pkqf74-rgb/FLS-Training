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

No special port; uses default port 22.

### Storage node → S1 trust (completed 2026-04-08)

The storage node's ed25519 public key has been installed in S1's `authorized_keys`.
Verified working:

```
storage (207.244.235.10) → S1 (217.77.2.114)  AUTH_OK
```

Storage node public key (`/root/.ssh/id_ed25519.pub`):
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIEJs7on7VwPl6nmYLNvFIWaTgwAqoQQCwtxPzmnT6a8 root@m66910.contaboserver.net
```

`sync_from_s1.sh` runs fully passwordless — no further action needed.

### Storage node → GPU nodes (required per RunPod pod)

GPU pods are ephemeral. When a new pod is launched, authorize the storage node once:

```bash
# From your Mac — install the storage node key into the new GPU pod
ssh-copy-id -f -i /tmp/storage_node.pub -p <PORT> root@<RUNPOD_IP>

# Or add to the RunPod pod template's startup script:
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIEJs7on7VwPl6nmYLNvFIWaTgwAqoQQCwtxPzmnT6a8 root@m66910.contaboserver.net" \
  >> /root/.ssh/authorized_keys
```

After that, `sync_to_gpu.sh` routes through S1 as ProxyJump and transfers without passwords.

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
