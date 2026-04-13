# FLS-Training Infrastructure Reference

**Last verified:** 2026-04-13

## Active Servers

### Hetzner Free Tier (FLS Primary)
- **IP:** 77.42.85.109
- **Name:** ubuntu-32gb-hel1-1
- **Specs:** 16 cores, 30GB RAM, 601GB disk
- **SSH:** `ssh -i ~/.ssh/id_ed25519 root@77.42.85.109`
- **Hetzner CLI:** `hcloud server list` (context: fls)
- **Role:** Harvest pipeline, scoring orchestration, API calls
- **State (2026-04-13):** ResearchFlow killed (ollama, rotation service, crons). FLS harvest+score pipeline running in tmux `fls`.
- **FLS repo:** `/opt/fls-training/`
- **API keys:** `/opt/fls-training/.env`

### Contabo (Data Storage)
- **IP:** 207.244.235.10
- **Specs:** 12 cores (Ryzen 9 7900), ~1TB NVMe
- **SSH:** `ssh -i ~/.ssh/id_ed25519 root@207.244.235.10`
- **Role:** LASANA dataset host, scored data backups
- **FLS data:** `/data/fls/` (280GB)
  - `/data/fls/lasana_layout/` — 1,270 LASANA trial directories with video.mkv
  - `/data/fls/lasana_processed/frames/` — 275K extracted frames
  - `/data/fls/raw-videos/lasana/` — Raw LASANA zip archives
  - `/data/fls/scored/` — Score file backups (715 files)
  - `/data/fls/backups/` — Daily tar.gz backups of datasets + scores

### GPU Compute (On-Demand)
- **RunPod:** Account active, credits available. Use for SFT/DPO training only.
- **Vast.ai:** Account active. Alternative GPU source.
- **DO NOT spin up until:** ≥80 ACCEPTED training examples validated

## Dead / Unreachable Infrastructure

The following are ResearchFlow fleet servers. They are NOT part of the FLS project and were all timing out as of 2026-04-13:

| Server | IP | Status |
|--------|-----|--------|
| S1 | 217.77.2.114 | UNREACHABLE |
| S2 | 194.164.72.131 | UNREACHABLE |
| S3 | 37.27.194.168 | UNREACHABLE |
| S4 | 37.27.204.182 | UNREACHABLE |
| S5 | 37.27.207.12 | UNREACHABLE |
| S6 | 37.27.192.164 | UNREACHABLE |
| S7 | 144.126.134.255 | UNREACHABLE |
| S8 (Contabo) | 207.244.235.10 | This IS reachable — it's the Contabo data server above |

The IP 178.156.138.70 in `~/.ssh/config` as `researchflow-hetzner` is an old Hetzner server. SSH key is not authorized — needs key injection via Hetzner console to regain access.

## Monitoring Commands

```bash
# Check Hetzner pipeline
ssh -i ~/.ssh/id_ed25519 root@77.42.85.109 'tmux attach -t fls'

# Quick status
ssh -i ~/.ssh/id_ed25519 root@77.42.85.109 'ls /opt/fls-training/harvested_videos/*.mp4 2>/dev/null | wc -l; cat /proc/loadavg'

# Check Contabo data
ssh -i ~/.ssh/id_ed25519 root@207.244.235.10 'du -sh /data/fls/*'

# Hetzner server via CLI
hcloud server list
hcloud server describe ubuntu-32gb-hel1-1
```

## API Keys

- **ANTHROPIC_API_KEY:** Set in `/opt/fls-training/.env` on Hetzner
- **OPENAI_API_KEY:** Set in `/opt/fls-training/.env` on Hetzner
- **GitHub token:** `$GITHUB_TOKEN (see password manager)` (repo + workflow scopes)
- **Hetzner API:** Configured in `hcloud` CLI context `fls`
