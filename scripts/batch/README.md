# Batch Scoring Scripts (from S5 Hetzner)

These scripts were used for the YouTube FLS video harvesting and scoring campaign (April 2026).

## Scripts
- `auto_batch2.sh` — Watchdog that waits for batch 1 to finish, then launches batch 2
- `run_batch_score.sh` — v1 batch scorer (Claude + GPT-4o dual scoring)
- `run_batch_score_v2.sh` — v2 scorer with schema-aware score extraction
- `run_batch_score_v3.sh` — v3 scorer (Sonnet + Haiku, robust, with rubric injection)
- `run_harvest_score.sh` — Combined harvest (yt-dlp download) + score + push-to-GitHub pipeline

## Notes
- All scripts source API keys from `.env` (not committed)
- Full script contents backed up on S8 at `/data/fls/fls-training-backup/`
- Scoring results (2012 JSON files) are in `memory/scores/` 
- Videos (560 mp4, 14GB) are on S8 at `/data/fls/harvested_videos/`
