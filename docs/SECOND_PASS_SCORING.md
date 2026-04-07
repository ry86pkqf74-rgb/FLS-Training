# Second-Pass Scoring: Fill Gaps & Generate Consensus

## Status as of 2026-04-07

Videos with BOTH Claude + GPT-4o scores (ready for consensus):
- lap_pre_video, post_practice_video, post2, post3, post4
- postV6, postV7, postV8, postV9
- V8 (GPT-4o only — needs Claude re-score or skip)
- V11, V12, V13, V14, V15, V16

Videos with Claude-ONLY scores (need GPT-4o pass):
- postV10, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26

## Step 1: Score Missing Videos with GPT-4o

Use the existing prompt at `prompts/v001_task5_chatgpt_system.md` (or `chatgpt prompt for scoring.txt` in repo root).

For each missing video, upload the frames to ChatGPT with this wrapper:

```
You are scoring FLS Task 5 (Intracorporeal Suture with Knot Tying).

Video: {VIDEO_ID}
Duration: {DURATION}s
Frames: {N_FRAMES} uniform + 3 final-state frames attached

[paste or reference the full system prompt from prompts/v001_task5_chatgpt_system.md]

Analyze the attached frames now.
```

Save each GPT-4o response as:
`memory/scores/2026-04-07/{VIDEO_ID}_gpt-4o_{TIMESTAMP}.json`

## Step 2: Generate Consensus (Claude or GPT-4o as Critique Agent)

For each video that now has BOTH teacher scores, run consensus using:
`prompts/v001_task5_consensus.md`

### Consensus Prompt Template (paste into Claude or ChatGPT):

```
You are the FLS Critique Agent. This is Round 1 of a multi-turn consensus review.

## Teacher A (Claude Sonnet 4):
{PASTE FULL CLAUDE JSON}

## Teacher B (GPT-4o):
{PASTE FULL GPT-4O JSON}

## Video Metadata:
- Video ID: {VIDEO_ID}
- Duration: {DURATION}s
- Resolution: {RESOLUTION}
- Frame timestamps: {TIMESTAMPS}

## FLS Task 5 Rubric Summary:
- Max time: 600s (score = 600 - time - penalties)
- 3 throws required: first is surgeon's knot (double), throws 2-3 are single
- Hand must switch between throws
- Suture through marked points, drain slit must close
- Penalties: deviation from marks, gap in drain, insecure knot, avulsed drain

Follow the protocol in prompts/v001_task5_consensus.md. Output ONLY JSON.
```

If Round 1 agreement_score < 0.92, run Round 2:

```
This is Round 2 (Final Arbitration). Here is the Round 1 output:
{PASTE ROUND 1 JSON}

The following fields were flagged as needs_rebuttal:
{LIST FIELDS}

Make your final ruling on each disputed field. Output the definitive consensus JSON.
```

Save consensus as:
`memory/comparisons/{VIDEO_ID}_consensus_{TIMESTAMP}.json`

## Step 3: Update Learning Ledger

For each consensus generated, append to `memory/learning_ledger.jsonl`:

```json
{"timestamp": "...", "event_type": "consensus_generated", "data": {"video_id": "...", "score_id": "...", "agreement_score": 0.XX, "rounds": N, "fls_score": XXX}}
```

## Priority Order

Score these first (most recent, best recording quality):
1. V25, V26 (latest videos, phone-on-monitor)
2. V22, V23, V24 (Insignia monitor setup, black suture)
3. V21 (first Insignia session)
4. V17-V20 (mixed quality, fatigue sessions — lower priority)
5. postV10 (older)
