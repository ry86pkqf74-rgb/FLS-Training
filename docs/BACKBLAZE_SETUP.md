# Backblaze B2 setup for FLS-Training

> **Why:** Phase 1, 2, and 3 of the deployment plan run on different
> pods. We need a cheap, durable place to park ~10 GB of artifacts
> (frames, features, checkpoints) that survives any pod shutdown.
> Backblaze B2 is the cheapest object storage on the market for our
> volume; total monthly cost should be well under $1.
>
> **One-time setup, ~10 minutes.** Once done, every pod just calls
> `bash scripts/067_b2_sync.sh push-* / pull-*`.

## 1. Create the account (you, web browser, ~3 min)

1. Go to https://www.backblaze.com/cloud-storage
2. Click **Sign Up**. Use the same email you use for the rest of the
   project. The free tier covers 10 GB of storage and 1 GB/day of
   downloads — easily enough for everything in this plan.
3. Verify your email.

## 2. Create the bucket (web browser, ~1 min)

1. Sign in. Navigate to **B2 Cloud Storage** → **Buckets**.
2. Click **Create a Bucket**.
3. Settings:
   - **Bucket Unique Name:** `fls-checkpoints` (or pick a unique
     suffix if it's taken — e.g. `fls-checkpoints-glosser`)
   - **Files in Bucket are:** Private
   - **Default Encryption:** Enable (SSE-B2 is free)
   - **Object Lock:** Disable
4. Click **Create a Bucket**.
5. **If you had to use a different name**, set it as the env var:
   `export B2_BUCKET=fls-checkpoints-glosser` and add it to `.env`.

## 3. Create an application key (web browser, ~2 min)

1. **App Keys** → **Add a New Application Key**.
2. Settings:
   - **Name of Key:** `fls-training-pods`
   - **Allow access to Bucket(s):** select **only** `fls-checkpoints`
     (not "All"). Limits blast radius if the key leaks.
   - **Type of Access:** **Read and Write**
   - **Allow List All Bucket Names:** unchecked
   - **File name prefix:** leave blank
   - **Duration:** leave blank (no expiry)
3. Click **Create New Key**.
4. **Copy both values immediately** — Backblaze only shows the
   `applicationKey` once:
   - `keyID` (looks like `005abc...`)
   - `applicationKey` (looks like `K005...`)
5. Save them somewhere durable (1Password, keychain). You will need
   them on every pod.

## 4. Test from your laptop (terminal, ~2 min)

```bash
# Install the b2 CLI if not already
pip install b2

# Authorize once on this machine — interactive
b2 account authorize <KEY_ID> <APP_KEY>

# Verify
b2 account get
b2 bucket list   # should show fls-checkpoints

# Smoke test the helper
bash scripts/067_b2_sync.sh status
```

If `b2 bucket list` shows your bucket, the setup is done.

## 5. Add to .env (one line)

Add the keys to your `.env` (NOT `.env.example`, which is committed):

```
B2_APPLICATION_KEY_ID=005abc...
B2_APPLICATION_KEY=K005...
B2_BUCKET=fls-checkpoints
```

The pod bootstrap script (`scripts/runpod_smoke_test.sh`) will pick
these up from the env automatically.

## 6. Use it on a pod

On every fresh pod (Phase 1, 2, or 3):

```bash
pip install b2

# Either inject the keys via env (cleaner for CI):
export B2_APPLICATION_KEY_ID=005abc...
export B2_APPLICATION_KEY=K005...

# Or run the interactive command once per pod:
b2 account authorize $B2_APPLICATION_KEY_ID $B2_APPLICATION_KEY

# Then sync:
bash scripts/067_b2_sync.sh pull-features
# ... do work ...
bash scripts/067_b2_sync.sh push-checkpoint v4_run1
```

## Cost estimate (recurring)

| Item | Volume | Rate | $/month |
|---|---|---|---|
| Storage (~10 GB across all artifacts) | 10 GB | $0.006/GB | $0.06 |
| Downloads (1 sync per pod, ~3 pods/run, 5 GB each) | 15 GB | $0.01/GB | $0.15 |
| Class B/C transactions | minimal | minimal | $0.01 |
| **Total** | | | **~$0.22/month** |

The free tier (10 GB storage, 1 GB/day download) covers most of this
on its own. Realistic cost is $0–$1/month even during heavy training
weeks. Compared to RunPod network volumes ($17.50/mo for 250 GB),
B2 is roughly 80x cheaper for our use case.

## Failure modes

- **"Key not authorized for bucket"** → you forgot to scope the key
  to the specific bucket in step 3. Create a new key with the right
  scope; revoke the old one.
- **"Cap exceeded"** → daily download cap hit. Either wait until
  midnight UTC or upgrade past the free tier (you'll still pay
  pennies).
- **"Bucket not found"** → bucket name typo. `b2 bucket list` to
  see the actual name; update `B2_BUCKET` env var.
