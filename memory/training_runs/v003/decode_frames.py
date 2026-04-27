"""Decode memory/frames/<vid>/frames.json (base64 JPGs) -> /workspace/v003_frames/<vid>/frame_NNN.jpg"""
import json, base64, sys
from pathlib import Path

src_root = Path("/workspace/FLS-Training/memory/frames")
dst_root = Path("/workspace/v003_frames")
dst_root.mkdir(parents=True, exist_ok=True)

total_vids = 0
total_frames = 0
for vid_dir in sorted(src_root.iterdir()):
    if not vid_dir.is_dir():
        continue
    f = vid_dir / "frames.json"
    if not f.exists():
        continue
    try:
        d = json.loads(f.read_text())
    except Exception as e:
        print(f"skip {vid_dir.name}: {e}", file=sys.stderr)
        continue
    frames = d.get("frames_b64", [])
    if not frames:
        continue
    out_dir = dst_root / vid_dir.name
    out_dir.mkdir(exist_ok=True)
    for i, b64 in enumerate(frames):
        try:
            blob = base64.b64decode(b64)
        except Exception:
            continue
        (out_dir / f"frame_{i:03d}.jpg").write_bytes(blob)
        total_frames += 1
    total_vids += 1
print(f"decoded {total_vids} videos / {total_frames} frames -> {dst_root}")
