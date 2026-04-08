#!/usr/bin/env python3
"""Download LASANA bitstreams from the checked-in HAL manifest.

The live Opara item and the checked-in manifest currently expose task-level
archives (for example ``PegTransfer_left.zip``), not per-trial HEVC objects.
This downloader therefore preserves the source filename and extension from the
manifest instead of forcing a synthetic ``.hevc`` suffix.

Expected runtime target: the single Contabo host.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


LOG = logging.getLogger("lasana_download")
DEFAULT_MANIFEST = Path("data/external/lasana/_meta/bitstreams.json")
DEFAULT_MIN_FREE_GB = 500.0
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class BitstreamRecord:
    trial_id: str
    task: str | None
    filename: str
    url: str
    size_bytes: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LASANA video archives from HAL manifest")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where downloaded LASANA bitstreams will be written",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST),
        help="Path to the checked-in HAL bitstreams.json manifest",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Limit downloads to the first N matched bitstreams; 0 means all",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed files and resume from partial .part files when present",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional task filter (for example PegTransfer or SutureAndKnot)",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Manifest is not valid JSON: {path}: {exc}") from exc


def detect_token_requirement(payload: dict[str, Any]) -> bool:
    serialized = json.dumps(payload).lower()
    markers = (
        "lasana_api_token",
        "authorization",
        "bearer",
        "api-token",
        "api_key",
        "authn",
        "authz",
    )
    return any(marker in serialized for marker in markers)


def normalize_task_name(raw: str) -> str:
    return "".join(ch.lower() for ch in raw if ch.isalnum())


def infer_task(filename: str) -> str | None:
    stem = Path(filename).stem
    if "_" not in stem:
        return None
    prefix = stem.split("_", 1)[0]
    if prefix in {"PegTransfer", "CircleCutting", "BalloonResection", "SutureAndKnot"}:
        return prefix
    return None


def extract_bitstreams(payload: dict[str, Any], task_filter: str | None) -> list[BitstreamRecord]:
    raw_entries = payload.get("_embedded", {}).get("bitstreams")
    if not isinstance(raw_entries, list):
        raise SystemExit("Manifest does not contain _embedded.bitstreams")

    requested_task = normalize_task_name(task_filter) if task_filter else None
    records: list[BitstreamRecord] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get("name") or "").strip()
        if not filename:
            continue
        task = infer_task(filename)
        if requested_task and normalize_task_name(task or "") != requested_task:
            continue
        if task is None:
            continue

        url = (
            entry.get("_links", {})
            .get("content", {})
            .get("href")
        )
        if not isinstance(url, str) or not url:
            LOG.warning("Skipping %s because content href is missing", filename)
            continue

        size_bytes_raw = entry.get("sizeBytes")
        size_bytes = int(size_bytes_raw) if isinstance(size_bytes_raw, int) else None
        trial_id = Path(filename).stem
        records.append(
            BitstreamRecord(
                trial_id=trial_id,
                task=task,
                filename=filename,
                url=url,
                size_bytes=size_bytes,
            )
        )

    records.sort(key=lambda item: item.filename)
    return records


def ensure_free_disk(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(out_dir)
    free_gb = usage.free / (1024 ** 3)
    min_free_gb_raw = os.getenv("LASANA_MIN_FREE_GB", str(DEFAULT_MIN_FREE_GB)).strip()
    try:
        min_free_gb = float(min_free_gb_raw)
    except ValueError as exc:
        raise SystemExit(f"LASANA_MIN_FREE_GB must be numeric, got: {min_free_gb_raw!r}") from exc
    min_free_bytes = int(min_free_gb * 1024 ** 3)
    LOG.info("Disk free at %s: %.1f GiB", out_dir, free_gb)
    if usage.free < min_free_bytes:
        raise SystemExit(
            f"Refusing to start download with only {free_gb:.1f} GiB free at {out_dir}; "
            f"need at least {min_free_gb:.1f} GiB free"
        )


def auth_headers_from_manifest(payload: dict[str, Any]) -> dict[str, str]:
    requires_token = detect_token_requirement(payload)
    token = os.getenv("LASANA_API_TOKEN")
    if requires_token and not token:
        raise SystemExit("Manifest indicates authenticated access but LASANA_API_TOKEN is unset")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def read_existing_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["trial_id"]: row for row in rows if row.get("trial_id")}


def write_manifest(path: Path, rows: dict[str, dict[str, str]]) -> None:
    fieldnames = [
        "trial_id",
        "task",
        "filename",
        "url",
        "bytes",
        "sha256",
        "status",
        "local_path",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_request(url: str, headers: dict[str, str], offset: int = 0) -> request.Request:
    req = request.Request(url, headers=headers)
    if offset > 0:
        req.add_header("Range", f"bytes={offset}-")
    return req


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, error.HTTPError):
        return exc.code in RETRYABLE_STATUS_CODES
    return isinstance(exc, error.URLError)


def download_with_retries(
    record: BitstreamRecord,
    destination: Path,
    headers: dict[str, str],
    resume: bool,
    attempts: int = 5,
    timeout: int = 60,
) -> tuple[int, str]:
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if resume and partial.exists() else 0
        req = build_request(record.url, headers, offset=offset)
        mode = "ab" if offset > 0 else "wb"
        try:
            with request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if offset > 0 and status == 200:
                    LOG.warning("Server ignored Range for %s; restarting from byte 0", record.filename)
                    partial.unlink(missing_ok=True)
                    offset = 0
                    mode = "wb"
                with partial.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            partial.replace(destination)
            return destination.stat().st_size, sha256_file(destination)
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts or not is_retryable(exc):
                raise
            sleep_seconds = min(60, 2 ** (attempt - 1))
            LOG.warning(
                "Download failed for %s on attempt %d/%d: %s; retrying in %ss",
                record.filename,
                attempt,
                attempts,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"unreachable retry loop for {record.filename}")


def manifest_row(
    record: BitstreamRecord,
    *,
    destination: Path,
    status: str,
    bytes_written: int | None = None,
    sha256: str = "",
    error_text: str = "",
) -> dict[str, str]:
    return {
        "trial_id": record.trial_id,
        "task": record.task or "",
        "filename": record.filename,
        "url": record.url,
        "bytes": "" if bytes_written is None else str(bytes_written),
        "sha256": sha256,
        "status": status,
        "local_path": str(destination),
        "error": error_text,
    }


def run() -> int:
    args = parse_args()
    configure_logging()

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_payload = load_manifest(manifest_path)
    ensure_free_disk(out_dir)
    headers = auth_headers_from_manifest(manifest_payload)

    records = extract_bitstreams(manifest_payload, args.task)
    if args.max_trials > 0:
        records = records[: args.max_trials]
    if not records:
        LOG.warning("No LASANA bitstreams matched the current filters")

    status_path = out_dir / "manifest.csv"
    status_rows = read_existing_manifest(status_path)

    failures = 0
    for index, record in enumerate(records, start=1):
        destination = out_dir / record.filename
        LOG.info("[%d/%d] %s", index, len(records), record.filename)

        if args.resume and destination.exists():
            status_rows[record.trial_id] = manifest_row(
                record,
                destination=destination,
                status="skipped_existing",
                bytes_written=destination.stat().st_size,
            )
            write_manifest(status_path, status_rows)
            continue

        try:
            bytes_written, digest = download_with_retries(
                record,
                destination=destination,
                headers=headers,
                resume=args.resume,
            )
            status_rows[record.trial_id] = manifest_row(
                record,
                destination=destination,
                status="ok",
                bytes_written=bytes_written,
                sha256=digest,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            partial = destination.with_suffix(destination.suffix + ".part")
            bytes_written = partial.stat().st_size if partial.exists() else None
            status_rows[record.trial_id] = manifest_row(
                record,
                destination=destination,
                status="failed",
                bytes_written=bytes_written,
                error_text=str(exc),
            )
            LOG.error("Failed to download %s: %s", record.filename, exc)
        finally:
            write_manifest(status_path, status_rows)

    if failures:
        LOG.warning("Completed with %d failed download(s); see %s", failures, status_path)
        return 1
    LOG.info("Completed successfully; manifest written to %s", status_path)
    return 0


if __name__ == "__main__":
    sys.exit(run())