#!/usr/bin/env python3
"""Launch (or reuse) a RunPod H100 SXM 80GB pod for the v003 training run.

Uses the RunPod **REST API** (``https://rest.runpod.io/v1``) because the legacy
GraphQL endpoint is currently Cloudflare-blocked for direct API access.

Reads the API key from one of:

    1. $RUNPOD_API_KEY  (preferred, never persisted)
    2. ~/.runpod/api_key (single line, chmod 600)

Subcommands::

    python deploy/runpod_v003_launch.py launch      # creates a new pod
    python deploy/runpod_v003_launch.py status      # prints pod + SSH info
    python deploy/runpod_v003_launch.py terminate   # stops + deletes the pod
    python deploy/runpod_v003_launch.py gpu-types   # lists in-stock GPUs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://rest.runpod.io/v1"
STATE_FILE = Path(__file__).resolve().parent / ".runpod_v003_pod.json"

POD_NAME = "fls-v003-training"
# RunPod REST API uses NVIDIA SKU names as GPU type IDs (no separate listing
# endpoint). H100 SXM is "NVIDIA H100 80GB HBM3"; H200 SXM is "NVIDIA H200".
GPU_PREFERENCES = [
    "NVIDIA H200",
    "NVIDIA H100 80GB HBM3",  # H100 SXM
    "NVIDIA H100 NVL",
    "NVIDIA H100 PCIe",
    "NVIDIA B200",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
VOLUME_GB = 200
CONTAINER_GB = 80
MIN_VCPU = 16
MIN_RAM_GB = 100


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key.strip()
    keyfile = Path.home() / ".runpod" / "api_key"
    if keyfile.exists():
        return keyfile.read_text().strip()
    raise SystemExit(
        "RUNPOD_API_KEY is not set. Either:\n"
        "  export RUNPOD_API_KEY='rpa_xxx'\n"
        "or write the key to ~/.runpod/api_key (one line, chmod 600)."
    )


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Accept": "application/json",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace") if exc.fp else ""
        raise SystemExit(f"RunPod API HTTP {exc.code} {method} {path}: {detail or '<empty>'}")


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def cmd_gpu_types() -> None:
    """Print the static GPU preference list (REST API has no listing endpoint).

    The OpenAPI ``/openapi.json`` enumerates valid IDs in the
    ``PodCreateInput.gpuTypeIds`` enum; we just echo our preferred order here.
    """
    print(json.dumps(GPU_PREFERENCES, indent=2))


def cmd_launch() -> None:
    state = _load_state()
    if state.get("pod_id"):
        print(f"Pod already tracked: {state['pod_id']} ({state.get('status')})")
        print("Run `terminate` first to start a new one.")
        return

    print(f"GPU preference order (RunPod will pick first available):")
    for g in GPU_PREFERENCES:
        print(f"  - {g}")

    body = {
        "name": POD_NAME,
        "imageName": IMAGE,
        "cloudType": "SECURE",
        "gpuCount": 1,
        "gpuTypeIds": GPU_PREFERENCES,
        "gpuTypePriority": "custom",
        "containerDiskInGb": CONTAINER_GB,
        "volumeInGb": VOLUME_GB,
        "volumeMountPath": "/workspace",
        "vcpuCount": MIN_VCPU,
        "minRAMPerGPU": MIN_RAM_GB,
        "ports": ["22/tcp", "8888/http"],
        "supportPublicIp": True,
        "interruptible": False,
        "env": {"PYTHONUNBUFFERED": "1"},
    }
    pod = _request("POST", "/pods", body)
    if not isinstance(pod, dict) or "id" not in pod:
        raise SystemExit(f"Unexpected create-pod response: {json.dumps(pod, indent=2)[:600]}")

    state = {
        "pod_id": pod["id"],
        "gpu_assigned": pod.get("machineId") or pod.get("gpuTypeId"),
        "gpu_preferences": GPU_PREFERENCES,
        "image": IMAGE,
        "launched_at": time.time(),
    }
    _save_state(state)
    print(f"Launched pod: {pod['id']}")
    cmd_status(wait_for_ssh=True)


def _query_pod(pod_id: str) -> dict[str, Any]:
    pod = _request("GET", f"/pods/{pod_id}")
    if not isinstance(pod, dict):
        raise SystemExit(f"Unexpected pod payload: {pod!r}")
    return pod


def _ssh_endpoint(pod: dict[str, Any]) -> dict[str, Any] | None:
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or pod.get("portMappings") or []
    for p in ports:
        # REST schema sometimes returns dicts, sometimes "22/tcp -> ip:port" strings.
        if isinstance(p, dict):
            if int(p.get("privatePort") or 0) == 22 and (p.get("isIpPublic") or p.get("publicPort")):
                return {
                    "ip": p.get("ip") or pod.get("publicIp"),
                    "publicPort": p.get("publicPort"),
                }
        elif isinstance(p, str) and p.startswith("22/tcp"):
            mapping = p.split("->")[-1].strip()
            if ":" in mapping:
                ip, port = mapping.rsplit(":", 1)
                return {"ip": ip.strip(), "publicPort": int(port)}
    return None


def cmd_status(wait_for_ssh: bool = False) -> None:
    state = _load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod tracked. Run `launch` first.")
        return

    deadline = time.time() + (10 * 60 if wait_for_ssh else 0)
    while True:
        pod = _query_pod(pod_id)
        ssh = _ssh_endpoint(pod)
        state.update({
            "status": pod.get("desiredStatus") or pod.get("status"),
            "uptime_seconds": (pod.get("runtime") or {}).get("uptimeInSeconds"),
            "ssh": ssh,
        })
        _save_state(state)

        printable = {k: v for k, v in state.items() if k != "ssh"}
        print(json.dumps(printable, indent=2))
        if ssh and ssh.get("publicPort"):
            print("\nSSH command:")
            print(f"  ssh root@{ssh['ip']} -p {ssh['publicPort']} -i ~/.ssh/id_ed25519")
            return
        if not wait_for_ssh or time.time() >= deadline:
            print("(SSH not ready yet — try `status` again in 30s)")
            return
        print("Waiting 15s for SSH...")
        time.sleep(15)


def cmd_terminate() -> None:
    state = _load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod tracked.")
        return
    _request("DELETE", f"/pods/{pod_id}")
    print(f"Terminated pod {pod_id}")
    STATE_FILE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("launch", help="Create a new H100 pod for v003 training.")
    sub.add_parser("status", help="Show pod state + SSH endpoint.")
    sub.add_parser("terminate", help="Stop and delete the tracked pod.")
    sub.add_parser("gpu-types", help="List GPU types + availability.")
    args = parser.parse_args()

    {
        "launch": cmd_launch,
        "status": cmd_status,
        "terminate": cmd_terminate,
        "gpu-types": cmd_gpu_types,
    }[args.cmd]()


if __name__ == "__main__":
    main()
