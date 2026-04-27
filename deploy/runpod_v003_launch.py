#!/usr/bin/env python3
"""Launch (or reuse) a RunPod H100 SXM 80GB pod for the v003 training run.

Reads the RunPod API key from one of:

    1. $RUNPOD_API_KEY  (preferred, never persisted)
    2. ~/.runpod/api_key (single line)

Usage::

    export RUNPOD_API_KEY='rpa_xxx'
    python deploy/runpod_v003_launch.py launch       # creates a new pod
    python deploy/runpod_v003_launch.py status       # prints pod + SSH info
    python deploy/runpod_v003_launch.py terminate    # stops + deletes the pod

Pod spec:

    GPU       : H100 SXM 80GB (1x), fall back to H200 141GB if unavailable.
    Image     : runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
    Volume    : 200 GB at /workspace
    Container : 80 GB
    Ports     : 22/tcp (SSH), 8888/tcp (Jupyter, optional)

The script writes the latest pod metadata to ``deploy/.runpod_v003_pod.json``
so subsequent ``status`` / ``terminate`` calls don't need re-discovery.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = "https://api.runpod.io/graphql"
STATE_FILE = Path(__file__).resolve().parent / ".runpod_v003_pod.json"

POD_NAME = "fls-v003-training"
GPU_PREFERENCES = ["H100 SXM", "H100 PCIe", "H200", "A100 SXM"]
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


def _gql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"RunPod API HTTP {exc.code}: {exc.read().decode(errors='replace')}")
    if body.get("errors"):
        raise SystemExit(f"RunPod GraphQL error: {json.dumps(body['errors'], indent=2)}")
    return body["data"]


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _find_gpu_type() -> dict[str, Any]:
    """Pick the first preferred GPU type that has stock and supports the spec."""
    data = _gql("""
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice(input: {gpuCount: 1}) {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
    """)
    by_name = {gt["displayName"]: gt for gt in data["gpuTypes"]}
    for pref in GPU_PREFERENCES:
        candidates = [gt for name, gt in by_name.items() if pref in name]
        for gt in sorted(candidates, key=lambda g: g["memoryInGb"], reverse=True):
            price = gt.get("lowestPrice") or {}
            if price.get("uninterruptablePrice"):
                return gt
    raise SystemExit(
        "No GPU stock for any of: " + ", ".join(GPU_PREFERENCES)
        + "\nRetry in a few minutes or pick a different region."
    )


def cmd_launch() -> None:
    state = _load_state()
    if state.get("pod_id"):
        print(f"Pod already tracked: {state['pod_id']} ({state.get('status')})")
        print("Run `terminate` first to start a new one.")
        return

    gpu = _find_gpu_type()
    print(f"Selected GPU: {gpu['displayName']} ({gpu['memoryInGb']} GB)")

    mutation = """
        mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                imageName
                machineId
                desiredStatus
            }
        }
    """
    variables = {
        "input": {
            "cloudType": "SECURE",
            "gpuCount": 1,
            "gpuTypeId": gpu["id"],
            "name": POD_NAME,
            "imageName": IMAGE,
            "containerDiskInGb": CONTAINER_GB,
            "volumeInGb": VOLUME_GB,
            "volumeMountPath": "/workspace",
            "minVcpuCount": MIN_VCPU,
            "minMemoryInGb": MIN_RAM_GB,
            "ports": "22/tcp,8888/http",
            "supportPublicIp": True,
            "startSsh": True,
            "env": [
                {"key": "PYTHONUNBUFFERED", "value": "1"},
            ],
        }
    }
    data = _gql(mutation, variables)
    pod = data["podFindAndDeployOnDemand"]
    print(f"Launched pod: {pod['id']}")
    state = {"pod_id": pod["id"], "gpu": gpu["displayName"], "image": IMAGE, "launched_at": time.time()}
    _save_state(state)
    print("Polling for SSH endpoint...")
    cmd_status(wait_for_ssh=True)


def _query_pod(pod_id: str) -> dict[str, Any]:
    data = _gql(
        """
        query Pod($id: String!) {
            pod(input: {podId: $id}) {
                id
                desiredStatus
                lastStatusChange
                machineId
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        privatePort
                        publicPort
                        type
                        isIpPublic
                    }
                }
            }
        }
        """,
        {"id": pod_id},
    )
    return data["pod"]


def cmd_status(wait_for_ssh: bool = False) -> None:
    state = _load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod tracked. Run `launch` first.")
        return

    deadline = time.time() + (10 * 60 if wait_for_ssh else 0)
    while True:
        pod = _query_pod(pod_id)
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []
        ssh_port = next(
            (p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")),
            None,
        )
        state.update(
            {
                "status": pod["desiredStatus"],
                "uptime_seconds": runtime.get("uptimeInSeconds"),
                "ssh": ssh_port,
            }
        )
        _save_state(state)

        print(json.dumps({k: v for k, v in state.items() if k != "ssh"}, indent=2))
        if ssh_port:
            print("\nSSH command:")
            print(f"  ssh root@{ssh_port['ip']} -p {ssh_port['publicPort']} -i ~/.ssh/id_ed25519")
            return
        if not wait_for_ssh or time.time() >= deadline:
            print("(SSH not ready yet)")
            return
        print("Waiting 15s for SSH...")
        time.sleep(15)


def cmd_terminate() -> None:
    state = _load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod tracked.")
        return
    _gql(
        """
        mutation Terminate($id: String!) {
            podTerminate(input: {podId: $id})
        }
        """,
        {"id": pod_id},
    )
    print(f"Terminated pod {pod_id}")
    STATE_FILE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("launch", help="Create a new H100 pod for v003 training.")
    sub.add_parser("status", help="Show pod state + SSH endpoint.")
    sub.add_parser("terminate", help="Stop and delete the tracked pod.")
    args = parser.parse_args()

    {
        "launch": cmd_launch,
        "status": cmd_status,
        "terminate": cmd_terminate,
    }[args.cmd]()


if __name__ == "__main__":
    main()
