"""Load-test the API in Docker with real test-period transactions.

Starts the `overseerml` image (which serves models/production.json), sends
WARMUP sequential requests, then REQUESTS POST /predict calls from CONCURRENCY
threads, and writes reports/loadtest.json with:

- client latency: full HTTP round trip measured by this script
- server latency: what the API logs to SQLite, from building the feature row to
  the finished response (scoring + SHAP); it excludes HTTP parsing, input
  validation and the SQLite write itself
- the hardware and Docker limits it ran on

Any failed request raises and stops the test. Client and server share one
machine, so they compete for the same CPU.
Percentiles use numpy's default linear interpolation.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from cryptoguard.data import FEATURE_COLUMNS, load_elliptic
from cryptoguard.features import VAL_END

from overseerml.train import DATA_DIR, ROOT

IMAGE, CONTAINER, PORT = "overseerml", "overseer-loadtest", 8000
REQUESTS, WARMUP, CONCURRENCY = 2000, 50, 8
SEED = 42


def powershell(command: str) -> str:
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True
    ).stdout.strip()


def host() -> dict:
    info = {"os": platform.platform(), "logical_cpus": os.cpu_count()}
    if platform.system() == "Windows":
        info["cpu"] = powershell("(Get-CimInstance Win32_Processor).Name")
        info["ram_bytes"] = int(
            powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
        )
    else:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        info["cpu"] = next(
            ln.split(":", 1)[1].strip()
            for ln in cpuinfo.splitlines()
            if ln.startswith("model name")
        )
        info["ram_bytes"] = int(Path("/proc/meminfo").read_text().split()[1]) * 1024
    docker = json.loads(
        subprocess.run(
            ["docker", "info", "--format", "{{json .}}"], capture_output=True, text=True, check=True
        ).stdout
    )
    info["docker"] = {
        k: docker[k] for k in ("ServerVersion", "OperatingSystem", "NCPU", "MemTotal")
    }
    return info


def percentiles(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "max": float(a.max()),
    }


def main() -> None:
    df = load_elliptic(DATA_DIR)
    test = df[df["time_step"] > VAL_END]
    rows = test[FEATURE_COLUMNS].sample(WARMUP + REQUESTS, random_state=SEED).to_numpy()
    bodies = [dict(zip(FEATURE_COLUMNS, map(float, r), strict=True)) for r in rows]

    logs = Path(tempfile.mkdtemp(prefix="overseer-loadtest-"))
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            CONTAINER,
            "-p",
            f"{PORT}:8000",
            "-v",
            f"{logs}:/app/logs",
            IMAGE,
        ],
        check=True,
        capture_output=True,
    )
    try:
        url = f"http://localhost:{PORT}"
        with httpx.Client(timeout=30) as client:
            for _ in range(60):
                try:
                    health = client.get(f"{url}/health").json()
                    break
                except httpx.TransportError:
                    time.sleep(1)
            else:
                raise RuntimeError("API did not become healthy within 60 s")

            def call(body: dict) -> float:
                start = time.perf_counter()
                response = client.post(f"{url}/predict", json=body)
                elapsed = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                return elapsed

            for body in bodies[:WARMUP]:
                call(body)
            start = time.perf_counter()
            with ThreadPoolExecutor(CONCURRENCY) as pool:
                client_ms = list(pool.map(call, bodies[WARMUP:]))
            wall_s = time.perf_counter() - start
    finally:
        subprocess.run(["docker", "stop", CONTAINER], capture_output=True)

    with sqlite3.connect(logs / "requests.db") as db:
        logged = [r[0] for r in db.execute("SELECT latency_ms FROM requests ORDER BY ts")]
    # Warm-up requests were sent first and one at a time, so they are the first rows.
    server_ms = logged[WARMUP:]

    result = {
        "model_version": health["model_version"],
        "requests": len(client_ms),
        "warmup_requests": WARMUP,
        "concurrency": CONCURRENCY,
        "server_logged_requests": len(server_ms),
        "wall_seconds": wall_s,
        "throughput_rps": len(client_ms) / wall_s,
        "client_latency_ms": percentiles(client_ms),
        "server_latency_ms": percentiles(server_ms),
        "host": host(),
    }
    out = ROOT / "reports" / "loadtest.json"
    out.write_text(json.dumps(result, indent=2) + "\n", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
