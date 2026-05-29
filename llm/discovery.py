"""Shared-filesystem rendezvous for the Phase-2 distributed LLM stack.

On SLURM there is no Redis and jobs land on arbitrary nodes, so the shared home
filesystem is the service registry. Three kinds of process find each other
through files under ``runtime/`` at the repo root (override with
``$GINLLM_RUNTIME_DIR``):

  runtime/master.json         the master writes ``{host, port, ...}``; the
                              training client reads it to find the master.
  runtime/workers/<id>.json   each worker writes ``{id, host, port, model, ts,
                              job}``; the master scans this dir, health-checks,
                              and load-balances over the live ones.

Writes are atomic (tmp file + ``os.replace``) so a reader never observes a
half-written file. ``ts`` is a wall-clock heartbeat the master uses to drop
workers that vanished without cleaning up (e.g. preempted array tasks).
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runtime_dir() -> str:
    d = os.environ.get("GINLLM_RUNTIME_DIR", os.path.join(_REPO_ROOT, "runtime"))
    os.makedirs(d, exist_ok=True)
    return d


def worker_dir() -> str:
    d = os.path.join(runtime_dir(), "workers")
    os.makedirs(d, exist_ok=True)
    return d


def master_path() -> str:
    return os.path.join(runtime_dir(), "master.json")


def worker_path(worker_id: str) -> str:
    return os.path.join(worker_dir(), f"{worker_id}.json")


def this_host() -> str:
    """Hostname other cluster nodes can connect to (short name resolves on CARC)."""
    return socket.gethostname()


def default_worker_id(port: int) -> str:
    """Stable per-worker id: SLURM array identity when present, else host+pid."""
    job = os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job:
        return f"{job}_{task}" if task is not None else f"{job}"
    return f"{this_host()}_{os.getpid()}_{port}"


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None


# --- master ----------------------------------------------------------------
def write_master(host: str, port: int, **extra: Any) -> str:
    info = {"host": host, "port": port, "ts": time.time(), **extra}
    _atomic_write_json(master_path(), info)
    return master_path()


def read_master() -> Optional[Dict[str, Any]]:
    return _read_json(master_path())


def master_url() -> Optional[str]:
    m = read_master()
    return f"http://{m['host']}:{m['port']}" if m else None


# --- workers ----------------------------------------------------------------
def write_worker(worker_id: str, host: str, port: int, model: str,
                 **extra: Any) -> str:
    info = {
        "id": worker_id, "host": host, "port": port, "model": model,
        "ts": time.time(), "job": os.environ.get("SLURM_JOB_ID", ""), **extra,
    }
    _atomic_write_json(worker_path(worker_id), info)
    return worker_path(worker_id)


def heartbeat_worker(worker_id: str) -> None:
    """Refresh just the ts of an already-registered worker file."""
    info = _read_json(worker_path(worker_id))
    if info is not None:
        info["ts"] = time.time()
        _atomic_write_json(worker_path(worker_id), info)


def remove_worker(worker_id: str) -> None:
    try:
        os.remove(worker_path(worker_id))
    except FileNotFoundError:
        pass


def _cli() -> None:
    """`python -m llm.discovery` prints the master URL (for the training launcher
    to do: export GINLLM_MASTER_URL=$(python -m llm.discovery)). With `workers`,
    prints the live worker registry as JSON."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "workers":
        print(json.dumps(list_workers(), indent=2))
        return
    url = master_url()
    if url is None:
        sys.stderr.write("no master registered in %s\n" % master_path())
        sys.exit(1)
    print(url)


def list_workers(max_age: Optional[float] = None) -> List[Dict[str, Any]]:
    """Return all registered workers, optionally dropping ones whose heartbeat
    is older than ``max_age`` seconds."""
    out: List[Dict[str, Any]] = []
    now = time.time()
    wd = worker_dir()
    for name in os.listdir(wd):
        if not name.endswith(".json"):
            continue
        info = _read_json(os.path.join(wd, name))
        if info is None:
            continue
        if max_age is not None and (now - info.get("ts", 0)) > max_age:
            continue
        out.append(info)
    return out


if __name__ == "__main__":
    _cli()
