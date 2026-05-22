"""Phase 5.3 benchmark harness for VibeProteinAgent.

Measures the framework's central performance bet — persistent workers + warm
pool — versus a naive fresh-subprocess baseline.

Sub-commands (each accepts ``--json`` and emits one JSON blob to stdout):

  - noop_warm_call            — N warm noop calls; per-call wall-clock stats.
  - noop_throughput           — Submit N noop jobs concurrently; jobs/sec.
  - proteinmpnn_cold_vs_warm  — Cold + warm ProteinMPNN end-to-end timings.
  - protenix_cold_vs_warm     — Cold + warm Protenix-v2 end-to-end timings.
  - vs_naive_proteinmpnn      — Warm vpa vs fresh ``python protein_mpnn_run.py``.
  - vs_naive_protenix         — Warm vpa vs fresh ``python runner/inference.py``.
  - multi_gpu_dispatch        — Throughput as we go from 1 → N workers.
  - workflow_cache_hit        — 3-noop workflow first vs second-run wall.
  - all                       — Run the suite; emit one aggregated JSON blob.

Each sub-command that needs workers boots a supervisor subprocess and tears it
down when done. ``all`` runs those sub-benchmarks sequentially and gives each
one an isolated supervisor lifecycle.

Run via:
    python scripts/benchmark.py <sub-command> [--json] [opts]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# We may be run as a script — make the package importable from a source checkout.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if (_SRC / "vibeproteinagent").is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


_REPO_FIXTURES = _REPO / "tests" / "fixtures"
_PROTEINMPNN_PDB = _REPO_FIXTURES / "6MRR.pdb"
_PROTENIX_INPUT = _REPO_FIXTURES / "protenix_v2_input_tiny.json"
_NAIVE_MPNN_RUNNER = Path("/tmp/proteinmpnn_upstream/protein_mpnn_run.py")
_NAIVE_MPNN_WEIGHTS = Path("/tmp/proteinmpnn_upstream/vanilla_model_weights")
_NAIVE_PROTENIX_RUNNER = Path("/opt/vpa-data/Protenix/runner/inference.py")
_NAIVE_PROTENIX_CKPT = Path(
    "/opt/vpa-data/protenix_ckpt/protenix_base_default_v1.0.0.pt"
)


# ---------------------------------------------------------------- env capture


def env_block() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        info["n_gpus"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_err"] = str(e)
    try:
        from vibeproteinagent import __version__ as v  # type: ignore[attr-defined]
        info["vpa"] = v
    except Exception:
        try:
            from vibeproteinagent._version import __version__ as v
            info["vpa"] = v
        except Exception:
            info["vpa"] = "unknown"
    return info


def provenance_block(argv: list[str] | None = None) -> dict[str, Any]:
    """Capture enough context to decide whether a benchmark JSON is canonical."""
    def _git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=_REPO,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        except Exception:
            return None

    status = _git(["status", "--short"])
    return {
        "hostname": socket.gethostname(),
        "argv": list(sys.argv if argv is None else argv),
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status),
        "git_status_short": status,
    }


# ---------------------------------------------------------------- stats


def stats_from_samples(samples_ms: list[float]) -> dict[str, float]:
    """Compute min/p50/p95/p99/max for a list of millisecond samples."""
    if not samples_ms:
        return {"n": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    xs = sorted(samples_ms)
    n = len(xs)

    def pct(p: float) -> float:
        if n == 1:
            return xs[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        if c == f:
            return xs[f]
        return xs[f] + (xs[c] - xs[f]) * (k - f)

    return {
        "n": n,
        "min": xs[0],
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": xs[-1],
        "mean": statistics.fmean(xs),
    }


_TERMINAL_STATES = {"completed", "failed", "cancelled"}


def _job_state_value(record: Any) -> str:
    state = record.state
    return str(getattr(state, "value", state))


def summarize_job_completion(
    records: list[Any],
    *,
    submitted_jobs: int,
    wall_clock_s: float,
) -> dict[str, Any]:
    """Summarise completed-job throughput without crediting unfinished jobs."""
    state_counts: dict[str, int] = {}
    completed_ms: list[float] = []
    for record in records:
        state = _job_state_value(record)
        state_counts[state] = state_counts.get(state, 0) + 1
        if state != "completed":
            continue
        started_at = getattr(record, "started_at", None)
        finished_at = getattr(record, "finished_at", None)
        if finished_at is not None and started_at is not None:
            completed_ms.append((finished_at - started_at) / 1000.0)

    completed_jobs = state_counts.get("completed", 0)
    terminal_jobs = sum(state_counts.get(state, 0) for state in _TERMINAL_STATES)
    throughput = completed_jobs / wall_clock_s if wall_clock_s > 0 else 0.0
    return {
        "submitted_jobs": submitted_jobs,
        "completed_jobs": completed_jobs,
        "terminal_jobs": terminal_jobs,
        "all_completed": completed_jobs == submitted_jobs,
        "all_terminal": terminal_jobs == submitted_jobs,
        "state_counts": state_counts,
        "throughput_per_s": throughput,
        "per_job_ms": stats_from_samples(completed_ms),
    }


# ---------------------------------------------------------------- supervisor bring-up


@dataclass
class DaemonHandle:
    proc: mp.Process
    runtime_dir: Path
    home_dir: Path
    workspace: str
    socket_path: Path
    gpu_ids: list[int]
    teardown_dirs: bool = False


def _supervisor_target(
    workspace_name: str,
    runtime_dir_s: str,
    home_dir_s: str,
    num_workers: int,
    gpu_ids: list[int],
) -> None:
    """Run in the spawned process. Mirrors the e2e tests' supervisor target."""
    os.environ["VPA_RUNTIME_DIR"] = runtime_dir_s
    os.environ["VPA_HOME"] = home_dir_s
    if all(g < 0 for g in gpu_ids):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(g) for g in gpu_ids if g >= 0
        )
    from vibeproteinagent.workers.supervisor import supervisor_main

    supervisor_main(workspace_name, num_workers, gpu_ids)


def _wait_socket(socket_path: Path, *, timeout_s: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if socket_path.exists():
            return True
        time.sleep(0.1)
    return False


def _stop_proc(proc: mp.Process, *, kill_timeout_s: float = 15.0) -> None:
    if not proc.is_alive():
        return
    try:
        proc.terminate()
        proc.join(timeout=kill_timeout_s)
    except Exception:
        pass
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5.0)


def start_supervisor(
    *,
    workspace_name: str = "bench",
    num_workers: int | None = None,
    gpu_ids: list[int] | None = None,
    base_tmp: Path | None = None,
    teardown_dirs: bool = True,
) -> DaemonHandle:
    """Spawn a supervisor + workers in a subprocess. Returns the handle."""
    if gpu_ids is None:
        try:
            import torch
            if torch.cuda.is_available():
                k = min(torch.cuda.device_count(), 8)
                gpu_ids = list(range(k))
            else:
                gpu_ids = [-1]
        except Exception:
            gpu_ids = [-1]
    if num_workers is None:
        num_workers = len(gpu_ids)
    assert len(gpu_ids) == num_workers, "num_workers must match len(gpu_ids)"

    if base_tmp is None:
        base_tmp = Path(tempfile.mkdtemp(prefix="vpa_bench_"))
    base_tmp.mkdir(parents=True, exist_ok=True)
    runtime_dir = base_tmp / "run"
    home_dir = base_tmp / "home"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)

    # Create the workspace BEFORE the supervisor starts so JobStore.submit()
    # has somewhere to land jobs.
    os.environ["VPA_HOME"] = str(home_dir)
    os.environ["VPA_RUNTIME_DIR"] = str(runtime_dir)
    os.environ.pop("VPA_WORKSPACE", None)
    from vibeproteinagent.workspace import Workspace

    try:
        Workspace.create(workspace_name)
    except Exception:
        # Already exists — fine.
        pass

    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_target,
        args=(workspace_name, str(runtime_dir), str(home_dir), num_workers, gpu_ids),
        name=f"vpa-bench-supervisor-{workspace_name}",
        daemon=False,
    )
    proc.start()

    socket_path = runtime_dir / "supervisor.sock"
    if not _wait_socket(socket_path, timeout_s=60.0):
        _stop_proc(proc)
        if teardown_dirs:
            shutil.rmtree(base_tmp, ignore_errors=True)
        raise RuntimeError(
            f"supervisor failed to come up at {socket_path} within 60s"
        )

    return DaemonHandle(
        proc=proc,
        runtime_dir=runtime_dir,
        home_dir=home_dir,
        workspace=workspace_name,
        socket_path=socket_path,
        gpu_ids=list(gpu_ids),
        teardown_dirs=teardown_dirs,
    )


def stop_supervisor(handle: DaemonHandle) -> None:
    _stop_proc(handle.proc)
    if handle.teardown_dirs:
        shutil.rmtree(handle.home_dir.parent, ignore_errors=True)


@contextlib.contextmanager
def daemon(
    *,
    workspace_name: str = "bench",
    num_workers: int | None = None,
    gpu_ids: list[int] | None = None,
):
    """Context-manager-style supervisor lifecycle."""
    handle = start_supervisor(
        workspace_name=workspace_name,
        num_workers=num_workers,
        gpu_ids=gpu_ids,
    )
    try:
        yield handle
    finally:
        stop_supervisor(handle)


# ---------------------------------------------------------------- submit + wait helpers


def _wait_terminal(store: Any, job_id: str, *, timeout_s: float) -> Any:
    """Poll a JobStore until the job reaches a terminal state."""

    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        cur = store.get(job_id)
        last = cur.state
        if cur.state.value in ("completed", "failed", "cancelled"):
            return cur
        time.sleep(0.05)
    raise TimeoutError(
        f"job {job_id} never reached terminal state within {timeout_s}s "
        f"(last={last})"
    )


def _submit_and_wait(
    handle: DaemonHandle,
    tool: str,
    version: str,
    params: dict[str, Any],
    *,
    timeout_s: float,
    force_new: bool = False,
) -> tuple[float, Any]:
    """Submit one job; return (wall_clock_seconds, final JobRecord).

    Phase 20: when ``force_new`` is set we pass a fresh random salt so the
    deterministic ``job_id`` (Phase 10 errata 2) doesn't collide on
    repeated submissions of the same params — every cold/warm iteration
    is a genuine fresh job.
    """
    import secrets

    from vibeproteinagent.workspace import JobState, JobStore, Workspace

    ws = Workspace.resolve(handle.workspace)
    store = JobStore(ws)

    salt = secrets.token_hex(8) if force_new else None
    t0 = time.time()
    rec = store.submit(tool, version, params, workspace_name=ws.name, salt=salt)
    store.transition(rec.job_id, JobState.QUEUED)
    final = _wait_terminal(store, rec.job_id, timeout_s=timeout_s)
    elapsed = time.time() - t0
    return elapsed, final


def _open_store(handle: DaemonHandle):
    from vibeproteinagent.workspace import JobStore, Workspace

    ws = Workspace.resolve(handle.workspace)
    return JobStore(ws), ws


# ---------------------------------------------------------------- sub-benchmarks


def bench_noop_warm_call(iterations: int = 50) -> dict[str, Any]:
    """N warm noop submissions; per-call wall-clock stats."""
    from vibeproteinagent.workers.supervisor_client import SupervisorClient
    from vibeproteinagent.workspace import JobState

    with daemon() as h:
        client = SupervisorClient(socket_path=h.socket_path)
        # Warm noop everywhere.
        client.warm("noop", version_spec="*", timeout_s=60.0)

        store, ws = _open_store(h)
        # Throwaway warm-up rounds: drive at least one call onto each worker so
        # the per-worker first-call penalty (kernel JIT, lazy adapter load on
        # whichever worker the warm RPC didn't visit) is paid before we start
        # the measurement window. Discarded from samples.
        for i in range(len(h.gpu_ids)):
            rec = store.submit("noop", "1.0.0", {"payload": f"warmup-{i}"},
                               workspace_name=ws.name)
            store.transition(rec.job_id, JobState.QUEUED)
            _wait_terminal(store, rec.job_id, timeout_s=30.0)

        samples_ms: list[float] = []
        wall_t0 = time.time()
        for i in range(iterations):
            t0 = time.time()
            rec = store.submit("noop", "1.0.0", {"payload": f"warm-{i}"},
                               workspace_name=ws.name)
            store.transition(rec.job_id, JobState.QUEUED)
            _wait_terminal(store, rec.job_id, timeout_s=30.0)
            samples_ms.append((time.time() - t0) * 1000.0)
        wall = time.time() - wall_t0

    return {
        "benchmark": "noop_warm_call",
        "config": {
            "tool": "noop",
            "iterations": iterations,
            "warmed": True,
        },
        "wall_clock_s": wall,
        "per_job_ms": stats_from_samples(samples_ms),
        "samples_ms": samples_ms,
        "env": env_block(),
        "ts": time.time(),
    }


def bench_noop_throughput(jobs: int = 100) -> dict[str, Any]:
    """Submit N noop jobs concurrently; report throughput + per-job stats."""
    from vibeproteinagent.workers.supervisor_client import SupervisorClient
    from vibeproteinagent.workspace import JobState

    with daemon() as h:
        client = SupervisorClient(socket_path=h.socket_path)
        client.warm("noop", version_spec="*", timeout_s=60.0)

        store, ws = _open_store(h)
        wall_t0 = time.time()
        ids: list[str] = []
        for i in range(jobs):
            rec = store.submit("noop", "1.0.0", {"payload": f"thr-{i}"},
                               workspace_name=ws.name)
            store.transition(rec.job_id, JobState.QUEUED)
            ids.append(rec.job_id)
        # Wait until all terminate.
        deadline = time.time() + 120.0
        while time.time() < deadline:
            recs = [store.get(j) for j in ids]
            states = [r.state.value for r in recs]
            if all(s in ("completed", "failed", "cancelled") for s in states):
                break
            time.sleep(0.05)
        wall = time.time() - wall_t0

        recs = [store.get(j) for j in ids]
        n_workers = len(h.gpu_ids)
        completion = summarize_job_completion(
            recs,
            submitted_jobs=jobs,
            wall_clock_s=wall,
        )

    return {
        "benchmark": "noop_throughput",
        "config": {
            "tool": "noop", "jobs": jobs, "workers": n_workers, "warmed": True,
        },
        "wall_clock_s": wall,
        **completion,
        "env": env_block(),
        "ts": time.time(),
    }


def bench_proteinmpnn_cold_vs_warm(
    pdb_path: Path | None = None, warm_iters: int = 5
) -> dict[str, Any]:
    """Time the first ProteinMPNN call (cold) and ``warm_iters`` warm calls."""
    from vibeproteinagent.workers.supervisor_client import SupervisorClient

    fixture = Path(pdb_path) if pdb_path else _PROTEINMPNN_PDB
    assert fixture.exists(), f"missing fixture {fixture}"

    cold_warm_load_s: float = 0.0
    cold_first_call_s: float = 0.0
    warm_samples_s: list[float] = []

    with daemon() as h:
        client = SupervisorClient(socket_path=h.socket_path)
        # Cold warm of the adapter (drives __init__ + ckpt load).
        t0 = time.time()
        client.warm("proteinmpnn", version_spec="*", timeout_s=180.0)
        cold_warm_load_s = time.time() - t0

        params = {
            "pdb_path": str(fixture),
            "num_designs": 4,
            "temperature": 0.1,
            "seed": 17,
        }
        # First post-warm call: still pays CUDA kernel JIT.
        elapsed, rec = _submit_and_wait(h, "proteinmpnn", "1.0.1", params,
                                        timeout_s=180.0, force_new=True)
        cold_first_call_s = elapsed
        assert rec.state.value == "completed", f"first cold-mpnn job failed: {rec}"

        for _ in range(warm_iters):
            elapsed, rec = _submit_and_wait(h, "proteinmpnn", "1.0.1", params,
                                            timeout_s=120.0, force_new=True)
            assert rec.state.value == "completed"
            warm_samples_s.append(elapsed)

    return {
        "benchmark": "proteinmpnn_cold_vs_warm",
        "config": {
            "tool": "proteinmpnn",
            "fixture": str(fixture),
            "warm_iters": warm_iters,
            "num_designs": 4,
        },
        "cold_warm_load_s": cold_warm_load_s,
        "cold_first_call_s": cold_first_call_s,
        "warm_per_call_ms": stats_from_samples(
            [x * 1000.0 for x in warm_samples_s]),
        "warm_samples_s": warm_samples_s,
        "env": env_block(),
        "ts": time.time(),
    }


def bench_protenix_cold_vs_warm(
    input_path: Path | None = None, warm_iters: int = 3
) -> dict[str, Any]:
    """Time the first Protenix-v2 call (cold) and ``warm_iters`` warm calls."""
    from vibeproteinagent.workers.supervisor_client import SupervisorClient

    fixture = Path(input_path) if input_path else _PROTENIX_INPUT
    assert fixture.exists(), f"missing fixture {fixture}"

    cold_warm_call_s: float = 0.0
    cold_warm_poll_s: float = 0.0
    cold_first_call_s: float = 0.0
    warm_samples_s: list[float] = []

    with daemon() as h:
        client = SupervisorClient(socket_path=h.socket_path)
        # Phase 4.2 hard rule: warm RPC times out at 240s; the worker can still
        # be loading. Poll worker_info() until protenix_v2 appears as a loaded
        # adapter (or 5 minutes elapse).
        t0 = time.time()
        try:
            client.warm("protenix_v2", version_spec="*", timeout_s=240.0)
        except Exception:
            pass
        cold_warm_call_s = time.time() - t0

        deadline = time.time() + 360.0
        while time.time() < deadline:
            info = client.worker_info()
            loaded = False
            for w in info.get("workers", []):
                for adp in w.get("loaded_adapters", []):
                    if adp.get("name") == "protenix_v2":
                        loaded = True
                        break
                if loaded:
                    break
            if loaded:
                break
            time.sleep(2.0)
        cold_warm_poll_s = time.time() - t0

        params = {
            "input_json": str(fixture),
            "seeds": [101],
            "use_msa": False,
            "use_template": False,
            "n_sample": 1,
            "n_step": 20,
            "n_cycle": 1,
        }
        elapsed, rec = _submit_and_wait(h, "protenix_v2", "1.0.0", params,
                                        timeout_s=360.0, force_new=True)
        cold_first_call_s = elapsed
        assert rec.state.value == "completed", (
            f"first protenix job failed: {rec.error_code} {rec.error_message}"
        )

        for _ in range(warm_iters):
            elapsed, rec = _submit_and_wait(h, "protenix_v2", "1.0.0", params,
                                            timeout_s=120.0, force_new=True)
            assert rec.state.value == "completed"
            warm_samples_s.append(elapsed)

    return {
        "benchmark": "protenix_cold_vs_warm",
        "config": {
            "tool": "protenix_v2",
            "fixture": str(fixture),
            "warm_iters": warm_iters,
            "n_sample": 1, "n_step": 20, "n_cycle": 1,
        },
        "cold_warm_call_s": cold_warm_call_s,
        "cold_warm_poll_s": cold_warm_poll_s,
        "cold_first_call_s": cold_first_call_s,
        "warm_per_call_ms": stats_from_samples(
            [x * 1000.0 for x in warm_samples_s]),
        "warm_samples_s": warm_samples_s,
        "env": env_block(),
        "ts": time.time(),
    }


def bench_vs_naive_proteinmpnn(
    pdb_path: Path | None = None, iters: int = 3
) -> dict[str, Any]:
    """Compare warm vpa proteinmpnn vs fresh ``python protein_mpnn_run.py``.

    Each naive iteration is a fresh subprocess (python startup + torch import
    + CUDA init + checkpoint load + forward). Each vpa-warm iteration is a
    submit against a daemon with ``proteinmpnn`` already warmed.
    """
    from vibeproteinagent.workers.supervisor_client import SupervisorClient

    fixture = Path(pdb_path) if pdb_path else _PROTEINMPNN_PDB
    assert fixture.exists(), f"missing fixture {fixture}"

    if not _NAIVE_MPNN_RUNNER.exists():
        return {
            "benchmark": "vs_naive_proteinmpnn",
            "config": {"fixture": str(fixture), "iters": iters},
            "error": f"naive runner not found at {_NAIVE_MPNN_RUNNER}",
            "env": env_block(),
            "ts": time.time(),
        }

    naive_samples_s: list[float] = []
    naive_error: str | None = None
    for i in range(iters):
        out_dir = Path(tempfile.mkdtemp(prefix=f"naive_mpnn_{i}_"))
        cmd = [
            sys.executable,
            str(_NAIVE_MPNN_RUNNER),
            "--pdb_path", str(fixture),
            "--out_folder", str(out_dir),
            "--num_seq_per_target", "4",
            "--sampling_temp", "0.1",
            "--seed", "17",
            "--batch_size", "4",
            "--path_to_model_weights", str(_NAIVE_MPNN_WEIGHTS),
            "--model_name", "v_48_020",
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300.0,
            )
            elapsed = time.time() - t0
            if proc.returncode != 0:
                naive_error = (
                    f"iter {i} exit={proc.returncode} "
                    f"stderr_tail={proc.stderr[-500:]!r}"
                )
                break
            naive_samples_s.append(elapsed)
        except subprocess.TimeoutExpired:
            naive_error = f"iter {i} timed out after 300s"
            break
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    # vpa warm side.
    warm_samples_s: list[float] = []
    with daemon() as h:
        client = SupervisorClient(socket_path=h.socket_path)
        client.warm("proteinmpnn", version_spec="*", timeout_s=180.0)

        params = {
            "pdb_path": str(fixture),
            "num_designs": 4,
            "temperature": 0.1,
            "seed": 17,
        }
        # One throw-away to defeat kernel JIT.
        _submit_and_wait(h, "proteinmpnn", "1.0.1", params, timeout_s=180.0,
                         force_new=True)
        for _ in range(iters):
            elapsed, rec = _submit_and_wait(h, "proteinmpnn", "1.0.1", params,
                                            timeout_s=120.0, force_new=True)
            assert rec.state.value == "completed"
            warm_samples_s.append(elapsed)

    naive_median = (statistics.median(naive_samples_s) if naive_samples_s
                    else None)
    warm_median = (statistics.median(warm_samples_s) if warm_samples_s
                   else None)
    speedup = (
        naive_median / warm_median if naive_median and warm_median else None
    )

    return {
        "benchmark": "vs_naive_proteinmpnn",
        "config": {
            "fixture": str(fixture),
            "iters": iters,
            "num_designs": 4,
            "naive_runner": str(_NAIVE_MPNN_RUNNER),
        },
        "naive_per_call_s": naive_samples_s,
        "naive_per_call_median_s": naive_median,
        "naive_error": naive_error,
        "vpa_warm_per_call_s": warm_samples_s,
        "vpa_warm_per_call_median_s": warm_median,
        "speedup": speedup,
        "env": env_block(),
        "ts": time.time(),
    }


def bench_vs_naive_protenix(
    input_path: Path | None = None, iters: int = 2
) -> dict[str, Any]:
    """Compare warm vpa protenix_v2 vs fresh ``python runner/inference.py``."""
    from vibeproteinagent.workers.supervisor_client import SupervisorClient

    fixture = Path(input_path) if input_path else _PROTENIX_INPUT
    assert fixture.exists(), f"missing fixture {fixture}"

    if not _NAIVE_PROTENIX_RUNNER.exists():
        return {
            "benchmark": "vs_naive_protenix",
            "config": {"fixture": str(fixture), "iters": iters},
            "error": f"naive runner not found at {_NAIVE_PROTENIX_RUNNER}",
            "env": env_block(),
            "ts": time.time(),
        }

    naive_samples_s: list[float] = []
    naive_error: str | None = None
    # The naive Protenix CLI pulls knobs from JSON and command-line; mirror the
    # adapter's warm-path config (1 sample, 20 steps, 1 cycle, bf16) so the
    # comparison is apples-to-apples.
    protenix_tree = Path("/opt/vpa-data/Protenix")
    protenix_ckpt_dir = Path("/opt/vpa-data/protenix_ckpt")
    for i in range(iters):
        out_dir = Path(tempfile.mkdtemp(prefix=f"naive_protenix_{i}_"))
        env = os.environ.copy()
        # Mirror the adapter's import-time env setup.
        env["PROTENIX_ROOT_DIR"] = str(protenix_ckpt_dir)
        env["PYTHONPATH"] = str(protenix_tree)
        cmd = [
            sys.executable,
            "-m", "runner.inference",
            f"--load_checkpoint_path={protenix_ckpt_dir}/protenix_base_default_v1.0.0.pt",
            f"--input_json_path={fixture}",
            f"--dump_dir={out_dir}",
            "--seeds=101",
            "--use_msa=false",
            "--use_template=false",
            "--use_rna_msa=false",
            "--sample_diffusion.N_sample=1",
            "--sample_diffusion.N_step=20",
            "--model.N_cycle=1",
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=900.0, env=env,
                cwd=str(protenix_tree),
            )
            elapsed = time.time() - t0
            if proc.returncode != 0:
                naive_error = (
                    f"iter {i} exit={proc.returncode} "
                    f"stderr_tail={proc.stderr[-800:]!r}"
                )
                break
            naive_samples_s.append(elapsed)
        except subprocess.TimeoutExpired:
            naive_error = f"iter {i} timed out after 900s"
            break
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    # vpa warm side: pay the cold load once, then time warm runs.
    warm_samples_s: list[float] = []
    cold_warm_poll_s = 0.0
    with daemon() as h:
        client = SupervisorClient(socket_path=h.socket_path)
        t0 = time.time()
        try:
            client.warm("protenix_v2", version_spec="*", timeout_s=240.0)
        except Exception:
            pass
        deadline = time.time() + 360.0
        loaded = False
        while time.time() < deadline:
            info = client.worker_info()
            for w in info.get("workers", []):
                for adp in w.get("loaded_adapters", []):
                    if adp.get("name") == "protenix_v2":
                        loaded = True
                        break
                if loaded:
                    break
            if loaded:
                break
            time.sleep(2.0)
        cold_warm_poll_s = time.time() - t0

        params = {
            "input_json": str(fixture),
            "seeds": [101],
            "use_msa": False,
            "use_template": False,
            "n_sample": 1,
            "n_step": 20,
            "n_cycle": 1,
        }
        # One throw-away to defeat JIT.
        _submit_and_wait(h, "protenix_v2", "1.0.0", params, timeout_s=360.0,
                         force_new=True)
        for _ in range(iters):
            elapsed, rec = _submit_and_wait(h, "protenix_v2", "1.0.0", params,
                                            timeout_s=240.0, force_new=True)
            assert rec.state.value == "completed"
            warm_samples_s.append(elapsed)

    naive_median = (statistics.median(naive_samples_s) if naive_samples_s
                    else None)
    warm_median = (statistics.median(warm_samples_s) if warm_samples_s
                   else None)
    speedup = (
        naive_median / warm_median if naive_median and warm_median else None
    )

    return {
        "benchmark": "vs_naive_protenix",
        "config": {
            "fixture": str(fixture),
            "iters": iters,
            "naive_runner": str(_NAIVE_PROTENIX_RUNNER),
        },
        "naive_per_call_s": naive_samples_s,
        "naive_per_call_median_s": naive_median,
        "naive_error": naive_error,
        "vpa_warm_per_call_s": warm_samples_s,
        "vpa_warm_per_call_median_s": warm_median,
        "vpa_cold_warm_poll_s": cold_warm_poll_s,
        "speedup": speedup,
        "env": env_block(),
        "ts": time.time(),
    }


def bench_multi_gpu_dispatch(
    tool: str = "noop", jobs: int = 200, worker_counts: list[int] | None = None
) -> dict[str, Any]:
    """Throughput as worker count scales from 1 → N. Spawns a fresh daemon per
    worker count (which is the only honest way to compare since one supervisor
    is bound to one worker pool size).
    """
    from vibeproteinagent.workers.supervisor_client import SupervisorClient
    from vibeproteinagent.workspace import JobState

    if worker_counts is None:
        try:
            import torch
            max_workers = (min(torch.cuda.device_count(), 8)
                           if torch.cuda.is_available() else 1)
        except Exception:
            max_workers = 1
        # Sensible ladder: 1, 2, 4, then full N.
        cand = [1, 2, 4, max_workers]
        worker_counts = sorted({c for c in cand if 1 <= c <= max_workers})

    results: list[dict[str, Any]] = []
    for n in worker_counts:
        gpu_ids = list(range(n))
        with daemon(workspace_name=f"bench_mg_{n}", gpu_ids=gpu_ids) as h:
            client = SupervisorClient(socket_path=h.socket_path)
            client.warm(tool, version_spec="*", timeout_s=60.0)
            store, ws = _open_store(h)
            ids: list[str] = []
            t0 = time.time()
            for i in range(jobs):
                rec = store.submit(tool, "1.0.0", {"payload": f"mg-{n}-{i}"},
                                   workspace_name=ws.name)
                store.transition(rec.job_id, JobState.QUEUED)
                ids.append(rec.job_id)
            deadline = time.time() + 180.0
            while time.time() < deadline:
                recs = [store.get(j) for j in ids]
                states = [r.state.value for r in recs]
                if all(s in ("completed", "failed", "cancelled") for s in states):
                    break
                time.sleep(0.05)
            wall = time.time() - t0
            recs = [store.get(j) for j in ids]
            completion = summarize_job_completion(
                recs,
                submitted_jobs=jobs,
                wall_clock_s=wall,
            )
            results.append({
                "workers": n,
                "jobs": jobs,
                "wall_clock_s": wall,
                "completed": completion["completed_jobs"],
                **completion,
            })

    return {
        "benchmark": "multi_gpu_dispatch",
        "config": {"tool": tool, "jobs": jobs, "worker_counts": worker_counts},
        "results": results,
        "env": env_block(),
        "ts": time.time(),
    }


def bench_workflow_cache_hit(msg: str = "bench") -> dict[str, Any]:
    """Run the 3-noop workflow twice; second run should hit cache on all nodes."""
    from vibeproteinagent.adapters import ToolRegistry
    from vibeproteinagent.workers.supervisor_client import SupervisorClient
    from vibeproteinagent.workflows import Workflow, WorkflowExecutor
    from vibeproteinagent.workspace import Workspace

    with daemon() as h:
        client = SupervisorClient(socket_path=h.socket_path)
        client.warm("noop", version_spec="*", timeout_s=60.0)

        ws = Workspace.resolve(h.workspace)
        registry = ToolRegistry()
        registry.discover()

        wf = Workflow(name="bench_noop3", version="1.0.0")
        wf.node("a", tool="noop", inputs={"payload": "${params.msg}"})
        wf.node("b", tool="noop", inputs={"payload": "${outputs.a.payload}-b"})
        wf.node("c", tool="noop", inputs={"payload": "${outputs.b.payload}-c"})

        # First run — cold (no cached nodes).
        ex1 = WorkflowExecutor(ws, registry, client)
        t0 = time.time()
        wf_id1 = ex1.submit(wf, {"msg": msg})
        rec1 = ex1.run_sync(wf_id1, max_wait_s=60, poll_s=0.05)
        first_run_wall_s = time.time() - t0
        assert rec1.state == "completed", rec1
        nodes1 = ex1.store.list_nodes(wf_id1)
        states1 = sorted({n.state for n in nodes1})

        # Second run — same params; all nodes should cache.
        ex2 = WorkflowExecutor(ws, registry, client)
        t0 = time.time()
        wf_id2 = ex2.submit(wf, {"msg": msg})
        rec2 = ex2.run_sync(wf_id2, max_wait_s=30, poll_s=0.05)
        second_run_wall_s = time.time() - t0
        assert rec2.state == "completed", rec2
        nodes2 = ex2.store.list_nodes(wf_id2)
        states2 = sorted({n.state for n in nodes2})

    return {
        "benchmark": "workflow_cache_hit",
        "config": {"msg": msg, "workflow": "3-noop chain"},
        "first_run_wall_s": first_run_wall_s,
        "second_run_wall_s": second_run_wall_s,
        "speedup": (first_run_wall_s / second_run_wall_s
                    if second_run_wall_s > 0 else None),
        "node_states_first": states1,
        "node_states_second": states2,
        "all_cached_on_second": states2 == ["cached"],
        "env": env_block(),
        "ts": time.time(),
    }


# ---------------------------------------------------------------- 'all'


def bench_all(
    *,
    quick: bool = False,
    include_protenix: bool = True,
    include_naive: bool = True,
    include_multi_gpu: bool = True,
) -> dict[str, Any]:
    """Run the full suite. Each sub-benchmark spawns its own supervisor."""
    started_at = time.time()
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def _run(label: str, fn, **kw):
        try:
            print(f"[bench] running {label} ...", file=sys.stderr, flush=True)
            t0 = time.time()
            r = fn(**kw)
            print(f"[bench]   {label} done in {time.time() - t0:.1f}s",
                  file=sys.stderr, flush=True)
            results[label] = r
            if isinstance(r, dict) and r.get("error"):
                errors[label] = str(r["error"])
        except Exception as exc:
            print(f"[bench]   {label} FAILED: {exc}", file=sys.stderr, flush=True)
            errors[label] = repr(exc)

    _run("noop_throughput", bench_noop_throughput, jobs=50 if quick else 100)
    _run("noop_warm_call", bench_noop_warm_call, iterations=20 if quick else 50)
    _run("workflow_cache_hit", bench_workflow_cache_hit, msg="bench")
    _run("proteinmpnn_cold_vs_warm", bench_proteinmpnn_cold_vs_warm,
         warm_iters=3 if quick else 5)
    if include_protenix:
        _run("protenix_cold_vs_warm", bench_protenix_cold_vs_warm,
             warm_iters=2 if quick else 3)
    if include_naive:
        _run("vs_naive_proteinmpnn", bench_vs_naive_proteinmpnn,
             iters=2 if quick else 3)
        if include_protenix:
            _run("vs_naive_protenix", bench_vs_naive_protenix,
                 iters=1 if quick else 2)
    if include_multi_gpu:
        _run("multi_gpu_dispatch", bench_multi_gpu_dispatch,
             jobs=100 if quick else 200)

    return {
        "benchmark": "all",
        "started_at": started_at,
        "finished_at": time.time(),
        "wall_clock_s": time.time() - started_at,
        "results": results,
        "errors": errors,
        "failed": bool(errors),
        "skipped": {
            "protenix": not include_protenix,
            "naive": not include_naive,
            "multi_gpu": not include_multi_gpu,
        },
        "env": env_block(),
        "provenance": provenance_block(),
    }


# ---------------------------------------------------------------- CLI


def _print(obj: Any, *, as_json: bool) -> None:
    if as_json:
        json.dump(obj, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        import pprint
        pprint.pprint(obj, width=120)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description=("VPA benchmark harness — measures the framework's "
                     "persistent-worker + warm-pool performance bet."),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--json", action="store_true",
                       help="emit a JSON blob (default: pretty)")

    p = sub.add_parser("noop_throughput")
    p.add_argument("--jobs", type=int, default=100)
    _add_common(p)

    p = sub.add_parser("noop_warm_call")
    p.add_argument("--iterations", type=int, default=50)
    _add_common(p)

    p = sub.add_parser("proteinmpnn_cold_vs_warm")
    p.add_argument("--pdb", type=Path, default=None)
    p.add_argument("--warm-iters", type=int, default=5)
    _add_common(p)

    p = sub.add_parser("protenix_cold_vs_warm")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--warm-iters", type=int, default=3)
    _add_common(p)

    p = sub.add_parser("vs_naive_proteinmpnn")
    p.add_argument("--pdb", type=Path, default=None)
    p.add_argument("--iters", type=int, default=3)
    _add_common(p)

    p = sub.add_parser("vs_naive_protenix")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--iters", type=int, default=2)
    _add_common(p)

    p = sub.add_parser("multi_gpu_dispatch")
    p.add_argument("--tool", type=str, default="noop")
    p.add_argument("--jobs", type=int, default=200)
    p.add_argument("--workers", type=str, default=None,
                   help="comma-separated worker counts, e.g. 1,2,4,8")
    _add_common(p)

    p = sub.add_parser("workflow_cache_hit")
    p.add_argument("--param", type=str, action="append", default=None,
                   help="KEY=VALUE (currently only msg=...)")
    _add_common(p)

    p = sub.add_parser("all")
    p.add_argument("--quick", action="store_true",
                   help="use shorter iterations everywhere")
    p.add_argument("--skip-protenix", action="store_true")
    p.add_argument("--skip-naive", action="store_true")
    p.add_argument("--skip-multi-gpu", action="store_true")
    _add_common(p)

    args = parser.parse_args(argv)

    if args.cmd == "noop_throughput":
        out = bench_noop_throughput(jobs=args.jobs)
    elif args.cmd == "noop_warm_call":
        out = bench_noop_warm_call(iterations=args.iterations)
    elif args.cmd == "proteinmpnn_cold_vs_warm":
        out = bench_proteinmpnn_cold_vs_warm(pdb_path=args.pdb,
                                             warm_iters=args.warm_iters)
    elif args.cmd == "protenix_cold_vs_warm":
        out = bench_protenix_cold_vs_warm(input_path=args.input,
                                          warm_iters=args.warm_iters)
    elif args.cmd == "vs_naive_proteinmpnn":
        out = bench_vs_naive_proteinmpnn(pdb_path=args.pdb, iters=args.iters)
    elif args.cmd == "vs_naive_protenix":
        out = bench_vs_naive_protenix(input_path=args.input, iters=args.iters)
    elif args.cmd == "multi_gpu_dispatch":
        wc = None
        if args.workers:
            wc = [int(x) for x in args.workers.split(",") if x.strip()]
        out = bench_multi_gpu_dispatch(
            tool=args.tool, jobs=args.jobs, worker_counts=wc,
        )
    elif args.cmd == "workflow_cache_hit":
        msg = "bench"
        for kv in (args.param or []):
            if kv.startswith("msg="):
                msg = kv[len("msg="):]
        out = bench_workflow_cache_hit(msg=msg)
    elif args.cmd == "all":
        out = bench_all(
            quick=args.quick,
            include_protenix=not args.skip_protenix,
            include_naive=not args.skip_naive,
            include_multi_gpu=not args.skip_multi_gpu,
        )
    else:  # pragma: no cover
        parser.error(f"unknown cmd {args.cmd!r}")
        return 2

    _print(out, as_json=getattr(args, "json", False))
    if args.cmd == "all" and out.get("errors"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
