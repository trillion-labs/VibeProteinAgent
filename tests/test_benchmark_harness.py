"""Phase 5.3 benchmark harness — fast tests for the shape of the JSON output.

The expensive sub-benchmarks (proteinmpnn/protenix/vs_naive/multi_gpu) live
behind ``@pytest.mark.slow``. Fast tests verify:

  - the script is importable, sub-commands parse, JSON shape matches the
    expected schema;
  - ``noop_throughput --jobs 5`` runs end-to-end and produces a valid blob;
  - the ``stats_from_samples`` helper is correct.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
_BENCH = _REPO / "scripts" / "benchmark.py"


@pytest.fixture(scope="module")
def benchmark_module():
    """Import ``scripts/benchmark.py`` once for unit tests."""
    mod_name = "vpa_benchmark_for_tests"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(_BENCH))
    assert spec and spec.loader, f"failed to spec {_BENCH}"
    mod = importlib.util.module_from_spec(spec)
    # dataclass needs the module reachable via sys.modules during exec.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_benchmark_file_exists() -> None:
    assert _BENCH.is_file(), f"missing {_BENCH}"


def test_stats_from_samples_basic(benchmark_module) -> None:
    """The percentile helper handles tiny inputs sanely."""
    stats = benchmark_module.stats_from_samples([])
    assert stats == {"n": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0,
                     "max": 0.0}

    stats = benchmark_module.stats_from_samples([42.0])
    assert stats["n"] == 1
    assert stats["min"] == 42.0
    assert stats["p50"] == 42.0
    assert stats["p95"] == 42.0
    assert stats["max"] == 42.0


def test_stats_from_samples_monotone(benchmark_module) -> None:
    """Percentiles respect ordering."""
    stats = benchmark_module.stats_from_samples([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["n"] == 5
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["p50"] == 3.0
    assert stats["p95"] >= stats["p50"]
    assert stats["p99"] >= stats["p95"]


def test_env_block_shape(benchmark_module) -> None:
    """``env_block`` always returns the canonical keys."""
    env = benchmark_module.env_block()
    assert "python" in env
    # vpa version is "unknown" if metadata not found, but always a string.
    assert "vpa" in env
    assert isinstance(env["python"], str)


def test_completion_summary_counts_only_completed_throughput(benchmark_module) -> None:
    records = [
        SimpleNamespace(state="completed", started_at=1_000, finished_at=4_000),
        SimpleNamespace(state="completed", started_at=2_000, finished_at=7_000),
        SimpleNamespace(state="queued", started_at=None, finished_at=None),
        SimpleNamespace(state="failed", started_at=3_000, finished_at=6_000),
    ]
    out = benchmark_module.summarize_job_completion(
        records,
        submitted_jobs=4,
        wall_clock_s=10.0,
    )
    assert out["submitted_jobs"] == 4
    assert out["completed_jobs"] == 2
    assert out["terminal_jobs"] == 3
    assert out["all_completed"] is False
    assert out["all_terminal"] is False
    assert out["state_counts"] == {"completed": 2, "queued": 1, "failed": 1}
    assert out["throughput_per_s"] == pytest.approx(0.2)
    assert out["per_job_ms"]["n"] == 2


def test_bench_all_marks_error_result_failed(
    benchmark_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark_module,
        "bench_noop_throughput",
        lambda **_kw: {"benchmark": "noop_throughput"},
    )
    monkeypatch.setattr(
        benchmark_module,
        "bench_noop_warm_call",
        lambda **_kw: {"benchmark": "noop_warm_call"},
    )
    monkeypatch.setattr(
        benchmark_module,
        "bench_workflow_cache_hit",
        lambda **_kw: {"benchmark": "workflow_cache_hit"},
    )
    monkeypatch.setattr(
        benchmark_module,
        "bench_proteinmpnn_cold_vs_warm",
        lambda **_kw: {"benchmark": "proteinmpnn_cold_vs_warm"},
    )
    monkeypatch.setattr(
        benchmark_module,
        "bench_vs_naive_proteinmpnn",
        lambda **_kw: {"benchmark": "vs_naive_proteinmpnn", "error": "missing runner"},
    )

    out = benchmark_module.bench_all(
        quick=True,
        include_protenix=False,
        include_naive=True,
        include_multi_gpu=False,
    )

    assert out["failed"] is True
    assert out["errors"]["vs_naive_proteinmpnn"] == "missing runner"


def test_cli_help_runs(benchmark_module) -> None:
    """``benchmark.py --help`` exits zero and lists all sub-commands."""
    proc = subprocess.run(
        [sys.executable, str(_BENCH), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    txt = proc.stdout + proc.stderr
    for expected in (
        "noop_throughput", "noop_warm_call", "proteinmpnn_cold_vs_warm",
        "protenix_cold_vs_warm", "vs_naive_proteinmpnn", "vs_naive_protenix",
        "multi_gpu_dispatch", "workflow_cache_hit", "all",
    ):
        assert expected in txt, f"sub-command {expected!r} missing from --help"


# -------------------------------------------------------------- slow tests


@pytest.mark.slow
def test_noop_throughput_small_run(tmp_path: Path) -> None:
    """End-to-end: 5-job noop throughput run produces a well-shaped JSON blob.

    Spawns a real supervisor + workers — that's why it's slow.
    """
    env = os.environ.copy()
    env.pop("VPA_WORKSPACE", None)
    # The harness uses tempfile.mkdtemp internally; we don't override here.
    proc = subprocess.run(
        [sys.executable, str(_BENCH), "noop_throughput", "--jobs", "5", "--json"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    blob = json.loads(proc.stdout)

    # Schema invariants.
    assert blob["benchmark"] == "noop_throughput"
    assert blob["config"]["jobs"] == 5
    assert blob["config"]["tool"] == "noop"
    assert blob["config"]["warmed"] is True
    assert isinstance(blob["wall_clock_s"], float)
    assert blob["wall_clock_s"] > 0
    assert isinstance(blob["throughput_per_s"], float)

    pj = blob["per_job_ms"]
    for k in ("n", "min", "p50", "p95", "p99", "max"):
        assert k in pj, f"missing per_job_ms.{k}"
    # We measured at least one completed job.
    assert pj["n"] >= 1
    # min <= p50 <= p95 <= max
    assert pj["min"] <= pj["p50"] <= pj["p95"] <= pj["max"]

    env_blob = blob["env"]
    assert "python" in env_blob
    assert "vpa" in env_blob


@pytest.mark.slow
def test_workflow_cache_hit_blob() -> None:
    """Workflow cache-hit benchmark; second run hits cache for all 3 nodes."""
    env = os.environ.copy()
    env.pop("VPA_WORKSPACE", None)
    proc = subprocess.run(
        [
            sys.executable,
            str(_BENCH),
            "workflow_cache_hit",
            "--param",
            "msg=harness-test",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    out = json.loads(proc.stdout)
    assert out["benchmark"] == "workflow_cache_hit"
    assert out["first_run_wall_s"] > 0
    assert out["second_run_wall_s"] > 0
    assert out["all_cached_on_second"], (
        f"second run node states were {out['node_states_second']}"
    )
    # Cache hit must be cheaper than the cold first run.
    assert out["second_run_wall_s"] < out["first_run_wall_s"] * 1.5
