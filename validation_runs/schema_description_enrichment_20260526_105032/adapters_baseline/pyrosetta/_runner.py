"""PyRosetta subprocess runner.

This module is executed inside the dedicated ``pyrosetta-vpa`` conda env
(NOT the vpa env). It reads a tiny JSON job spec, dispatches to the
right PyRosetta primitive, and writes the result back as JSON.

Invocation::

    <env>/bin/python -m vibeproteinagent.adapters.pyrosetta._runner \
        --input <input.json> --output <result.json>

The runner imports ``pyrosetta`` lazily so that ``--help`` (and any test
that imports this module purely for static checks) does not require the
PyRosetta install.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_INITIALIZED = False
_INITIALIZED_SEED: int | None = None


def _init_pyrosetta(seed: int | None) -> None:
    """Initialise PyRosetta with quiet, deterministic flags.

    ``-mute all`` keeps the subprocess stderr tractable; the adapter
    forwards stdout/stderr to the job log anyway. ``-constant_seed
    -jran <seed>`` makes runs reproducible when a seed is supplied.
    """
    import pyrosetta  # type: ignore[import-not-found]

    flags = [
        "-mute", "all",
        "-ignore_unrecognized_res",
        "-ignore_zero_occupancy", "false",
        "-load_PDB_components", "false",
    ]
    if seed is not None:
        flags += ["-constant_seed", "-jran", str(int(seed))]
    pyrosetta.init(extra_options=" ".join(flags))


def _init_pyrosetta_once(seed: int | None) -> None:
    """Initialise PyRosetta once for a persistent sidecar process."""
    global _INITIALIZED, _INITIALIZED_SEED
    if _INITIALIZED:
        if seed is not None and _INITIALIZED_SEED != int(seed):
            raise ValueError(
                "persistent PyRosetta sidecar was already initialised with "
                f"seed={_INITIALIZED_SEED}; received seed={seed}. Use a separate "
                "sidecar instance for a different deterministic seed."
            )
        return
    _init_pyrosetta(seed)
    _INITIALIZED = True
    _INITIALIZED_SEED = int(seed) if seed is not None else None


def _score(pose, score_fn) -> float:
    return float(score_fn(pose))


def _interface(pose, chains_a: str, chains_b: str) -> dict[str, Any]:
    from pyrosetta.rosetta.protocols.analysis import (  # type: ignore[import-not-found]
        InterfaceAnalyzerMover,
    )

    if not chains_a or not chains_b:
        raise ValueError("interface mode requires non-empty chains_a and chains_b")

    interface_str = f"{chains_a}_{chains_b}"
    mover = InterfaceAnalyzerMover(interface_str)
    mover.set_pack_separated(True)
    mover.set_compute_packstat(True)
    mover.set_compute_interface_sc(True)
    mover.set_compute_interface_energy(True)
    mover.apply(pose)
    data = mover.get_all_data()

    metrics: dict[str, Any] = {
        "dG_separated": float(data.dG[1]) if hasattr(data, "dG") and len(data.dG) > 1 else None,
        "dSASA": float(data.dSASA[1]) if hasattr(data, "dSASA") and len(data.dSASA) > 1 else None,
        "sc": float(getattr(data, "sc_value", 0.0)) if hasattr(data, "sc_value") else None,
        "packstat": float(getattr(data, "packstat", 0.0)) if hasattr(data, "packstat") else None,
        "nres_int": int(getattr(data, "interface_nres", [0, 0])[1])
            if hasattr(data, "interface_nres") and len(data.interface_nres) > 1 else None,
        "interface_hbonds": int(getattr(data, "interface_hbonds", 0))
            if hasattr(data, "interface_hbonds") else None,
    }
    # dG per interface area, when both are available.
    if metrics["dG_separated"] is not None and metrics["dSASA"] and metrics["dSASA"] > 0:
        metrics["dG_separated_per_int_area"] = metrics["dG_separated"] / metrics["dSASA"]
    else:
        metrics["dG_separated_per_int_area"] = None
    return metrics


def _relax(pose, score_fn, max_cycles: int) -> None:
    from pyrosetta.rosetta.protocols.relax import FastRelax  # type: ignore[import-not-found]

    relax = FastRelax(int(max_cycles))
    relax.set_scorefxn(score_fn)
    relax.apply(pose)


def _pack(pose, score_fn) -> None:
    from pyrosetta.rosetta.core.pack.task import TaskFactory  # type: ignore[import-not-found]
    from pyrosetta.rosetta.core.pack.task.operation import (  # type: ignore[import-not-found]
        RestrictToRepacking,
    )
    from pyrosetta.rosetta.protocols.minimization_packing import (  # type: ignore[import-not-found]
        PackRotamersMover,
    )

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    packer = PackRotamersMover(score_fn)
    packer.task_factory(tf)
    packer.apply(pose)


def _execute(spec: dict[str, Any]) -> dict[str, Any]:
    import pyrosetta  # type: ignore[import-not-found]

    t0 = time.time()
    seed = spec.get("seed")

    pdb_path = Path(spec["pdb_path"]).expanduser().resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"pdb_path does not exist: {pdb_path}")

    mode: str = spec.get("mode", "score")
    score_fn_tag: str = spec.get("score_fn", "ref2015")

    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    score_fn = pyrosetta.create_score_function(score_fn_tag)

    relaxed_pdb_path: str | None = None
    interface_metrics: dict[str, Any] | None = None

    if mode == "score":
        pass  # just score below
    elif mode == "interface":
        interface_metrics = _interface(
            pose,
            chains_a=spec.get("chains_a") or "",
            chains_b=spec.get("chains_b") or "",
        )
    elif mode == "relax":
        max_cycles = int(spec.get("max_relax_cycles", 1))
        _relax(pose, score_fn, max_cycles)
        out_pdb = Path(spec["out_pdb"])
        out_pdb.parent.mkdir(parents=True, exist_ok=True)
        pose.dump_pdb(str(out_pdb))
        relaxed_pdb_path = str(out_pdb)
    elif mode == "pack":
        _pack(pose, score_fn)
        out_pdb = Path(spec["out_pdb"])
        out_pdb.parent.mkdir(parents=True, exist_ok=True)
        pose.dump_pdb(str(out_pdb))
        relaxed_pdb_path = str(out_pdb)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    total_score = _score(pose, score_fn)
    runtime_ms = int((time.time() - t0) * 1000)

    result: dict[str, Any] = {
        "mode": mode,
        "total_score": total_score,
        "interface_metrics": interface_metrics,
        "relaxed_pdb_path": relaxed_pdb_path,
        "runtime_ms": runtime_ms,
        "seed_used": int(seed) if seed is not None else None,
    }
    return result


def run(spec: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Dispatch one cold subprocess job, write result JSON, and return it."""
    seed = spec.get("seed")
    _init_pyrosetta(seed)
    result = _execute(spec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def warm_sidecar(seed: int | None = None) -> dict[str, Any]:
    """Pre-initialise PyRosetta in a persistent sidecar."""
    _init_pyrosetta_once(seed)
    return {"initialized": True, "seed": _INITIALIZED_SEED}


def run_sidecar(spec: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a job inside an already-warm or lazily-warmed sidecar."""
    _init_pyrosetta_once(spec.get("seed"))
    return _execute(spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vibeproteinagent.adapters.pyrosetta._runner")
    parser.add_argument("--input", required=True, help="Path to job-spec JSON.")
    parser.add_argument("--output", required=True, help="Path to write result JSON.")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.input).read_text())
    out = Path(args.output)
    try:
        run(spec, out)
    except Exception as exc:  # pragma: no cover - exercised by real subprocess
        err = {
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(err, indent=2))
        print(f"pyrosetta runner failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
