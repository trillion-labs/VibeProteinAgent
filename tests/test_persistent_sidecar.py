from __future__ import annotations

import os
import sys
from pathlib import Path

from vibeproteinagent.workers.sidecar import (
    PersistentPythonSidecar,
    SidecarPool,
    SidecarRuntimeSpec,
    with_vpa_src_pythonpath,
)


def test_persistent_python_sidecar_calls_json(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    sidecar = PersistentPythonSidecar(
        python=Path(sys.executable),
        env=env,
        name="unit-json",
        startup_timeout_s=10.0,
    )
    try:
        assert sidecar.health()["ok"] is True
        out = sidecar.call("json:loads", {"s": '{"answer": 42}'}, timeout_s=10.0)
        assert out == {"answer": 42}
    finally:
        sidecar.close()


def test_with_vpa_src_pythonpath_prepends_checkout_src():
    env = {"PYTHONPATH": "/parent/path"}
    out = with_vpa_src_pythonpath(env)
    assert out["PYTHONPATH"].split(os.pathsep)[0].endswith("/src")
    assert "/parent/path" in out["PYTHONPATH"]


def test_sidecar_runtime_spec_and_pool_reuse():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    spec = SidecarRuntimeSpec(
        python=Path(sys.executable),
        env=env,
        name="unit-pool",
        timeout_s=10.0,
        run_target="json:loads",
    )
    pool = SidecarPool()
    try:
        first = pool.get("seed-1", spec)
        second = pool.get("seed-1", spec)
        assert first is second
        assert first.call(spec.run_target or "", {"s": '{"ok": true}'}, timeout_s=10.0) == {
            "ok": True
        }
    finally:
        pool.close_all()
