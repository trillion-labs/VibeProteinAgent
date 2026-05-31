from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from importlib import import_module
from typing import Any

CUDA_PROFILE_CHOICES = ("auto", "cu128", "cu126", "cpu")


@dataclass(frozen=True)
class CudaStack:
    name: str
    cuda_version: str | None
    torch_spec: str
    torch_index_url: str
    pyg_wheel_url: str | None
    jax_extra: str | None
    bindcraft_cuda: str | None
    dgl_package: str | None
    dgl_wheel_url: str | None
    minimum_driver: str | None
    summary: str


CUDA_STACKS: dict[str, CudaStack] = {
    "cu128": CudaStack(
        name="cu128",
        cuda_version="12.8",
        torch_spec="torch==2.7.1+cu128",
        torch_index_url="https://download.pytorch.org/whl/cu128",
        pyg_wheel_url="https://data.pyg.org/whl/torch-2.7.0+cu128.html",
        jax_extra="jax[cuda12]",
        bindcraft_cuda="12.8",
        dgl_package="dgl==2.1.0+cu121",
        dgl_wheel_url="https://data.dgl.ai/wheels/cu121/repo.html",
        minimum_driver="570.26",
        summary="primary tested CUDA 12.8 stack",
    ),
    "cu126": CudaStack(
        name="cu126",
        cuda_version="12.6",
        torch_spec="torch==2.7.1+cu126",
        torch_index_url="https://download.pytorch.org/whl/cu126",
        pyg_wheel_url="https://data.pyg.org/whl/torch-2.7.0+cu126.html",
        jax_extra="jax[cuda12]",
        bindcraft_cuda="12.6",
        dgl_package="dgl==2.1.0+cu121",
        dgl_wheel_url="https://data.dgl.ai/wheels/cu121/repo.html",
        minimum_driver="560.35",
        summary="best-effort CUDA 12.6 fallback for older drivers",
    ),
    "cpu": CudaStack(
        name="cpu",
        cuda_version=None,
        torch_spec="torch==2.7.1",
        torch_index_url="https://download.pytorch.org/whl/cpu",
        pyg_wheel_url="https://data.pyg.org/whl/torch-2.7.0+cpu.html",
        jax_extra=None,
        bindcraft_cuda=None,
        dgl_package=None,
        dgl_wheel_url=None,
        minimum_driver=None,
        summary="CPU-only fallback for base, agent, and lightweight smoke checks",
    ),
}


GPU_HEAVY_TOOLS = frozenset(
    {
        "bindcraft",
        "boltz2_affinity",
        "boltzgen",
        "esm3",
        "ligandmpnn",
        "mmseqs2_gpu",
        "proteinmpnn",
        "proteina_complexa",
        "protenix_v2",
        "pxdesign",
        "rfdiffusion",
        "rfdiffusion3",
        "rfpeptides",
    }
)

CUDA_COMPILING_TOOLS = frozenset(
    {
        "bindcraft",
        "proteina_complexa",
        "protenix_v2",
        "pxdesign",
    }
)

RF_DGL_TOOLS = frozenset({"rfdiffusion", "rfpeptides"})


def resolve_cuda_stack(requested: str | None = None) -> CudaStack:
    profile = (requested or os.environ.get("VPA_CUDA_PROFILE") or "auto").strip().lower()
    if profile not in CUDA_PROFILE_CHOICES:
        raise ValueError(
            f"unknown CUDA profile '{profile}'. Choose one of: {', '.join(CUDA_PROFILE_CHOICES)}"
        )
    if profile == "auto":
        return CUDA_STACKS[_auto_profile_name()]
    return CUDA_STACKS[profile]


def preflight_report(
    *,
    requested_profile: str | None = None,
    selected_tools: tuple[str, ...] = (),
) -> dict[str, Any]:
    requested = (requested_profile or os.environ.get("VPA_CUDA_PROFILE") or "auto").strip().lower()
    stack = resolve_cuda_stack(requested)
    host = _host_probe()
    checks = _checks_for(host, stack, selected_tools)
    status = _overall_status(checks)
    return {
        "status": status,
        "requested_cuda_profile": requested,
        "cuda_profile": stack.name,
        "cuda_version": stack.cuda_version,
        "summary": stack.summary,
        "selected_tools": list(selected_tools),
        "host": host,
        "checks": checks,
        "recommendations": _recommendations(status, stack, selected_tools),
    }


def _auto_profile_name() -> str:
    host = _host_probe()
    driver = _first_driver_version(host)
    if driver and _version_lt(driver, CUDA_STACKS["cu128"].minimum_driver or "0"):
        if not _version_lt(driver, CUDA_STACKS["cu126"].minimum_driver or "0"):
            return "cu126"
    return "cu128"


def _host_probe() -> dict[str, Any]:
    return {
        "nvidia_smi": _nvidia_smi_info(),
        "nvcc": _command_version(["nvcc", "--version"]),
        "gcc": _command_version(["gcc", "--version"], first_line=True),
        "cuda_home": os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"),
        "torch": _torch_probe(),
    }


def _nvidia_smi_info() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "error": "nvidia-smi not found on PATH", "gpus": []}
    cmd = [
        exe,
        "--query-gpu=name,driver_version,compute_cap,memory.total",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return {
            "available": False,
            "error": (proc.stderr or proc.stdout).strip()[-1000:],
            "gpus": [],
        }
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        memory_mib = _to_int(parts[3])
        gpus.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "compute_capability": parts[2],
                "memory_total_mib": memory_mib,
                "memory_total_gib": round(memory_mib / 1024, 1) if memory_mib else None,
            }
        )
    return {"available": True, "path": exe, "gpus": gpus}


def _torch_probe() -> dict[str, Any]:
    try:
        torch = import_module("torch")
        cpp_extension = import_module("torch.utils.cpp_extension")
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    info: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpp_extension_cuda_home": str(cpp_extension.CUDA_HOME)
        if cpp_extension.CUDA_HOME
        else None,
    }
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "compute_capability": ".".join(str(v) for v in torch.cuda.get_device_capability(i)),
            }
            for i in range(torch.cuda.device_count())
        ]
    return info


def _checks_for(
    host: dict[str, Any],
    stack: CudaStack,
    selected_tools: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected = set(selected_tools)
    checks: list[dict[str, Any]] = []
    driver = _first_driver_version(host)
    if stack.name == "cpu":
        checks.append(
            {
                "name": "cuda_profile",
                "status": "pass",
                "message": "CPU profile selected; GPU checks are advisory only.",
            }
        )
    elif not driver:
        checks.append(
            {
                "name": "nvidia_driver",
                "status": "warn",
                "message": "Could not read NVIDIA driver version with nvidia-smi.",
            }
        )
    elif stack.minimum_driver and _version_lt(driver, stack.minimum_driver):
        checks.append(
            {
                "name": "nvidia_driver",
                "status": "fail",
                "message": (
                    f"{stack.name} needs NVIDIA driver >= {stack.minimum_driver}; "
                    f"detected {driver}."
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "nvidia_driver",
                "status": "pass",
                "message": f"NVIDIA driver {driver} is compatible with {stack.name}.",
            }
        )

    gpus = list((host.get("nvidia_smi") or {}).get("gpus") or [])
    if stack.name != "cpu" and not gpus:
        checks.append(
            {
                "name": "gpu_visibility",
                "status": "warn",
                "message": "No NVIDIA GPUs were visible to nvidia-smi during preflight.",
            }
        )
    else:
        for gpu in gpus:
            cc = str(gpu.get("compute_capability") or "")
            status = "pass" if _cc_at_least(cc, "8.6") else "warn"
            checks.append(
                {
                    "name": "gpu_compute_capability",
                    "status": status,
                    "gpu": gpu.get("name"),
                    "message": (
                        f"{gpu.get('name')} reports compute capability {cc}. "
                        "VPA's cu128 path is targeted at 8.6/8.9/9.0/10.0-class GPUs."
                    ),
                }
            )

    nvcc = host.get("nvcc") or {}
    if selected.intersection(CUDA_COMPILING_TOOLS):
        if nvcc.get("available"):
            checks.append(
                {
                    "name": "nvcc",
                    "status": "pass",
                    "message": f"nvcc is available: {nvcc.get('version')}",
                }
            )
        else:
            checks.append(
                {
                    "name": "nvcc",
                    "status": "warn",
                    "message": (
                        "Selected tools may compile CUDA extensions. Install a matching CUDA "
                        "toolkit/nvcc or use the container/prebuilt sidecar path."
                    ),
                }
            )

    if selected.intersection(RF_DGL_TOOLS):
        checks.append(
            {
                "name": "rfdiffusion_dgl",
                "status": "warn",
                "message": (
                    "RFdiffusion/RFpeptides use a best-effort DGL CUDA wheel because DGL "
                    "does not publish a first-class cu128 wheel for this stack."
                ),
            }
        )

    if stack.name == "cpu" and selected.intersection(GPU_HEAVY_TOOLS):
        checks.append(
            {
                "name": "cpu_profile_gpu_tools",
                "status": "warn",
                "message": "GPU-heavy tools are selected under the CPU profile.",
            }
        )
    return checks


def _recommendations(
    status: str,
    stack: CudaStack,
    selected_tools: tuple[str, ...],
) -> list[str]:
    recommendations: list[str] = []
    if status == "fail":
        recommendations.append(
            "Use a newer NVIDIA driver, choose --cuda-profile cu126 when appropriate, "
            "or use the Docker cu128 image on a compatible host."
        )
    if set(selected_tools).intersection(CUDA_COMPILING_TOOLS):
        recommendations.append(
            "For CUDA-compiling tools, prefer a host CUDA toolkit matching the selected "
            f"{stack.name} stack or a prebuilt sidecar/container."
        )
    if stack.name == "cu128":
        recommendations.append("Run a small smoke job before long workflows.")
    return recommendations


def _command_version(cmd: list[str], *, first_line: bool = False) -> dict[str, Any]:
    exe = shutil.which(cmd[0])
    if exe is None:
        return {"available": False, "error": f"{cmd[0]} not found on PATH"}
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    text = (proc.stdout or proc.stderr).strip()
    if first_line:
        text = text.splitlines()[0] if text else ""
    return {"available": proc.returncode == 0, "path": exe, "version": text[-1000:]}


def _first_driver_version(host: dict[str, Any]) -> str | None:
    for gpu in (host.get("nvidia_smi") or {}).get("gpus") or []:
        driver = gpu.get("driver_version")
        if driver:
            return str(driver)
    return None


def _version_lt(left: str, right: str) -> bool:
    return _version_tuple(left) < _version_tuple(right)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value)[:3])


def _cc_at_least(left: str, right: str) -> bool:
    return _version_tuple(left) >= _version_tuple(right)


def _to_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def _overall_status(checks: list[dict[str, Any]]) -> str:
    if any(check.get("status") == "fail" for check in checks):
        return "fail"
    if any(check.get("status") == "warn" for check in checks):
        return "warn"
    return "pass"
