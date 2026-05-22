"""RosettaCommons RFdiffusion adapter.

This adapter is the general backbone-generation entry point for VPA. It wraps
the RFdiffusion ``scripts/run_inference.py`` Hydra interface and exposes the
major backbone workflows as explicit modes while still allowing advanced Hydra
overrides for users who know the native RFdiffusion knobs.

The default checkout is ``vendor/rfpeptides`` because that tree is an
RFdiffusion branch already vendored with VPA. Operators may point to a different
RosettaCommons/RFdiffusion checkout with ``VPA_RFDIFFUSION_TREE`` and to a
weights directory with ``VPA_RFDIFFUSION_WEIGHTS``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.core.compat import import_dgl_with_graphbolt_guard

_ADAPTER_DIR = Path(__file__).parent
_VPA_ROOT = _ADAPTER_DIR.parent.parent.parent.parent

RFDIFFUSION_TREE_DEFAULT = _VPA_ROOT / "vendor" / "rfpeptides"
RFDIFFUSION_TREE = Path(
    os.environ.get("VPA_RFDIFFUSION_TREE", str(RFDIFFUSION_TREE_DEFAULT))
).expanduser()
RFDIFFUSION_WEIGHTS = Path(
    os.environ.get("VPA_RFDIFFUSION_WEIGHTS", str(RFDIFFUSION_TREE / "weights"))
).expanduser()

RFdiffusionMode = Literal[
    "unconditional",
    "unconditional_generation",
    "backbone_generation",
    "motif_scaffolding",
    "symmetric_motif_scaffolding",
    "partial_diffusion",
    "binder_design",
    "binder_backbone_design",
    "fold_conditioning",
    "symmetric_oligomer",
    "symmetric_oligomer_design",
]

_BASE_CKPT = "Base_ckpt.pt"
_COMPLEX_CKPT = "Complex_base_ckpt.pt"
_FOLD_CKPT = "Complex_Fold_base_ckpt.pt"


class RFdiffusionInput(BaseModel):
    mode: RFdiffusionMode = "unconditional"
    target_pdb: str | None = Field(
        default=None,
        json_schema_extra={"vpa:digest": "sha256-if-path"},
    )
    target_chain: str = Field(default="A", min_length=1, max_length=1)
    target_residues: str | None = Field(default=None, pattern=r"^[0-9]+-[0-9]+$")
    target_hotspots: list[int | str] = Field(default_factory=list)
    contigs: list[str] | None = None
    length_min: int = Field(default=100, ge=20, le=5000)
    length_max: int | None = Field(default=None, ge=20, le=5000)
    total_length: int | None = Field(default=None, ge=20, le=5000)
    binder_length_min: int = Field(default=70, ge=20, le=1000)
    binder_length_max: int | None = Field(default=None, ge=20, le=1000)
    symmetry: str | None = Field(default=None, min_length=1)
    num_designs: int = Field(default=10, ge=1, le=256)
    num_diffusion_steps: int = Field(default=50, ge=15, le=200)
    partial_T: int | None = Field(default=None, ge=1, le=200)
    noise_scale_ca: float = Field(default=1.0, ge=0.0, le=1.0)
    noise_scale_frame: float = Field(default=1.0, ge=0.0, le=1.0)
    inpaint_seq: list[str] | None = None
    inpaint_str: list[str] | None = None
    provide_seq: list[str] | None = None
    contig_length: str | None = Field(default=None, pattern=r"^[0-9]+-[0-9]+$")
    checkpoint_path: str | None = Field(
        default=None,
        json_schema_extra={"vpa:digest": "sha256-if-path"},
    )
    model_directory_path: str | None = Field(
        default=None,
        json_schema_extra={"vpa:digest": "sha256-if-path"},
    )
    scaffold_dir: str | None = Field(
        default=None,
        json_schema_extra={"vpa:digest": "sha256-if-path"},
    )
    target_ss: str | None = Field(
        default=None,
        json_schema_extra={"vpa:digest": "sha256-if-path"},
    )
    target_adj: str | None = Field(
        default=None,
        json_schema_extra={"vpa:digest": "sha256-if-path"},
    )
    guiding_potentials: list[str] | None = None
    guide_scale: float | None = Field(default=None, ge=0.0)
    guide_decay: Literal["constant", "linear", "quadratic", "cubic"] | None = None
    olig_intra_all: bool | None = None
    olig_inter_all: bool | None = None
    hydra_overrides: list[str] = Field(default_factory=list)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_mode_contract(self) -> RFdiffusionInput:
        self.mode = _canonical_mode(self.mode)
        if self.length_max is not None and self.length_max < self.length_min:
            raise ValueError("length_max must be >= length_min")
        if (
            self.binder_length_max is not None
            and self.binder_length_max < self.binder_length_min
        ):
            raise ValueError("binder_length_max must be >= binder_length_min")
        if self.partial_T is not None and self.partial_T > self.num_diffusion_steps:
            raise ValueError("partial_T must be <= num_diffusion_steps")

        needs_target = {
            "motif_scaffolding",
            "symmetric_motif_scaffolding",
            "partial_diffusion",
            "binder_design",
            "fold_conditioning",
        }
        if self.mode in needs_target and self.target_pdb is None:
            raise ValueError(f"{self.mode} requires target_pdb")
        if "symmetric" in self.mode and not self.symmetry:
            raise ValueError(f"{self.mode} requires symmetry")
        if self.mode in {"motif_scaffolding", "symmetric_motif_scaffolding"}:
            if not self.contigs:
                raise ValueError(f"{self.mode} requires explicit contigs")
        if self.mode == "partial_diffusion" and self.partial_T is None:
            raise ValueError("partial_diffusion requires partial_T")
        if self.mode == "fold_conditioning":
            missing = [
                name
                for name in ("scaffold_dir", "target_ss", "target_adj")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "fold_conditioning requires " + ", ".join(missing)
                )
        return self


class RFdiffusionDesign(BaseModel):
    pdb_path: str
    trb_path: str | None = None
    sequence: str | None = None
    length: int
    score: float | None = None


class RFdiffusionOutput(BaseModel):
    mode: RFdiffusionMode
    contigs: list[str]
    designs: list[RFdiffusionDesign]
    backbone_dir: str = ""
    manifest_jsonl: str = ""
    top_candidates: list[RFdiffusionDesign] = Field(default_factory=list)
    runtime_ms: int
    num_requested: int = 0
    num_generated: int = 0
    num_designs_returned: int


@register_adapter("descriptor.yaml")
class RFdiffusionAdapter(ToolAdapter):
    """General RFdiffusion backbone generation adapter."""

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        del config, weights_dir
        self.device = device
        _check_paths()

    @torch.no_grad()
    def run(self, params: RFdiffusionInput, ctx: AdapterRunContext) -> RFdiffusionOutput:
        t0 = time.time()
        ctx.emit_progress(0.02)

        if not torch.cuda.is_available():
            raise RuntimeError("RFdiffusion adapter requires CUDA")

        run_inputs = ctx.job_dir / "inputs"
        run_outputs = ctx.job_dir / "outputs"
        run_inputs.mkdir(parents=True, exist_ok=True)
        run_outputs.mkdir(parents=True, exist_ok=True)

        staged_target = _stage_file(params.target_pdb, run_inputs)
        staged_target_ss = _stage_file(params.target_ss, run_inputs)
        staged_target_adj = _stage_file(params.target_adj, run_inputs)

        contigs = _build_contigs(params, staged_target)
        checkpoint = _required_weight(params)
        if not checkpoint.is_file():
            mode = _canonical_mode(params.mode)
            raise FileNotFoundError(
                f"Required RFdiffusion checkpoint for mode={mode!r} missing at "
                f"{checkpoint}. "
                "Download the RFdiffusion weights listed in "
                "vendor/rfpeptides/UPSTREAM.md or set checkpoint_path."
            )

        ctx.emit_progress(0.10)
        cmd = _build_cmd(
            params=params,
            output_prefix=run_outputs / "design",
            contigs=contigs,
            target_pdb=staged_target,
            target_ss=staged_target_ss,
            target_adj=staged_target_adj,
            checkpoint=checkpoint,
        )

        try:
            subprocess.run(
                cmd,
                check=True,
                env=_build_env(params.seed),
                cwd=str(run_outputs),
                timeout=4 * 3600,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"RFdiffusion subprocess failed with exit code {exc.returncode}"
            ) from exc

        ctx.emit_progress(0.92)
        designs = _collect_designs(run_outputs / "design", contigs)
        manifest_jsonl = _write_manifest_jsonl(run_outputs / "backbones.jsonl", designs)
        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return RFdiffusionOutput(
            mode=params.mode,
            contigs=contigs,
            designs=designs,
            backbone_dir=str(run_outputs),
            manifest_jsonl=str(manifest_jsonl),
            top_candidates=designs[: min(5, len(designs))],
            runtime_ms=runtime_ms,
            num_requested=params.num_designs,
            num_generated=len(designs),
            num_designs_returned=len(designs),
        )

    def estimate_vram(self, params: RFdiffusionInput) -> int:
        if params.mode in {"binder_design", "fold_conditioning"}:
            return 24 * 1024
        return 16 * 1024

    def healthcheck(self) -> bool:
        return bool(availability_report().get("available"))

    def unload(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def availability_report() -> dict[str, object]:
    """Return descriptor-level readiness without hiding mode-specific gaps.

    RFdiffusion can be partially ready: the base checkpoint is enough for
    unconditional, motif, partial-diffusion, and symmetric base workflows, while
    binder and fold-conditioned workflows need extra checkpoints. The registry
    should still show the tool as available for base modes, but expose the
    missing optional mode checkpoints for agent planning and operator reporting.
    """

    checks: dict[str, object] = {}
    failures: list[str] = []

    run_inference = RFDIFFUSION_TREE / "scripts" / "run_inference.py"
    tree_ok = RFDIFFUSION_TREE.is_dir() and run_inference.is_file()
    checks["tree"] = {
        "ok": tree_ok,
        "path": str(RFDIFFUSION_TREE),
        "run_inference": str(run_inference),
    }
    if not tree_ok:
        failures.append(
            f"RFdiffusion tree not present at {RFDIFFUSION_TREE}. "
            "Set VPA_RFDIFFUSION_TREE to a RosettaCommons/RFdiffusion checkout."
        )

    checkpoint_checks = _mode_checkpoint_checks(RFDIFFUSION_WEIGHTS)
    checks["mode_checkpoints"] = checkpoint_checks
    base_ok = bool(checkpoint_checks["unconditional"]["ok"])
    checks["base_checkpoint"] = checkpoint_checks["unconditional"]
    missing_mode_checkpoints = [
        mode
        for mode, item in checkpoint_checks.items()
        if not bool(item["ok"])
    ]
    checks["missing_mode_checkpoints"] = missing_mode_checkpoints
    checks["full_mode_set_available"] = not missing_mode_checkpoints
    if not base_ok:
        failures.append(
            f"RFdiffusion {_BASE_CKPT} missing under {RFDIFFUSION_WEIGHTS}. "
            "Download weights per vendor/rfpeptides/UPSTREAM.md or set "
            "VPA_RFDIFFUSION_WEIGHTS."
        )

    try:
        dgl = import_dgl_with_graphbolt_guard()
    except Exception as exc:
        checks["dgl"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        failures.append(
            f"DGL import failed under this env: {type(exc).__name__}: {exc}. "
            "RFdiffusion needs the CUDA DGL stack described in "
            "vendor/rfpeptides/UPSTREAM.md."
        )
    else:
        checks["dgl"] = {"ok": True}
        if not torch.cuda.is_available():
            checks["cuda"] = {"ok": False}
            failures.append("RFdiffusion requires CUDA; no CUDA device is visible.")
        else:
            checks["cuda"] = {"ok": True}
            try:
                g = dgl.graph(([0, 1], [1, 0])).to("cuda")
                _ = g.all_edges()
            except Exception as exc:
                checks["dgl_cuda"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(
                    f"DGL CUDA graph ops unavailable: {type(exc).__name__}: {exc}. "
                    "Install a CUDA-enabled DGL build for this torch/CUDA stack."
                )
            else:
                checks["dgl_cuda"] = {"ok": True}

    return {
        "available": not failures,
        "reason": "; ".join(failures) if failures else None,
        "checks": checks,
    }


def availability_skip_reason() -> str | None:
    report = availability_report()
    reason = report.get("reason")
    return str(reason) if reason else None


def mode_readiness(params: dict[str, object] | RFdiffusionInput | str) -> dict[str, object]:
    """Return checkpoint readiness for the concrete RFdiffusion mode request."""
    if isinstance(params, RFdiffusionInput):
        request = params
    elif isinstance(params, str):
        request = RFdiffusionInput(mode=params)
    else:
        request = RFdiffusionInput.model_validate(params)
    checkpoint = _required_weight(request)
    ready = checkpoint.is_file()
    mode = _canonical_mode(request.mode)
    return {
        "ready": ready,
        "mode": mode,
        "checkpoint": str(checkpoint),
        "reason": None if ready else f"RFdiffusion checkpoint for mode={mode!r} is missing",
    }


def _check_paths() -> None:
    if not RFDIFFUSION_TREE.is_dir():
        raise FileNotFoundError(
            f"RFdiffusion tree not found at {RFDIFFUSION_TREE}. "
            "Set VPA_RFDIFFUSION_TREE to a RosettaCommons/RFdiffusion checkout."
        )
    if not (RFDIFFUSION_TREE / "scripts" / "run_inference.py").is_file():
        raise FileNotFoundError(
            f"{RFDIFFUSION_TREE / 'scripts' / 'run_inference.py'} not found."
        )


def _stage_file(raw: str | None, inputs_dir: Path) -> Path | None:
    if raw is None:
        return None
    src = Path(raw).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"input file not found: {src}")
    dest = inputs_dir / src.name
    if src != dest:
        shutil.copyfile(src, dest)
    return dest


def _required_weight(params: RFdiffusionInput) -> Path:
    if params.checkpoint_path:
        return Path(params.checkpoint_path).expanduser()
    weights = (
        Path(params.model_directory_path).expanduser()
        if params.model_directory_path
        else RFDIFFUSION_WEIGHTS
    )
    mode = _canonical_mode(params.mode)
    if mode == "fold_conditioning":
        return weights / _FOLD_CKPT
    if mode == "binder_design":
        return weights / _COMPLEX_CKPT
    return weights / _BASE_CKPT


def _mode_checkpoint_checks(weights: Path) -> dict[str, dict[str, object]]:
    base = weights / _BASE_CKPT
    complex_ckpt = weights / _COMPLEX_CKPT
    fold_ckpt = weights / _FOLD_CKPT
    return {
        "unconditional": _checkpoint_check(base),
        "motif_scaffolding": _checkpoint_check(base),
        "symmetric_motif_scaffolding": _checkpoint_check(base),
        "partial_diffusion": _checkpoint_check(base),
        "symmetric_oligomer": _checkpoint_check(base),
        "binder_design": _checkpoint_check(complex_ckpt),
        "fold_conditioning": _checkpoint_check(fold_ckpt),
    }


def _checkpoint_check(path: Path) -> dict[str, object]:
    return {"ok": path.is_file(), "path": str(path), "checkpoint": path.name}


def _build_contigs(params: RFdiffusionInput, target_pdb: Path | None) -> list[str]:
    mode = _canonical_mode(params.mode)
    if params.contigs:
        return params.contigs
    if mode == "unconditional":
        return [_range(params.length_min, params.length_max)]
    if mode == "symmetric_oligomer":
        length = params.total_length or params.length_min
        return [_range(length, params.length_max or length)]
    if mode == "partial_diffusion":
        assert target_pdb is not None
        span_len = _chain_length(target_pdb, params.target_chain)
        return [f"{span_len}-{span_len}"]
    if mode == "binder_design":
        assert target_pdb is not None
        target_span = _target_span(params, target_pdb)
        binder = _range(params.binder_length_min, params.binder_length_max)
        return [f"{target_span}/0 {binder}"]
    if mode == "fold_conditioning":
        return []
    raise ValueError(f"{mode} requires explicit contigs")


def _range(start: int, end: int | None) -> str:
    stop = start if end is None else end
    return f"{start}-{stop}"


def _target_span(params: RFdiffusionInput, target_pdb: Path) -> str:
    chain = params.target_chain.upper()
    if params.target_residues is not None:
        return f"{chain}{params.target_residues}"
    start, end = _longest_contiguous_span(target_pdb, chain)
    return f"{chain}{start}-{end}"


def _chain_length(pdb: Path, chain: str) -> int:
    residues = _chain_residues(pdb, chain)
    if not residues:
        raise ValueError(f"No ATOM residues found for chain {chain} in {pdb}")
    return len(residues)


def _longest_contiguous_span(pdb: Path, chain: str) -> tuple[int, int]:
    residues = _chain_residues(pdb, chain)
    if not residues:
        raise ValueError(f"No ATOM residues found for chain {chain} in {pdb}")
    runs: list[tuple[int, int]] = []
    start = previous = residues[0]
    for residue in residues[1:]:
        if residue == previous + 1:
            previous = residue
            continue
        runs.append((start, previous))
        start = previous = residue
    runs.append((start, previous))
    return max(runs, key=lambda pair: (pair[1] - pair[0], -pair[0]))


def _chain_residues(pdb: Path, chain: str) -> list[int]:
    values: set[int] = set()
    with pdb.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if len(line) <= 26 or line[21].strip().upper() != chain.upper():
                continue
            try:
                values.add(int(line[22:26]))
            except ValueError:
                continue
    return sorted(values)


def _build_cmd(
    *,
    params: RFdiffusionInput,
    output_prefix: Path,
    contigs: list[str],
    target_pdb: Path | None,
    target_ss: Path | None,
    target_adj: Path | None,
    checkpoint: Path,
) -> list[str]:
    run_inference = RFDIFFUSION_TREE / "scripts" / "run_inference.py"
    config_name = "symmetry" if _uses_symmetry_config(params) else "base"
    argv = [
        str(run_inference),
        "--config-name",
        config_name,
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={params.num_designs}",
        f"diffuser.T={params.num_diffusion_steps}",
        f"denoiser.noise_scale_ca={params.noise_scale_ca}",
        f"denoiser.noise_scale_frame={params.noise_scale_frame}",
        f"inference.model_directory_path={checkpoint.parent}",
    ]
    if checkpoint.name not in {_BASE_CKPT} or params.checkpoint_path:
        argv.append(f"inference.ckpt_override_path={checkpoint}")
    mode = _canonical_mode(params.mode)
    if target_pdb is not None and mode != "fold_conditioning":
        argv.append(f"inference.input_pdb={target_pdb}")
    if contigs:
        argv.append(f"contigmap.contigs={_hydra_list(contigs)}")
    if params.contig_length:
        argv.append(f"contigmap.length={params.contig_length}")
    if params.partial_T is not None:
        argv.append(f"diffuser.partial_T={params.partial_T}")
    if params.symmetry:
        argv.append(f"inference.symmetry={params.symmetry}")
    if params.target_hotspots:
        argv.append(f"ppi.hotspot_res={_hotspot_list(params)}")
    argv.extend(_optional_list_override("contigmap.inpaint_seq", params.inpaint_seq))
    argv.extend(_optional_list_override("contigmap.inpaint_str", params.inpaint_str))
    argv.extend(_optional_list_override("contigmap.provide_seq", params.provide_seq))
    if mode == "fold_conditioning":
        assert target_pdb is not None
        assert target_ss is not None
        assert target_adj is not None
        assert params.scaffold_dir is not None
        argv.extend(
            [
                "scaffoldguided.scaffoldguided=True",
                "scaffoldguided.target_pdb=True",
                f"scaffoldguided.target_path={target_pdb}",
                f"scaffoldguided.target_ss={target_ss}",
                f"scaffoldguided.target_adj={target_adj}",
                f"scaffoldguided.scaffold_dir={Path(params.scaffold_dir).expanduser()}",
            ]
        )
    if params.guiding_potentials:
        argv.append(f"potentials.guiding_potentials={_hydra_list(params.guiding_potentials)}")
    if params.guide_scale is not None:
        argv.append(f"potentials.guide_scale={params.guide_scale}")
    if params.guide_decay is not None:
        argv.append(f"potentials.guide_decay={params.guide_decay}")
    if params.olig_intra_all is not None:
        argv.append(f"potentials.olig_intra_all={params.olig_intra_all!s}")
    if params.olig_inter_all is not None:
        argv.append(f"potentials.olig_inter_all={params.olig_inter_all!s}")
    argv.extend(params.hydra_overrides)

    preload = (
        "import sys, types; "
        "sys.modules.setdefault('dgl.graphbolt', types.ModuleType('dgl.graphbolt')); "
        "import sys as _s; _s.argv = "
        + repr(argv)
        + "; import runpy; runpy.run_path("
        + repr(str(run_inference))
        + ", run_name='__main__')"
    )
    return [sys.executable, "-c", preload]


def _uses_symmetry_config(params: RFdiffusionInput) -> bool:
    return _canonical_mode(params.mode) in {
        "symmetric_motif_scaffolding",
        "symmetric_oligomer",
    }


def _canonical_mode(mode: str) -> str:
    return {
        "backbone_generation": "unconditional",
        "unconditional_generation": "unconditional",
        "binder_backbone_design": "binder_design",
        "symmetric_oligomer_design": "symmetric_oligomer",
    }.get(mode, mode)


def _hydra_list(values: list[str]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _optional_list_override(key: str, values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [f"{key}={_hydra_list(values)}"]


def _hotspot_list(params: RFdiffusionInput) -> str:
    chain = params.target_chain.upper()
    hotspots = []
    for value in params.target_hotspots:
        text = str(value).strip()
        hotspots.append(text if text[:1].isalpha() else f"{chain}{text}")
    return "[" + ",".join(hotspots) + "]"


def _build_env(seed: int | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = ":".join(
        [str(RFDIFFUSION_TREE), str(RFDIFFUSION_TREE / "env" / "SE3Transformer")]
    )
    env["DGLBACKEND"] = "pytorch"
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
    return env


def _collect_designs(prefix: Path, contigs: list[str]) -> list[RFdiffusionDesign]:
    parent = prefix.parent
    stem = prefix.name
    pdbs = sorted(parent.glob(f"{stem}_*.pdb"))
    if not pdbs:
        raise RuntimeError(
            f"RFdiffusion produced no PDB outputs under {parent} with stem '{stem}'."
        )
    length_hint = _length_hint(contigs)
    designs: list[RFdiffusionDesign] = []
    for pdb in pdbs:
        trb = pdb.with_suffix(".trb")
        designs.append(
            RFdiffusionDesign(
                pdb_path=str(pdb),
                trb_path=str(trb) if trb.is_file() else None,
                sequence=None,
                length=_pdb_ca_count(pdb) or length_hint,
                score=None,
            )
        )
    return designs


def _write_manifest_jsonl(path: Path, designs: list[RFdiffusionDesign]) -> Path:
    with path.open("w") as handle:
        for index, design in enumerate(designs):
            handle.write(
                json.dumps(
                    {
                        "rank": index + 1,
                        "pdb_path": design.pdb_path,
                        "trb_path": design.trb_path,
                        "sequence": design.sequence,
                        "length": design.length,
                        "score": design.score,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def _length_hint(contigs: list[str]) -> int:
    for contig in contigs:
        first = contig.split("/", 1)[0].split(" ", 1)[0]
        if "-" in first and not first[0].isalpha():
            lo, _, hi = first.partition("-")
            try:
                return int(hi or lo)
            except ValueError:
                continue
    return 1


def _pdb_ca_count(pdb: Path) -> int:
    count = 0
    with pdb.open() as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                count += 1
    return count


__all__ = [
    "RFDIFFUSION_TREE",
    "RFDIFFUSION_WEIGHTS",
    "RFdiffusionAdapter",
    "RFdiffusionDesign",
    "RFdiffusionInput",
    "RFdiffusionOutput",
    "availability_skip_reason",
    "mode_readiness",
]
