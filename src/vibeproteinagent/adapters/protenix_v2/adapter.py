"""Real Protenix-v2 adapter.

The adapter consumes a Protenix checkout read-only from ``VPA_PROTENIX_TREE``.
When running from a source checkout, it can also use the repository's
``vendor/protenix-pxd`` tree as a local fallback.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.msa_policy import (
    MSABackend,
    json_file_has_prepared_msa,
    should_use_remote_msa,
    strip_prepared_msa,
)
from vibeproteinagent.adapters.residue_mapping import parse_structure_residue_refs
from vibeproteinagent.candidates.capabilities import candidate_capability_flags
from vibeproteinagent.candidates.schema import stable_candidate_id

# --------------------------------------------------------------------------- paths

def _default_protenix_tree() -> str:
    vpa_home = os.environ.get("VPA_HOME")
    if vpa_home:
        return str(Path(vpa_home).expanduser() / "sidecars" / "protenix")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "vendor" / "protenix-pxd"
        if (candidate / "runner").is_dir() and (candidate / "protenix").is_dir():
            return str(candidate)
    return str(Path("~/.vibeproteinagent/sidecars/protenix").expanduser())


PROTENIX_TREE_DEFAULT = _default_protenix_tree()


def _default_protenix_ckpt_dir() -> str:
    vpa_home = os.environ.get("VPA_HOME")
    if vpa_home:
        return str(Path(vpa_home).expanduser() / "cache" / "protenix")
    return str(Path("~/.vibeproteinagent/cache/protenix").expanduser())


PROTENIX_CKPT_DEFAULT = _default_protenix_ckpt_dir()
PROTENIX_MODEL_NAME = os.environ.get(
    "VPA_PROTENIX_MODEL_NAME", "protenix_base_default_v1.0.0"
)

PROTENIX_TREE = Path(
    os.environ.get("VPA_PROTENIX_TREE", PROTENIX_TREE_DEFAULT)
).expanduser()
PROTENIX_CKPT_DIR = Path(
    os.environ.get("VPA_PROTENIX_CKPT", PROTENIX_CKPT_DEFAULT)
).expanduser()


def _resolve_checkpoint() -> Path:
    """Pick the Protenix base checkpoint, validating it exists."""
    ckpt = PROTENIX_CKPT_DIR / f"{PROTENIX_MODEL_NAME}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Protenix checkpoint not found at {ckpt}. "
            f"Set $VPA_PROTENIX_CKPT to the directory holding "
            f"{PROTENIX_MODEL_NAME}.pt (and its common/ CCD cache)."
        )
    return ckpt


def _stub_fair_esm_symbols() -> None:
    """Inject ``FastaBatchedDataset`` / ``pretrained`` into the installed ``esm``.

    Protenix's ``protenix.data.esm.compute_esm`` (executed during the
    ``protenix.data.inference`` package import chain) does
    ``from esm import FastaBatchedDataset, pretrained`` at module top — those
    names live in the *legacy* ``fair-esm`` 2.x package. The vpa env ships
    ``esm`` 3.2.4 (EvolutionaryScale's ESM-3 community fork), which doesn't
    expose them. We never use the ESM path (model_name = base, no "_esm" /
    "_ism" suffix), so stubs that raise on call are enough to satisfy the
    import without changing behaviour.
    """
    try:
        import esm as _esm
    except ImportError:
        return

    if not hasattr(_esm, "FastaBatchedDataset"):
        class _StubFastaBatchedDataset:  # pragma: no cover - never instantiated
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise NotImplementedError(
                    "fair-esm's FastaBatchedDataset is not available in this env; "
                    "Protenix ESM-augmented variants are not supported here."
                )

        _esm.FastaBatchedDataset = _StubFastaBatchedDataset  # type: ignore[attr-defined]

    if not hasattr(_esm, "pretrained"):
        import types as _types

        _pretrained = _types.ModuleType("esm.pretrained")

        def _raise(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
            raise NotImplementedError(
                "fair-esm's pretrained loaders are not available in this env."
            )

        _pretrained.load_model_and_alphabet_local = _raise
        _pretrained.load_model_and_alphabet = _raise
        _esm.pretrained = _pretrained  # type: ignore[attr-defined]
        sys.modules.setdefault("esm.pretrained", _pretrained)


def _ensure_protenix_importable() -> None:
    """Insert the Protenix tree onto sys.path and seed PROTENIX_ROOT_DIR.

    Protenix's data and inference config modules read ``$PROTENIX_ROOT_DIR``
    at IMPORT time to compute default CCD / checkpoint paths, so we have to
    set the env var *before* importing anything from Protenix.
    """
    if not PROTENIX_TREE.is_dir():
        raise FileNotFoundError(
            f"Protenix source tree not found at {PROTENIX_TREE}. "
            f"Set $VPA_PROTENIX_TREE to a Protenix checkout."
        )
    if not (PROTENIX_TREE / "runner").is_dir() or not (PROTENIX_TREE / "protenix").is_dir():
        raise FileNotFoundError(
            f"{PROTENIX_TREE} does not look like a Protenix checkout "
            f"(missing 'runner/' or 'protenix/')."
        )
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(PROTENIX_CKPT_DIR))
    env_bin = Path(sys.executable).resolve().parent
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(env_bin) not in path_parts:
        os.environ["PATH"] = os.pathsep.join([str(env_bin), *path_parts])
    p = str(PROTENIX_TREE)
    if p not in sys.path:
        sys.path.insert(0, p)
    for prefix in ("protenix", "runner", "configs"):
        existing = sys.modules.get(prefix)
        existing_file = Path(getattr(existing, "__file__", "") or "")
        if existing is not None and not existing_file.resolve().is_relative_to(PROTENIX_TREE):
            for mod_name in list(sys.modules):
                if mod_name == prefix or mod_name.startswith(f"{prefix}."):
                    sys.modules.pop(mod_name, None)
    _stub_fair_esm_symbols()


# --------------------------------------------------------------------------- I/O models


class ProtenixV2Input(BaseModel):
    input_json: str | None = Field(
        None, description="Absolute path to a Protenix-format input JSON."
    )
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=lambda: [101], min_length=1)
    use_msa: bool = Field(default=False)
    msa_backend: MSABackend = Field(default="auto")
    use_template: bool = Field(default=False)
    use_rna_msa: bool = Field(default=False)
    enable_cache: bool = Field(default=True)
    enable_tf32: bool = Field(default=True)
    n_sample: int = Field(default=5, ge=1, le=25)
    n_step: int = Field(default=200, ge=10, le=1000)
    n_cycle: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="before")
    @classmethod
    def _validate_canonical_candidates(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        candidates = data.get("candidates")
        if not candidates:
            return data
        canonical = [_is_canonical_candidate(item) for item in candidates]
        if any(canonical) and not all(canonical):
            raise ValueError(
                "Mixed canonical and legacy candidate items are not supported. "
                "Pass either all canonical Candidate[] or all legacy items."
            )
        if not all(canonical):
            return data
        for candidate in candidates:
            if not candidate.get("sequence"):
                raise ValueError("protenix_v2 canonical Candidate[] input requires sequence")
        return data

    @model_validator(mode="after")
    def _has_input_source(self) -> ProtenixV2Input:
        has_canonical_candidates = bool(self.candidates) and all(
            _is_canonical_candidate(item) for item in self.candidates
        )
        if not self.input_json and not has_canonical_candidates:
            raise ValueError("protenix_v2 requires input_json or canonical Candidate[]")
        return self


class ProtenixV2PredictionRecord(BaseModel):
    candidate_id: str | None = None
    seed: int
    rank: int
    cif_path: str
    pdb_path: str | None = None
    summary_json: str
    pae_path: str | None = None
    full_confidence_json: str | None = None
    ranking_score: float | None = None
    plddt: float | None = None
    iptm: float | None = None
    ptm: float | None = None


class ProtenixV2Metrics(BaseModel):
    plddt: float
    iptm: float
    ptm: float
    ranking_score: float
    summary_json: str | None = None


class ProtenixV2Output(BaseModel):
    predicted_cif: str
    predicted_pdb: str | None = None
    pae_path: str | None = None
    metrics: ProtenixV2Metrics
    all_predictions: list[ProtenixV2PredictionRecord]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    runtime_ms: int
    seeds_used: list[int]


# --------------------------------------------------------------------------- helpers


def _read_summary(path: Path) -> dict[str, Any]:
    """Parse a Protenix summary_confidence_sample_<rank>.json file."""
    with path.open() as fh:
        raw = json.load(fh)
    # Protenix's DataDumper saves a per-sample dict (post round_values) keyed by
    # the metric names: plddt, iptm, ptm, ranking_score, gpde, has_clash, ...
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to an object (got {type(raw).__name__})")
    return raw


def _coerce_float(value: Any) -> float | None:
    """Best-effort scalar coercion — Protenix sometimes hands back a [1] tensor/list."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if not value:
            return None
        return _coerce_float(value[0])
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _max_numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        vals = [_max_numeric(item) for item in value]
        vals = [item for item in vals if item is not None]
        return max(vals) if vals else None
    return None


def _scale_unit_interval(value: Any) -> Any:
    max_value = _max_numeric(value)
    if max_value is None or max_value > 1.0:
        return value
    if isinstance(value, list):
        return [_scale_unit_interval(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value) * 100.0, 2)
    return value


def _write_ipsae_pae_json(full_data_path: Path, summary_path: Path) -> Path | None:
    """Convert Protenix full confidence JSON into the AF3-like JSON ipSAE accepts."""
    if not full_data_path.is_file():
        return None
    with full_data_path.open() as fh:
        full_data = json.load(fh)
    if not isinstance(full_data, dict):
        return None
    pae = full_data.get("pae") or full_data.get("token_pair_pae")
    if pae is None:
        return None

    stem = full_data_path.stem
    if "_full_data_sample_" in stem:
        out_stem = stem.replace("_full_data_sample_", "_confidences_sample_")
    else:
        out_stem = f"{stem}_confidences"
    pae_path = full_data_path.with_name(f"{out_stem}.json")
    payload: dict[str, Any] = {"pae": pae}
    atom_plddt = full_data.get("atom_plddts", full_data.get("atom_plddt"))
    if atom_plddt is not None:
        payload["atom_plddts"] = _scale_unit_interval(atom_plddt)
    pae_path.write_text(json.dumps(payload, separators=(",", ":")))

    if summary_path.is_file():
        summary_stem = out_stem.replace("confidences", "summary_confidences")
        summary_out = full_data_path.with_name(f"{summary_stem}.json")
        if summary_out != summary_path:
            summary_out.write_text(summary_path.read_text())
    return pae_path


def _convert_mmcif_to_pdb(cif_path: Path, warnings: list[str]) -> Path | None:
    """Materialize a best-effort PDB sibling without changing CIF-primary output."""
    pdb_path = cif_path.with_suffix(".pdb")
    try:
        from Bio.PDB import MMCIFParser, PDBIO

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(cif_path.stem, str(cif_path))
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(pdb_path))
    except Exception as exc:
        warnings.append(
            "could not convert Protenix mmCIF to PDB for PDB-only downstream "
            f"consumers: {type(exc).__name__}: {exc}"
        )
        return None
    if not pdb_path.is_file() or pdb_path.stat().st_size == 0:
        warnings.append(
            f"could not convert Protenix mmCIF to PDB: empty output at {pdb_path}"
        )
        return None
    return pdb_path


def _confidence_sample_name(prediction_sample_name: str, seed: int) -> str:
    seed_suffix = f"_seed_{seed}"
    if prediction_sample_name.endswith(seed_suffix):
        return prediction_sample_name.removesuffix(seed_suffix)
    return prediction_sample_name


def _supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return only kwargs accepted by a runtime-loaded third-party callable."""
    signature = inspect.signature(fn)
    if any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _ensure_model_config(model_name: str) -> None:
    """Make runtime-patched Protenix trees accept the installed checkpoint name."""
    configs_model_type = importlib.import_module("configs.configs_model_type")
    model_configs = configs_model_type.model_configs
    if model_name in model_configs:
        return
    family = "_".join(model_name.split("_")[:3])
    candidates = [
        name for name in model_configs if name.startswith(f"{family}_v")
    ]
    if not candidates:
        return
    compat_config = deepcopy(model_configs[sorted(candidates)[-1]])
    compat_config["load_strict"] = False
    model_configs[model_name] = compat_config


# --------------------------------------------------------------------------- adapter


@register_adapter("descriptor.yaml")
class ProtenixV2Adapter(ToolAdapter):
    """Adapter for ByteDance's Protenix v1.0.0 (AlphaFold3 reproduction)."""

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        # ``weights_dir`` is supplied by the worker as a placeholder (currently
        # Path("/tmp"); see workers/worker.py:333 and project note
        # 962e4227-…); we ignore it and resolve via env vars instead.
        del config, weights_dir
        self.device = device
        self.ckpt_path = _resolve_checkpoint()
        _ensure_protenix_importable()

        # Bind Protenix imports at construction time so import-time CCD path
        # config is captured under the correct PROTENIX_ROOT_DIR.
        batch_inference = importlib.import_module("runner.batch_inference")
        inference_mod = importlib.import_module("runner.inference")
        configs_inference_mod = importlib.import_module("configs.configs_inference")
        _ensure_model_config(PROTENIX_MODEL_NAME)

        # Point Protenix at the local checkpoint dir (not the default
        # ~/checkpoint). batch_inference reads inference_configs at runtime
        # inside get_default_runner.
        self._configs_inference = configs_inference_mod.inference_configs
        self._configs_inference["load_checkpoint_dir"] = str(PROTENIX_CKPT_DIR)
        self._configs_inference["dump_dir"] = str(Path.cwd())  # overridden per-run
        self._configs_inference["num_workers"] = 0

        # Build the runner warm — this loads the model + weights into VRAM
        # and runs download_inference_cache (no-op since everything is on disk).
        # Force MSA off for the construction-time defaults; per-run we override
        # the dataloader-affecting flags below.
        # ``cuequivariance`` would be ~3x faster but pulls a ~600 MiB CUDA dep
        # we don't have locally; the pure-torch kernels work and the spec
        # accepts the trade-off (Phase 4.2 is about correctness, not perf).
        runner_kwargs = dict(
            seeds=[101],
            n_cycle=10,
            n_step=200,
            n_sample=5,
            model_name=PROTENIX_MODEL_NAME,
            use_msa=False,
            trimul_kernel="torch",
            triatt_kernel="torch",
        )
        self.runner = batch_inference.get_default_runner(
            **_supported_kwargs(batch_inference.get_default_runner, runner_kwargs)
        )
        self.runner.configs.need_atom_confidence = True
        self.runner.configs.num_workers = 0
        self.runner.dumper.need_atom_confidence = True
        self._infer_predict = inference_mod.infer_predict
        self._preprocess_input = getattr(
            batch_inference,
            "preprocess_input",
            lambda input_json, out_dir, use_msa, **_kwargs: batch_inference.update_infer_json(
                input_json,
                out_dir=out_dir,
                use_msa=use_msa,
            ),
        )

    @torch.no_grad()
    def run(self, params: ProtenixV2Input, ctx: AdapterRunContext) -> ProtenixV2Output:
        t0 = time.time()
        ctx.emit_progress(0.02)

        job_inputs = ctx.job_dir / "inputs"
        job_outputs = ctx.job_dir / "outputs"
        job_inputs.mkdir(parents=True, exist_ok=True)
        job_outputs.mkdir(parents=True, exist_ok=True)

        if params.input_json:
            input_path = Path(params.input_json).expanduser()
            if not input_path.is_file():
                raise FileNotFoundError(f"input_json not found: {input_path}")
        else:
            input_path = _write_protenix_input_json_from_candidates(
                params.candidates,
                params.seeds,
                job_inputs / "canonical_candidates.json",
            )

        # Copy the input json into the job dir so the artifact is self-contained.
        staged = job_inputs / input_path.name
        staged.write_text(input_path.read_text())
        ctx.emit_progress(0.05)

        # Optionally run Protenix's MSA / template / RNA-MSA preprocess. With
        # all three flags off (default for tests) this is a no-op and returns
        # the input path unchanged.
        has_input_msa = json_file_has_prepared_msa(staged)
        remote_msa = should_use_remote_msa(
            use_msa=params.use_msa,
            msa_backend=params.msa_backend,
            has_prepared_msa=has_input_msa,
        )
        effective_use_msa = params.use_msa and params.msa_backend != "none"
        if not effective_use_msa or params.msa_backend == "remote":
            staged.write_text(json.dumps(strip_prepared_msa(json.loads(staged.read_text()))))
        prepared_json = self._preprocess_input(
            str(staged),
            out_dir=str(job_outputs),
            use_msa=remote_msa,
            use_template=params.use_template,
            use_rna_msa=params.use_rna_msa,
            msa_server_mode="protenix",
        )
        ctx.emit_progress(0.10)

        # Apply the per-run config overrides on the warm runner.
        cfg = self.runner.configs
        cfg["input_json_path"] = str(prepared_json)
        cfg["dump_dir"] = str(job_outputs)
        cfg["seeds"] = list(params.seeds)
        cfg["num_workers"] = 0
        cfg.use_msa = effective_use_msa
        cfg.use_template = params.use_template
        cfg.use_rna_msa = params.use_rna_msa
        cfg.enable_diffusion_shared_vars_cache = params.enable_cache
        cfg.enable_tf32 = params.enable_tf32
        cfg.model.N_cycle = params.n_cycle
        cfg.sample_diffusion.N_sample = params.n_sample
        cfg.sample_diffusion.N_step = params.n_step

        # DataDumper is constructor-bound to base_dir; rewrite it for this call.
        self.runner.dump_dir = str(job_outputs)
        self.runner.dumper.base_dir = str(job_outputs)
        self.runner.dumper.need_atom_confidence = True
        self.runner.error_dir = str(job_outputs / "ERR")
        Path(self.runner.error_dir).mkdir(parents=True, exist_ok=True)

        ctx.emit_progress(0.15)
        # The heavy lifting: model forward over every (seed x sample) cross.
        self._infer_predict(self.runner, cfg)
        ctx.emit_progress(0.90)

        all_records, best, warnings = self._collect_outputs(job_outputs, params.seeds)
        if best is None:
            raise RuntimeError(
                f"Protenix produced no predictions under {job_outputs}. "
                f"Check {self.runner.error_dir} for per-sample errors."
            )

        ctx.emit_progress(1.0)
        runtime_ms = int((time.time() - t0) * 1000)
        output = ProtenixV2Output(
            predicted_cif=str(best.cif_path),
            predicted_pdb=best.pdb_path,
            pae_path=best.pae_path,
            metrics=ProtenixV2Metrics(
                plddt=float(best.plddt if best.plddt is not None else 0.0),
                iptm=float(best.iptm if best.iptm is not None else 0.0),
                ptm=float(best.ptm if best.ptm is not None else 0.0),
                ranking_score=float(
                    best.ranking_score if best.ranking_score is not None else 0.0
                ),
                summary_json=str(best.summary_json),
            ),
            all_predictions=all_records,
            warnings=warnings,
            runtime_ms=runtime_ms,
            seeds_used=list(params.seeds),
        )
        output.candidates = _protenix_v2_candidates(params, all_records)
        return output

    @staticmethod
    def _collect_outputs(
        dump_dir: Path, seeds: list[int]
    ) -> tuple[list[ProtenixV2PredictionRecord], ProtenixV2PredictionRecord | None, list[str]]:
        """Walk Protenix's DataDumper layout and return (all, best_by_ranking)."""
        records: list[ProtenixV2PredictionRecord] = []
        warnings: list[str] = []
        # DataDumper layout: <base>/<dataset_name>/<sample_name>/seed_<seed>/
        #     predictions/<sample_name>_sample_<rank>.cif
        #     predictions/<sample_name>_summary_confidence_sample_<rank>.json
        # ``dataset_name`` is "" (called via infer_predict), so the first level
        # under dump_dir is the sample directory.
        for summary_path in dump_dir.rglob("*_summary_confidence_sample_*.json"):
            parent = summary_path.parent  # .../seed_<seed>/predictions
            seed_dir = parent.parent      # .../seed_<seed>
            stem = summary_path.stem
            # "<sample_name>_summary_confidence_sample_<rank>"
            try:
                rank = int(stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            prediction_sample_name = stem.removesuffix(
                f"_summary_confidence_sample_{rank}"
            )
            cif_path = parent / f"{prediction_sample_name}_sample_{rank}.cif"
            if not cif_path.is_file() or cif_path.stat().st_size == 0:
                continue
            try:
                seed_str = seed_dir.name
                seed_val = int(seed_str.removeprefix("seed_"))
            except ValueError:
                seed_val = seeds[0] if seeds else -1

            summary = _read_summary(summary_path)
            confidence_sample_name = _confidence_sample_name(
                prediction_sample_name, seed_val
            )
            full_confidence = (
                parent / f"{confidence_sample_name}_full_data_sample_{rank}.json"
            )
            pae_path = _write_ipsae_pae_json(full_confidence, summary_path)
            pdb_path = _convert_mmcif_to_pdb(cif_path, warnings)
            rec = ProtenixV2PredictionRecord(
                candidate_id=(
                    f"protenix_v2:{prediction_sample_name}:seed_{seed_val}:rank_{rank}"
                ),
                seed=seed_val,
                rank=rank,
                cif_path=str(cif_path),
                pdb_path=str(pdb_path) if pdb_path is not None else None,
                summary_json=str(summary_path),
                pae_path=str(pae_path) if pae_path is not None else None,
                full_confidence_json=(
                    str(full_confidence) if full_confidence.is_file() else None
                ),
                ranking_score=_coerce_float(summary.get("ranking_score")),
                plddt=_coerce_float(summary.get("plddt")),
                iptm=_coerce_float(summary.get("iptm")),
                ptm=_coerce_float(summary.get("ptm")),
            )
            records.append(rec)

        if not records:
            return records, None, warnings
        # Pick the highest ranking_score (Protenix default rank metric); fall
        # back to (rank=0, lowest seed) ordering when scores are absent.
        records.sort(
            key=lambda r: (
                -(r.ranking_score if r.ranking_score is not None else -1e9),
                r.seed,
                r.rank,
            )
        )
        return records, records[0], warnings

    def estimate_vram(self, params: ProtenixV2Input) -> int:
        # Conservative ceiling; tighten after measuring on a fixture run.
        return 16 * 1024

    def healthcheck(self) -> bool:
        return self.ckpt_path.is_file() and PROTENIX_TREE.is_dir()

    def unload(self) -> None:
        if hasattr(self, "runner"):
            try:
                del self.runner.model
            except Exception:
                pass
            del self.runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "PROTENIX_CKPT_DIR",
    "PROTENIX_TREE",
    "ProtenixV2Adapter",
    "ProtenixV2Input",
    "ProtenixV2Metrics",
    "ProtenixV2Output",
    "ProtenixV2PredictionRecord",
]


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _write_protenix_input_json_from_candidates(
    candidates: list[dict[str, Any]],
    seeds: list[int],
    path: Path,
) -> Path:
    pxdesign_payload = _pxdesign_complex_payload(candidates, seeds)
    if pxdesign_payload is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pxdesign_payload, indent=2) + "\n")
        return path

    sequences: list[dict[str, Any]] = []
    for candidate in candidates:
        sequence = candidate.get("sequence")
        if not sequence:
            raise ValueError("protenix_v2 canonical Candidate[] input requires sequence")
        sequences.append({"proteinChain": {"sequence": str(sequence), "count": 1}})
    if not sequences:
        raise ValueError("protenix_v2 canonical Candidate[] input requires at least one candidate")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": "canonical_candidates",
            "sequences": sequences,
            "modelSeeds": list(seeds),
        }
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _pxdesign_complex_payload(
    candidates: list[dict[str, Any]],
    seeds: list[int],
) -> list[dict[str, Any]] | None:
    contexts = [_pxdesign_target_context(candidate) for candidate in candidates]
    if not contexts or not any(context is not None for context in contexts):
        return None
    if not all(context is not None for context in contexts):
        raise ValueError(
            "protenix_v2 canonical Candidate[] input cannot mix PXDesign "
            "target-context candidates with candidates that lack PXDesign target context"
        )

    payload: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        sequence = candidate.get("sequence")
        if not sequence:
            raise ValueError("protenix_v2 canonical Candidate[] input requires sequence")
        context = contexts[index] or {}
        target_sequence = _target_chain_sequence(
            Path(str(context["target_pdb"])),
            str(context["target_chain"]),
        )
        payload.append(
            {
                "name": _candidate_sample_name(index),
                "sequences": [
                    {"proteinChain": {"sequence": target_sequence, "count": 1}},
                    {"proteinChain": {"sequence": str(sequence), "count": 1}},
                ],
                "modelSeeds": list(seeds),
            }
        )
    return payload


def _pxdesign_target_context(candidate: dict[str, Any]) -> dict[str, str] | None:
    artifacts = candidate.get("artifacts")
    pxdesign = artifacts.get("pxdesign") if isinstance(artifacts, dict) else None
    if not isinstance(pxdesign, dict):
        return None
    target_pdb = pxdesign.get("target_pdb")
    target_chain = pxdesign.get("target_chain")
    if not target_pdb or not target_chain:
        return None
    return {"target_pdb": str(target_pdb), "target_chain": str(target_chain)}


def _target_chain_sequence(target_pdb: Path, target_chain: str) -> str:
    refs = parse_structure_residue_refs(target_pdb.expanduser(), target_chain)
    sequence = "".join(ref.aa for ref in refs if ref.aa and ref.aa != "X")
    if not sequence:
        raise ValueError(
            f"protenix_v2 could not extract target chain {target_chain!r} "
            f"sequence from {target_pdb}"
        )
    return sequence


def _candidate_sample_name(index: int) -> str:
    return f"canonical_candidate_{index:04d}"


def _protenix_v2_candidates(
    params: ProtenixV2Input,
    records: list[ProtenixV2PredictionRecord],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        source_candidate = _source_candidate_for_record(params, record)
        sequence = (
            str(source_candidate.get("sequence"))
            if isinstance(source_candidate, dict) and source_candidate.get("sequence")
            else _source_sequence(params)
        )
        local_id = f"protenix_{index:04d}"
        scores = {
            "seed": record.seed,
            "rank": record.rank,
            "ranking_score": record.ranking_score,
            "plddt": record.plddt,
            "iptm": record.iptm,
            "ptm": record.ptm,
        }
        artifacts = {
            "protenix_v2": {
                "summary_json": record.summary_json,
                "pae_path": record.pae_path,
                "full_confidence_json": record.full_confidence_json,
                "predicted_cif": record.cif_path,
                "predicted_pdb": record.pdb_path,
            }
        }
        if (
            isinstance(source_candidate, dict)
            and _pxdesign_target_context(source_candidate) is not None
        ):
            artifacts["protenix_v2"]["complex_input_mode"] = (
                "pxdesign_target_sequence_binder_sequence"
            )
        candidate: dict[str, Any] = {
            "schema_version": "candidate_v1",
            "candidate_id": stable_candidate_id(
                "protenix_v2",
                local_id,
                native_id=record.candidate_id,
                sequence=sequence,
                structure_path=record.cif_path,
                index=index,
            ),
            "source": {
                "tool": "protenix_v2",
                "node_id": None,
                "job_id": None,
                "local_id": local_id,
                "native_id": record.candidate_id,
                "index": index,
            },
            "structure_path": record.cif_path,
            "scores": {"protenix_v2": scores},
            "artifacts": artifacts,
            "capabilities": candidate_capability_flags(
                has_sequence=bool(sequence),
                has_backbone_structure=bool(record.cif_path),
                has_full_structure=bool(record.cif_path),
                has_candidate_scores=True,
            ),
        }
        if isinstance(source_candidate, dict):
            source_scores = source_candidate.get("scores")
            if isinstance(source_scores, dict):
                candidate["scores"] = {**source_scores, **candidate["scores"]}
            source_artifacts = source_candidate.get("artifacts")
            if isinstance(source_artifacts, dict):
                candidate["artifacts"] = {**source_artifacts, **candidate["artifacts"]}
        if sequence:
            candidate["sequence"] = sequence
        candidates.append(candidate)
    return candidates


def _source_candidate_for_record(
    params: ProtenixV2Input,
    record: ProtenixV2PredictionRecord,
) -> dict[str, Any] | None:
    if not params.candidates:
        return None
    sample_name = _record_sample_name(record)
    if sample_name is None:
        return None
    prefix = "canonical_candidate_"
    if not sample_name.startswith(prefix):
        return None
    try:
        index = int(sample_name.removeprefix(prefix))
    except ValueError:
        return None
    if 0 <= index < len(params.candidates):
        candidate = params.candidates[index]
        return candidate if _is_canonical_candidate(candidate) else None
    return None


def _record_sample_name(record: ProtenixV2PredictionRecord) -> str | None:
    native = record.candidate_id
    if not isinstance(native, str):
        return None
    parts = native.split(":")
    if len(parts) >= 3 and parts[0] == "protenix_v2":
        return parts[1]
    return None


def _source_sequence(params: ProtenixV2Input) -> str | None:
    for candidate in params.candidates:
        if _is_canonical_candidate(candidate) and candidate.get("sequence"):
            return str(candidate["sequence"])
    if not params.input_json:
        return None
    try:
        payload = json.loads(Path(params.input_json).expanduser().read_text())
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, dict):
            continue
        for sequence_item in item.get("sequences") or []:
            if not isinstance(sequence_item, dict):
                continue
            chain = sequence_item.get("proteinChain")
            if isinstance(chain, dict) and chain.get("sequence"):
                return str(chain["sequence"])
    return None
