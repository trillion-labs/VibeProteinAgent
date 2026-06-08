"""Boltz-2 + affinity adapter.

Pointer-style adapter — the Boltz Python source lives on disk at
``$VPA_BOLTZ_TREE`` (default ``$VPA_HOME/sidecars/boltz``) and the
model checkpoints + CCD molecule cache live at ``$VPA_BOLTZ_CACHE`` (default
``$VPA_HOME/cache/boltz``). See ``vendor/boltz2-affinity/
UPSTREAM.md`` for the full provenance.

Why pointer-only: Boltz ships ~4.3 GiB of checkpoints + ~1.8 GiB of CCD
molecules — far too large to vendor under ``vendor/``. The adapter validates
the expected files at construction time and refuses to run if any is missing
(no network downloads from the worker path).

Calls Boltz's own ``boltz.main.predict`` callback directly. We deliberately
do NOT pre-load the model into VRAM at warm time — Boltz's predict() builds
its trainer + module fresh each call (predict_args are baked in at
``Boltz2.load_from_checkpoint`` time), and refactoring that would require
forking the upstream flow. Warm-load amortises Python imports + numba JIT
caches + page-cached weight files; the per-call ``load_from_checkpoint``
cost still dominates. ``pin_warm: true`` is preserved for consistency with
other heavyweight structure adapters and to keep the worker process alive
for hot OS page cache.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.msa_policy import (
    MSABackend,
    should_use_remote_msa,
    strip_prepared_msa,
    yaml_file_has_prepared_msa,
)
from vibeproteinagent.candidates.chain_roles import chain_role_entry
from vibeproteinagent.candidates.schema import stable_candidate_id
from vibeproteinagent.core.compat import ensure_rdkit_allchem_descriptors
from vibeproteinagent.core.config import vpa_home
from vibeproteinagent.core.torch_compat import allow_full_torch_load

# --------------------------------------------------------------------------- paths

BOLTZ_TREE_DEFAULT = str(vpa_home() / "sidecars" / "boltz")
BOLTZ_CACHE_DEFAULT = str(vpa_home() / "cache" / "boltz")

BOLTZ_TREE = Path(os.environ.get("VPA_BOLTZ_TREE", BOLTZ_TREE_DEFAULT)).expanduser()
BOLTZ_CACHE = Path(os.environ.get("VPA_BOLTZ_CACHE", BOLTZ_CACHE_DEFAULT)).expanduser()

# Required artifacts under BOLTZ_CACHE for Boltz-2 + affinity.
_REQUIRED_CACHE_FILES = ("boltz2_conf.ckpt", "boltz2_aff.ckpt")
_REQUIRED_CACHE_DIRS = ("mols",)


def _ensure_boltz_importable() -> None:
    """Place Boltz on sys.path and seed BOLTZ_CACHE.

    Boltz's ``get_cache_path()`` reads ``$BOLTZ_CACHE`` at first use, and
    ``download_boltz2(cache)`` short-circuits when the expected files are
    already present in ``cache``. We set the env var before any boltz import
    so internal call paths (e.g. ``boltz.main`` module-level constants) see
    the right cache.
    """
    if not BOLTZ_TREE.is_dir():
        raise FileNotFoundError(
            f"Boltz source tree not found at {BOLTZ_TREE}. "
            f"Set $VPA_BOLTZ_TREE to a Boltz checkout (must contain src/boltz/)."
        )
    boltz_src = BOLTZ_TREE / "src"
    if not (boltz_src / "boltz" / "main.py").is_file():
        raise FileNotFoundError(
            f"{boltz_src / 'boltz' / 'main.py'} not found — VPA_BOLTZ_TREE "
            f"does not look like a Boltz source checkout."
        )
    if not BOLTZ_CACHE.is_dir():
        raise FileNotFoundError(
            f"Boltz cache directory not found at {BOLTZ_CACHE}. "
            f"Set $VPA_BOLTZ_CACHE to a directory holding "
            f"{', '.join(_REQUIRED_CACHE_FILES)} and the extracted mols/."
        )
    for name in _REQUIRED_CACHE_FILES:
        p = BOLTZ_CACHE / name
        if not p.is_file():
            raise FileNotFoundError(
                f"Required Boltz checkpoint not found at {p}. "
                f"Place the upstream {name} (from huggingface.co/boltz-community/boltz-2) "
                f"under $VPA_BOLTZ_CACHE."
            )
    for name in _REQUIRED_CACHE_DIRS:
        d = BOLTZ_CACHE / name
        if not d.is_dir():
            raise FileNotFoundError(
                f"Required Boltz CCD cache directory not found at {d}. "
                f"Untar mols.tar from the Boltz release under $VPA_BOLTZ_CACHE."
            )
    os.environ["BOLTZ_CACHE"] = str(BOLTZ_CACHE)
    src = str(boltz_src)
    if src not in sys.path:
        sys.path.insert(0, src)
    _ensure_rdkit_descriptors_compat()


def _ensure_rdkit_descriptors_compat() -> None:
    """Expose RDKit Descriptors where the pinned Boltz checkout expects them.

    Some RDKit builds expose molecular descriptors as ``rdkit.Chem.Descriptors``
    but not as ``rdkit.Chem.AllChem.Descriptors``. The current Boltz parser
    dereferences the latter while computing ligand molecular weight for affinity
    records. Bridge that public module alias once during adapter setup.
    """
    ensure_rdkit_allchem_descriptors()


# --------------------------------------------------------------------------- I/O models


class Boltz2AffinityInput(BaseModel):
    input_yaml: str | None = Field(
        default=None,
        description=(
            "Absolute path to a Boltz-format input YAML "
            "(protein + ligand + affinity block)."
        ),
    )
    sequences: list[dict[str, Any]] | None = Field(
        default=None,
        description="Inline list of typed inputs (mutually exclusive with input_yaml).",
    )
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    binder_chain: str | None = Field(
        default=None,
        description="Chain id of the binder ligand to score (only used with `sequences`).",
    )
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1)
    recycling_steps: int = Field(default=3, ge=1, le=20)
    sampling_steps: int = Field(default=200, ge=5, le=1000)
    diffusion_samples: int = Field(default=1, ge=1, le=25)
    sampling_steps_affinity: int = Field(default=200, ge=5, le=1000)
    diffusion_samples_affinity: int = Field(default=3, ge=1, le=25)
    use_msa: bool = Field(default=False)
    msa_backend: MSABackend = Field(default="auto")
    affinity_mw_correction: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_sequences(cls, data: Any) -> Any:
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
        out = dict(data)
        if out.get("input_yaml") is not None or out.get("sequences") is not None:
            return out
        sequences = []
        for index, candidate in enumerate(candidates):
            sequence = candidate.get("sequence")
            if not sequence:
                raise ValueError(
                    "boltz2_affinity canonical Candidate[] input requires sequence "
                    "when input_yaml/sequences are not provided"
                )
            chain_id = chr(ord("A") + index)
            sequences.append({"type": "protein", "id": chain_id, "sequence": sequence})
        out["sequences"] = sequences
        out.setdefault("binder_chain", sequences[-1]["id"])
        return out

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Boltz2AffinityInput:
        if (self.input_yaml is None) == (self.sequences is None):
            raise ValueError("provide exactly one of input_yaml or sequences")
        if self.sequences is not None and self.binder_chain is None:
            raise ValueError("binder_chain is required when sequences is provided")
        return self


class Boltz2AffinityOutput(BaseModel):
    predicted_cif: str
    affinity_pred_value: float
    affinity_probability_binary: float
    confidence_score: float
    runtime_ms: int
    seeds_used: list[int]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    pae_path: str | None = None
    msa_paths: list[str] = Field(default_factory=list)
    msa_index_csv: str | None = None
    plddt_per_residue: list[float] | None = None


# --------------------------------------------------------------------------- adapter


@register_adapter("descriptor.yaml")
class Boltz2AffinityAdapter(ToolAdapter):
    """Adapter for Boltz-2 structure prediction + binding affinity scoring.

    Drives ``boltz.main.predict`` (the same function the upstream
    ``boltz predict`` CLI invokes). Per call, predict() processes the input
    YAML, loads ``boltz2_conf.ckpt`` for structure prediction, then loads
    ``boltz2_aff.ckpt`` for affinity scoring on the top-ranked complex.

    Cold-load: roughly equal to per-call cost (predict() rebuilds modules
    fresh every invocation; the upstream flow couples ``predict_args`` to
    ``load_from_checkpoint``). Future Phase 16+ work can refactor this into
    a pre-loaded module pair with mutable predict_args dict; for now we
    accept the per-call rebuild and rely on the OS page cache for warmth.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        del config, weights_dir
        self.device = device
        _ensure_boltz_importable()
        # Import the boltz CLI entry point and unwrap the click Command to get
        # the underlying python callable.
        boltz_main = importlib.import_module("boltz.main")
        predict_cmd = boltz_main.predict
        self._predict = getattr(predict_cmd, "callback", predict_cmd)
        # Sanity check that we have a callable, not a Click command, after unwrap.
        if not callable(self._predict):
            raise RuntimeError(
                f"boltz.main.predict.callback is not callable: {self._predict!r}"
            )

    @torch.no_grad()
    def run(
        self, params: Boltz2AffinityInput, ctx: AdapterRunContext
    ) -> Boltz2AffinityOutput:
        t0 = time.time()
        ctx.emit_progress(0.02)

        job_inputs = ctx.job_dir / "inputs"
        job_outputs = ctx.job_dir / "outputs"
        job_inputs.mkdir(parents=True, exist_ok=True)
        job_outputs.mkdir(parents=True, exist_ok=True)

        staged_yaml = _stage_input(params, job_inputs)
        record_stem = staged_yaml.stem
        remote_msa = should_use_remote_msa(
            use_msa=params.use_msa,
            msa_backend=params.msa_backend,
            has_prepared_msa=yaml_file_has_prepared_msa(staged_yaml),
        )
        ctx.emit_progress(0.05)

        seeds = list(params.seeds)
        # Boltz's predict() takes one seed at a time. Run for the first seed
        # (the brief's default tiny fixture uses [42]); multi-seed support can
        # be layered later by iterating and picking the best by confidence.
        seed = seeds[0]

        # Boltz needs an accelerator on B200; refuse CPU fallback explicitly.
        if not torch.cuda.is_available():
            raise RuntimeError("Boltz-2 affinity adapter requires CUDA")

        # Make sure boltz's own download/cache short-circuits to our local cache.
        os.environ["BOLTZ_CACHE"] = str(BOLTZ_CACHE)
        # Boltz's predict() touches torch.set_grad_enabled and matmul precision;
        # both are global. We're inside @torch.no_grad so the first is a no-op
        # on us; the matmul precision change is acceptable.
        #
        # The Boltz-2 .ckpt files were saved by Lightning with full pickled
        # state (including ``omegaconf.dictconfig.DictConfig`` hyperparameter
        # blobs). PyTorch 2.6 flipped the default of ``torch.load`` to
        # ``weights_only=True``, which rejects those globals. We explicitly
        # trust our local cache, so disable the weights-only safety for the
        # duration of this call.
        with allow_full_torch_load():
            self._predict(
                data=str(staged_yaml),
                out_dir=str(job_outputs),
                cache=str(BOLTZ_CACHE),
                checkpoint=str(BOLTZ_CACHE / "boltz2_conf.ckpt"),
                affinity_checkpoint=str(BOLTZ_CACHE / "boltz2_aff.ckpt"),
                devices=1,
                accelerator="gpu",
                recycling_steps=params.recycling_steps,
                sampling_steps=params.sampling_steps,
                diffusion_samples=params.diffusion_samples,
                sampling_steps_affinity=params.sampling_steps_affinity,
                diffusion_samples_affinity=params.diffusion_samples_affinity,
                max_parallel_samples=None,
                step_scale=None,
                write_full_pae=False,
                write_full_pde=False,
                output_format="mmcif",
                num_workers=0,
                override=True,
                seed=seed,
                use_msa_server=remote_msa,
                msa_server_url="https://api.colabfold.com",
                msa_pairing_strategy="greedy",
                msa_server_username=None,
                msa_server_password=None,
                api_key_header=None,
                api_key_value=None,
                use_potentials=False,
                model="boltz2",
                method=None,
                affinity_mw_correction=params.affinity_mw_correction,
                preprocessing_threads=1,
                max_msa_seqs=8192,
                subsample_msa=True,
                num_subsampled_msa=1024,
                no_kernels=True,
                write_embeddings=False,
            )
        ctx.emit_progress(0.95)

        result = _collect_outputs(job_outputs, record_stem)
        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return Boltz2AffinityOutput(
            predicted_cif=str(result["predicted_cif"]),
            affinity_pred_value=float(result["affinity_pred_value"]),
            affinity_probability_binary=float(result["affinity_probability_binary"]),
            confidence_score=float(result["confidence_score"]),
            runtime_ms=runtime_ms,
            seeds_used=[seed],
            candidates=_boltz2_affinity_candidates(
                params,
                str(result["predicted_cif"]),
                {
                    "predicted_affinity": float(result["affinity_pred_value"]),
                    "affinity_pred_value": float(result["affinity_pred_value"]),
                    "affinity_probability_binary": float(
                        result["affinity_probability_binary"]
                    ),
                    "confidence_score": float(result["confidence_score"]),
                },
            ),
            pae_path=None,
            msa_paths=list(result["msa_paths"]),
            msa_index_csv=result["msa_index_csv"],
            plddt_per_residue=None,
        )

    def estimate_vram(self, params: Boltz2AffinityInput) -> int:
        # Conservative — Boltz-2 confidence model is ~600M params + activation
        # memory scales with chain length. Tighten in Phase 20 with measured
        # fixtures.
        del params
        return 16 * 1024

    def healthcheck(self) -> bool:
        if not BOLTZ_TREE.is_dir() or not BOLTZ_CACHE.is_dir():
            return False
        for name in _REQUIRED_CACHE_FILES:
            if not (BOLTZ_CACHE / name).is_file():
                return False
        for name in _REQUIRED_CACHE_DIRS:
            if not (BOLTZ_CACHE / name).is_dir():
                return False
        return True

    def unload(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- helpers


def _stage_input(params: Boltz2AffinityInput, dest_dir: Path) -> Path:
    """Materialise the user-provided input into ``dest_dir/<stem>.yaml``.

    Boltz parses the YAML at runtime; the produced ``record.id`` is the file
    stem, so we standardise on ``input.yaml``. If `input_yaml` is a path, we
    copy it (after sanity-loading); if `sequences` is provided we build the
    Boltz-format dict from it.
    """
    target = dest_dir / "input.yaml"
    if params.input_yaml is not None:
        src = Path(params.input_yaml).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"input_yaml not found: {src}")
        # Sanity-load to give the user a clear early error if YAML is malformed.
        with src.open("r") as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict) or "sequences" not in doc:
            raise ValueError(
                f"input_yaml at {src} does not look like a Boltz input: "
                f"missing top-level 'sequences' key"
            )
        if not params.use_msa or params.msa_backend == "none":
            doc = _force_single_sequence_msa(doc)
            target.write_text(yaml.safe_dump(doc, sort_keys=False))
        elif params.msa_backend == "remote":
            doc = strip_prepared_msa(doc)
            target.write_text(yaml.safe_dump(doc, sort_keys=False))
        else:
            target.write_text(src.read_text())
        return target

    # sequences path — build the Boltz schema dict.
    assert params.sequences is not None
    include_prepared_msa = params.use_msa and params.msa_backend != "remote"
    sequences: list[dict[str, Any]] = []
    for entry in params.sequences:
        # Each entry is { type: "...", id: ..., sequence|smiles|ccd|msa|binder ... }
        kind = entry["type"]
        body: dict[str, Any] = {"id": entry["id"]}
        for key in ("sequence", "smiles", "ccd"):
            if key in entry and entry[key] is not None:
                body[key] = entry[key]
        if kind == "protein" and not params.use_msa:
            body["msa"] = "empty"
        elif include_prepared_msa and entry.get("msa") is not None:
            body["msa"] = entry["msa"]
        sequences.append({kind: body})
    doc: dict[str, Any] = {"version": 1, "sequences": sequences}
    if params.binder_chain is not None:
        doc["properties"] = [{"affinity": {"binder": params.binder_chain}}]
    target.write_text(yaml.safe_dump(doc, sort_keys=False))
    return target


def _force_single_sequence_msa(value: Any) -> Any:
    """Return Boltz YAML data with protein entries marked as single-sequence.

    Boltz uses the literal ``msa: empty`` to disable MSA for a protein chain.
    Removing the field entirely asks Boltz to generate an MSA and then fails
    unless ``--use_msa_server`` is enabled.
    """
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if key == "protein" and isinstance(item, dict):
                protein = {
                    k: _force_single_sequence_msa(v)
                    for k, v in item.items()
                    if k not in {"msa", "msa_path", "msa_file", "paired_msa", "unpaired_msa"}
                }
                protein["msa"] = "empty"
                converted[key] = protein
            else:
                converted[key] = _force_single_sequence_msa(item)
        return converted
    if isinstance(value, list):
        return [_force_single_sequence_msa(item) for item in value]
    return value


def _collect_outputs(job_outputs: Path, record_stem: str) -> dict[str, Any]:
    """Walk Boltz's writer layout and return the canonical adapter outputs.

    Layout per ``boltz.main.predict``::

        <job_outputs>/boltz_results_<stem>/predictions/<record_id>/
            <record_id>_model_0.cif         # top-ranked structure
            confidence_<record_id>_model_0.json
            affinity_<record_id>.json
    """
    results_dir = job_outputs / f"boltz_results_{record_stem}"
    pred_dir = results_dir / "predictions" / record_stem
    if not pred_dir.is_dir():
        raise RuntimeError(
            f"Boltz did not produce a predictions directory at {pred_dir}. "
            f"Check {results_dir} for error logs."
        )
    cif_candidates = sorted(pred_dir.glob(f"{record_stem}_model_*.cif"))
    if not cif_candidates:
        raise RuntimeError(f"Boltz produced no CIF files under {pred_dir}")
    # Boltz ranks by confidence; model_0 is best. If a different scheme is
    # ever used the file naming would change.
    predicted_cif = pred_dir / f"{record_stem}_model_0.cif"
    if not predicted_cif.is_file():
        predicted_cif = cif_candidates[0]

    aff_json = pred_dir / f"affinity_{record_stem}.json"
    if not aff_json.is_file():
        raise RuntimeError(
            f"Boltz affinity output missing at {aff_json}. "
            f"The input YAML must declare 'properties: [{{affinity: {{binder: <ligand_id>}}}}]'."
        )
    with aff_json.open("r") as fh:
        aff_data = json.load(fh)
    for key in ("affinity_pred_value", "affinity_probability_binary"):
        if key not in aff_data:
            raise RuntimeError(f"Boltz affinity output missing key '{key}' in {aff_json}")

    conf_json = pred_dir / f"confidence_{record_stem}_model_0.json"
    confidence_score = 0.0
    if conf_json.is_file():
        with conf_json.open("r") as fh:
            conf_data = json.load(fh)
        cs = conf_data.get("confidence_score")
        if cs is not None:
            confidence_score = float(cs)

    msa_dir = results_dir / "msa"
    msa_paths: list[Path] = []
    if msa_dir.is_dir():
        for suffix in ("*.a3m", "*.sto", "*.fa", "*.fasta", "*.aln"):
            msa_paths.extend(msa_dir.rglob(suffix))
    msa_paths = sorted({p.resolve() for p in msa_paths if p.is_file()})
    msa_index_csv = results_dir / "msa" / f"{record_stem}_0.csv"

    return {
        "predicted_cif": predicted_cif,
        "affinity_pred_value": aff_data["affinity_pred_value"],
        "affinity_probability_binary": aff_data["affinity_probability_binary"],
        "confidence_score": confidence_score,
        "msa_paths": [str(path) for path in msa_paths],
        "msa_index_csv": str(msa_index_csv) if msa_index_csv.is_file() else None,
    }


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _boltz2_affinity_candidates(
    params: Boltz2AffinityInput,
    predicted_cif: str,
    scores: dict[str, Any],
) -> list[dict[str, Any]]:
    source = _select_source_candidate(params)
    sequence = source.get("sequence") if isinstance(source, dict) else None
    if not sequence:
        sequence = _protein_sequence_for_chain(params, params.binder_chain)
    local_id = (
        _source_local_id(source, "boltz2_affinity_0000")
        if isinstance(source, dict)
        else "boltz2_affinity_0000"
    )
    if _is_canonical_candidate(source):
        candidate = {
            key: value
            for key, value in source.items()
            if key
            in {
                "schema_version",
                "candidate_id",
                "source",
                "sequence",
                "structure_path",
                "capabilities",
            }
        }
        candidate["scores"] = dict(source.get("scores") or {})
        candidate["artifacts"] = dict(source.get("artifacts") or {})
        candidate["structure_path"] = predicted_cif
    else:
        candidate = {
            "schema_version": "candidate_v1",
            "candidate_id": stable_candidate_id(
                "boltz2_affinity",
                local_id,
                sequence=sequence,
                structure_path=predicted_cif,
                index=0,
            ),
            "source": {
                "tool": "boltz2_affinity",
                "node_id": None,
                "job_id": None,
                "local_id": local_id,
                "native_id": None,
                "index": 0,
            },
            "structure_path": predicted_cif,
            "scores": {},
            "artifacts": {},
        }
        if sequence:
            candidate["sequence"] = sequence
    candidate.setdefault("scores", {})["boltz2_affinity"] = scores
    artifact = {
        "predicted_cif": predicted_cif,
    }
    complex_rows = _boltz2_complex_chain_rows(params, source, predicted_cif)
    if complex_rows:
        artifact["complex_chains"] = complex_rows
    candidate.setdefault("artifacts", {})["boltz2_affinity"] = artifact
    return [candidate]


def _select_source_candidate(params: Boltz2AffinityInput) -> dict[str, Any]:
    candidates = [item for item in params.candidates if _is_canonical_candidate(item)]
    if not candidates:
        return {}
    if len(candidates) == 1:
        return candidates[0]

    sequence_matches = _candidate_sequence_matches(params, candidates)
    if len(sequence_matches) == 1:
        return sequence_matches[0]
    if len(sequence_matches) > 1:
        raise ValueError(
            "boltz2_affinity could not select upstream candidate: multiple "
            "canonical candidates match the protein sequences from sequences/input_yaml"
        )
    raise ValueError(
        "boltz2_affinity multiple canonical candidates require a unique sequence "
        "match from sequences/input_yaml"
    )


def _candidate_sequence_matches(
    params: Boltz2AffinityInput,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    binder_sequence = _protein_sequence_for_chain(params, params.binder_chain)
    protein_sequences = (
        {binder_sequence}
        if binder_sequence
        else _protein_sequences_from_params(params)
    )
    if not protein_sequences:
        return []
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate.get("sequence"), str)
        and candidate["sequence"] in protein_sequences
    ]


def _protein_sequences_from_params(params: Boltz2AffinityInput) -> set[str]:
    sequences: set[str] = set()
    for entry in params.sequences or []:
        if entry.get("type") != "protein":
            continue
        sequence = entry.get("sequence")
        if isinstance(sequence, str) and sequence:
            sequences.add(sequence)
    if sequences or params.input_yaml is None:
        return sequences

    src = Path(params.input_yaml).expanduser()
    try:
        with src.open("r") as fh:
            doc = yaml.safe_load(fh)
    except OSError:
        return sequences
    if not isinstance(doc, dict):
        return sequences
    for entry in doc.get("sequences") or []:
        if not isinstance(entry, dict):
            continue
        protein = entry.get("protein")
        if not isinstance(protein, dict):
            continue
        sequence = protein.get("sequence")
        if isinstance(sequence, str) and sequence:
            sequences.add(sequence)
    return sequences


def _typed_sequences_from_params(params: Boltz2AffinityInput) -> list[dict[str, Any]]:
    if params.sequences is not None:
        return [
            {
                "type": str(entry.get("type") or ""),
                "id": str(entry.get("id") or ""),
                "sequence": entry.get("sequence"),
                "smiles": entry.get("smiles"),
                "ccd": entry.get("ccd"),
            }
            for entry in params.sequences
            if isinstance(entry, dict)
        ]
    if params.input_yaml is None:
        return []
    try:
        with Path(params.input_yaml).expanduser().open("r") as fh:
            doc = yaml.safe_load(fh)
    except OSError:
        return []
    if not isinstance(doc, dict):
        return []
    out: list[dict[str, Any]] = []
    for entry in doc.get("sequences") or []:
        if not isinstance(entry, dict):
            continue
        for kind, body in entry.items():
            if not isinstance(body, dict):
                continue
            out.append(
                {
                    "type": str(kind),
                    "id": str(body.get("id") or ""),
                    "sequence": body.get("sequence"),
                    "smiles": body.get("smiles"),
                    "ccd": body.get("ccd"),
                }
            )
    return out


def _protein_sequence_for_chain(
    params: Boltz2AffinityInput,
    chain_id: str | None,
) -> str | None:
    if not chain_id:
        return None
    for entry in _typed_sequences_from_params(params):
        if entry.get("type") == "protein" and entry.get("id") == chain_id:
            sequence = entry.get("sequence")
            return str(sequence) if isinstance(sequence, str) and sequence else None
    return None


def _boltz2_complex_chain_rows(
    params: Boltz2AffinityInput,
    source: dict[str, Any],
    predicted_cif: str,
) -> list[dict[str, Any]]:
    entries = _typed_sequences_from_params(params)
    if not entries:
        return []
    source_sequence = source.get("sequence") if isinstance(source, dict) else None
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        kind = entry.get("type")
        chain_id = str(entry.get("id") or "")
        sequence = entry.get("sequence")
        sequence_text = str(sequence) if isinstance(sequence, str) and sequence else None
        if kind == "protein":
            role = "target"
            if chain_id == params.binder_chain and (
                not source_sequence or source_sequence == sequence_text
            ):
                role = "binder"
            elif source_sequence and source_sequence == sequence_text:
                role = "design"
            rows.append(
                chain_role_entry(
                    role=role,
                    sequence=sequence_text,
                    chain_id=chain_id or None,
                    input_order=index,
                    structure_path=predicted_cif,
                )
            )
        elif chain_id == params.binder_chain:
            rows.append(
                {
                    "role": "ligand",
                    "chain_id": chain_id,
                    "input_order": index,
                }
            )
    return rows


def _source_local_id(candidate: dict[str, Any], fallback: str) -> str:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    return str(source.get("local_id") or fallback)


__all__ = [
    "BOLTZ_CACHE",
    "BOLTZ_TREE",
    "Boltz2AffinityAdapter",
    "Boltz2AffinityInput",
    "Boltz2AffinityOutput",
]
