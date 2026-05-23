from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.core.binary import find_binary

# Helix codes per DSSP: H = alpha, G = 3-10, I = pi.
_HELIX = set("HGI")
# Strand codes: E = beta-bridge participant, B = isolated beta-bridge.
_SHEET = set("EB")
# Everything else (T, S, P, ' ') counts as coil for the fraction.

_BINARY_CANDIDATES = ("mkdssp", "dssp")


def _find_binary() -> str:
    return str(find_binary(_BINARY_CANDIDATES, env_override="VPA_DSSP_BINARY"))


def _parse_dssp_ss(text: str) -> str:
    """Extract the per-residue SS character string from a classic DSSP output.

    The body of a DSSP file begins after the `#  RESIDUE AA STRUCTURE…` header
    line. The SS code is at column 16 of every residue row; chain breaks
    (resname == "!") are skipped.
    """
    lines = text.splitlines()
    body_start: int | None = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") and "RESIDUE" in line and "STRUCTURE" in line:
            body_start = i + 1
            break
    if body_start is None:
        return ""
    chars: list[str] = []
    for line in lines[body_start:]:
        if len(line) < 17:
            continue
        aa = line[13:14]
        if aa == "!":
            continue
        ss = line[16:17]
        if ss == " ":
            ss = "-"
        chars.append(ss)
    return "".join(chars)


def _run_dssp_binary(binary: str, pdb_path: Path, output_path: Path, output_format: str) -> None:
    """Run DSSP across both modern and legacy mkdssp CLIs.

    Newer mkdssp accepts ``--output-format {dssp,mmcif} input output``. Older
    DSSP 2.x builds only accept ``-i input -o output`` and always emit classic
    DSSP text, which is still enough for the adapter's json/dssp modes.
    """
    modern_cmd = [
        binary,
        "--output-format",
        output_format,
        str(pdb_path),
        str(output_path),
    ]
    modern = subprocess.run(
        modern_cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if modern.returncode == 0:
        return

    stderr = modern.stderr.lower()
    stdout = modern.stdout.lower()
    unsupported_format_flag = "unknown option" in stderr or "unrecognised option" in stderr
    unsupported_format_flag = unsupported_format_flag or "unknown option" in stdout
    if output_format != "dssp" or not unsupported_format_flag:
        modern.check_returncode()

    legacy_cmd = [binary, "-i", str(pdb_path), "-o", str(output_path)]
    legacy = subprocess.run(
        legacy_cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    legacy.check_returncode()


class DSSPInput(BaseModel):
    pdb_path: str = Field(..., min_length=1)
    format: str = Field("json", pattern=r"^(dssp|mmcif|json)$")


class DSSPOutput(BaseModel):
    dssp_output_path: str
    secondary_structure_string: str
    helix_fraction: float
    sheet_fraction: float
    coil_fraction: float
    num_residues: int
    runtime_ms: int


@register_adapter("descriptor.yaml")
class DSSPAdapter(ToolAdapter):
    """Subprocess wrapper around the `mkdssp` binary.

    First binary-backed adapter in the project — establishes the shape Phase
    13's MMseqs2 / Foldseek / TM-align / Vina adapters will follow:
    descriptor declares ``runtime: python`` + ``backend: binary``, ``__init__``
    locates the binary on ``$PATH``, ``run()`` invokes via ``subprocess.run``.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.binary_path = _find_binary()

    def run(self, params: DSSPInput, ctx: AdapterRunContext) -> DSSPOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        pdb_path = Path(params.pdb_path).expanduser().resolve()
        if not pdb_path.exists():
            raise FileNotFoundError(f"DSSP: input PDB not found: {pdb_path}")

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        # mkdssp's --output-format flag accepts {dssp, mmcif}; we emit JSON
        # ourselves from the parsed dssp output.
        if params.format == "json":
            internal_format = "dssp"
            classic_path = out_dir / f"{pdb_path.stem}.dssp"
            final_path = out_dir / f"{pdb_path.stem}.dssp.json"
        elif params.format == "dssp":
            internal_format = "dssp"
            classic_path = out_dir / f"{pdb_path.stem}.dssp"
            final_path = classic_path
        else:  # mmcif
            internal_format = "mmcif"
            classic_path = out_dir / f"{pdb_path.stem}.dssp.cif"
            final_path = classic_path

        # Always also produce the classic .dssp text so we have an SS string.
        # When format=mmcif, run twice (cheap) to get the SS for parsing.
        text_for_ss: str = ""
        ctx.emit_progress(0.25)
        if internal_format == "dssp":
            _run_dssp_binary(self.binary_path, pdb_path, classic_path, "dssp")
            text_for_ss = classic_path.read_text()
            if params.format == "json":
                ss = _parse_dssp_ss(text_for_ss)
                final_path.write_text(json.dumps({
                    "secondary_structure": ss,
                    "source": str(classic_path),
                    "format": "json",
                }, indent=2))
        else:  # mmcif
            _run_dssp_binary(self.binary_path, pdb_path, classic_path, "mmcif")
            # Separate dssp text purely for SS-string extraction.
            classic_aux = out_dir / f"{pdb_path.stem}.dssp"
            _run_dssp_binary(self.binary_path, pdb_path, classic_aux, "dssp")
            text_for_ss = classic_aux.read_text()

        ctx.emit_progress(0.85)
        ss_string = _parse_dssp_ss(text_for_ss)
        n = len(ss_string)
        helix_n = sum(1 for c in ss_string if c in _HELIX)
        sheet_n = sum(1 for c in ss_string if c in _SHEET)
        coil_n = n - helix_n - sheet_n
        helix_f = (helix_n / n) if n else 0.0
        sheet_f = (sheet_n / n) if n else 0.0
        coil_f = (coil_n / n) if n else 0.0

        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return DSSPOutput(
            dssp_output_path=str(final_path),
            secondary_structure_string=ss_string,
            helix_fraction=float(helix_f),
            sheet_fraction=float(sheet_f),
            coil_fraction=float(coil_f),
            num_residues=int(n),
            runtime_ms=runtime_ms,
        )

    def estimate_vram(self, params: DSSPInput) -> int:
        return 0

    def healthcheck(self) -> bool:
        try:
            _find_binary()
            return True
        except FileNotFoundError:
            return False


__all__ = ["DSSPAdapter", "DSSPInput", "DSSPOutput"]
