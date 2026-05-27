"""Artifact fallback for PXDesign preview pipeline.

The upstream preview pipeline can write raw diffusion CIFs with an ``xpb``
placeholder binder chain before AF2/ColabDesign post-processing finishes. On
B200 hosts, the ColabDesign JAX path may be unavailable on GPU, while CPU MPNN
sequence design remains fast enough for the tiny preview contract. This helper
runs inside the dedicated PXDesign pipeline env and materialises the same
``design_outputs/<task>/summary.csv`` shape expected from upstream.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_atom_site_rows(cif_path: Path) -> list[dict[str, str]]:
    cols: list[str] = []
    rows: list[dict[str, str]] = []
    in_atom_loop = False
    for line in cif_path.read_text().splitlines():
        if line.startswith("loop_"):
            cols = []
            in_atom_loop = False
            continue
        if line.startswith("_atom_site."):
            in_atom_loop = True
            cols.append(line.strip().removeprefix("_atom_site."))
            continue
        if not in_atom_loop:
            continue
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            in_atom_loop = False
            continue
        if len(parts) < len(cols):
            continue
        row = {col: parts[idx] for idx, col in enumerate(cols)}
        if row.get("group_PDB") == "ATOM":
            rows.append(row)
    if not rows:
        raise ValueError(f"No _atom_site ATOM rows found in {cif_path}")
    return rows


def _write_mpnn_pdb(cif_path: Path, pdb_path: Path) -> None:
    rows = _read_atom_site_rows(cif_path)
    coords = [
        (float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"]))
        for row in rows
    ]
    mean_x = sum(x for x, _, _ in coords) / len(coords)
    mean_y = sum(y for _, y, _ in coords) / len(coords)
    mean_z = sum(z for _, _, z in coords) / len(coords)
    max_abs = max(
        max(abs(x - mean_x), abs(y - mean_y), abs(z - mean_z))
        for x, y, z in coords
    )
    # PXDesign preview smokes with very few diffusion steps can emit huge
    # coordinates that overflow fixed-width PDB columns. Rescale only the
    # temporary MPNN input; the returned CIF keeps the original coordinates.
    scale = 200.0 / max_abs if max_abs > 200.0 else 1.0
    with pdb_path.open("w") as fh:
        for serial, row in enumerate(rows, 1):
            x = (float(row["Cartn_x"]) - mean_x) * scale
            y = (float(row["Cartn_y"]) - mean_y) * scale
            z = (float(row["Cartn_z"]) - mean_z) * scale
            resname = row["label_comp_id"].upper()
            if resname == "XPB":
                resname = "GLY"
            atom_name = row["label_atom_id"]
            atom_field = f" {atom_name:<3}" if len(atom_name) < 4 else atom_name[:4]
            chain = (row["label_asym_id"] or "A")[0]
            resseq = int(row["label_seq_id"])
            elem = row["type_symbol"].upper()[:2]
            fh.write(
                f"ATOM  {serial:5d} {atom_field} {resname:>3s} {chain:1s}"
                f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {elem:>2s}\n"
            )
        fh.write("TER\nEND\n")


def materialize_preview_outputs(
    *,
    raw_cif: Path,
    task_name: str,
    output_root: Path,
    temperature: float,
    seed: int | None,
) -> Path:
    """Run CPU ProteinMPNN and write final preview design_outputs artifacts."""
    from ml_collections import ConfigDict
    from pxdbench.tools.protmpnn.main_mpnn import design_binder
    from pxdesign.utils.pipeline import replace_last_xpb_chain_sequence

    task_dir = output_root / "design_outputs" / task_name
    orig_dir = task_dir / "orig_designed"
    work_dir = task_dir / "_mpnn_input"
    orig_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    pdb_stem = raw_cif.stem
    pdb_path = work_dir / f"{pdb_stem}.pdb"
    _write_mpnn_pdb(raw_cif, pdb_path)

    mpnn_cfg = ConfigDict(
        {
            "weights": "original",
            "rm_aa": "C",
            "temperature": str(float(temperature)),
            "fix_interface": False,
        }
    )
    results = design_binder(
        str(work_dir),
        [pdb_stem],
        1,
        binder_chains=["B"],
        cond_chains=["A"],
        mpnn_cfg=mpnn_cfg,
        if_print=True,
    )
    if not results:
        raise RuntimeError("ProteinMPNN fallback returned no sequences")
    sequence = str(results[0]["sequence"])
    if not sequence or "X" in sequence:
        raise RuntimeError(f"ProteinMPNN fallback returned invalid sequence {sequence!r}")

    final_cif = orig_dir / "rank_0.cif"
    replace_last_xpb_chain_sequence(str(raw_cif), str(final_cif), sequence)
    summary_csv = task_dir / "summary.csv"
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "task_name",
                "sequence",
                "chosen_struct_path",
                "preview_fallback",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "rank": "0",
                "task_name": task_name,
                "sequence": sequence,
                "chosen_struct_path": "orig_designed/rank_0.cif",
                "preview_fallback": "proteinmpnn_cpu",
                "seed": "" if seed is None else str(seed),
            }
        )
    return final_cif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-cif", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    materialize_preview_outputs(
        raw_cif=Path(args.raw_cif),
        task_name=args.task_name,
        output_root=Path(args.output_root),
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
