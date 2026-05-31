# PyRosetta - vendor pointer

This directory is a pointer-only vendor entry. VPA does not check PyRosetta
binaries into git.

## Install Contract

PyRosetta is installed by `vpa install` without requiring a VPA license key.
The installer uses the current PyRosetta package channel and treats PyRosetta as
an installable shared-env dependency.

The install contract is intentionally separate from usage terms. The upstream
PyRosetta site states that commercial use requires a separate license. VPA is a
non-commercial repo, but operators should still follow the current upstream
PyRosetta/Rosetta Commons terms for their own use case.

## Upstream

| field | value |
| --- | --- |
| Homepage | https://www.pyrosetta.org/ |
| Downloads | https://www.pyrosetta.org/downloads |
| Runtime package | `pyrosetta` |
| VPA install target | shared `vpa` env |

## Verify

```bash
python -c "import pyrosetta; pyrosetta.init(extra_options='-mute all'); print('ok')"
```

The adapter keeps `pyrosetta.init()` and Rosetta global state isolated through a
persistent local sidecar process, even when PyRosetta is importable from the
shared `vpa` env.

## Env Vars Consumed By The Adapter

| var | default | role |
| --- | --- | --- |
| `VPA_PYROSETTA_ENV` | shared `vpa` env when `pyrosetta` is importable; otherwise legacy env discovery | optional override for a dedicated env root |
| `CONDA_EXE` | first `conda` on PATH | used for legacy named-env discovery only |

## Capabilities Exposed

The adapter wraps four primitives. All share a single `PyRosettaInput` shape;
`mode` selects the dispatch.

| mode | output | notes |
| --- | --- | --- |
| `score` | `total_score` (REF2015 REU) | the default |
| `interface` | `total_score` + `interface_metrics` | requires `chains_a` + `chains_b` |
| `relax` | `total_score` + `relaxed_pdb_path` | FastRelax(`max_relax_cycles`) |
| `pack` | `total_score` + `relaxed_pdb_path` | RestrictToRepacking + PackRotamersMover |

## Promotion To VERIFIED

The adapter is verified when both hold:

1. `python -c "import pyrosetta"` exits 0 in the selected runtime env.
2. `tests/test_adapter_pyrosetta_real.py::test_score_mode_on_fixture` passes
   against `tests/fixtures/6MRR.pdb`.
