# PyRosetta — vendor pointer

This directory is a **pointer-only** vendor entry. PyRosetta cannot be
redistributed: it requires an academic licence from Rosetta Commons and
ships from a private pip index. We never check the binary or its
weights into git, and we never install it into the shared ``vpa``
conda env.

## Upstream

| field | value |
| --- | --- |
| Homepage | https://www.pyrosetta.org/ |
| Licence | Rosetta Commons academic licence (non-redistributable) |
| Custom index (conda) | `https://USER:PASS@conda.rosettacommons.org` |
| Custom index (pip)   | `https://USER:PASS@west.rosettacommons.org/pyrosetta/pypi/<channel>` |
| Channel | `release` (latest stable), or `monthly`/`weekly` for newer features |

`USER` and `PASS` are obtained by registering at
https://els2.comotion.uw.edu/product/pyrosetta (academic) — typically a
one-day approval turnaround.

## Why a dedicated env (NOT vpa)

The Phase 12-20 roadmap and the brief both forbid touching ``vpa``'s
pinned scientific stack (torch 2.7.1+cu128, numpy 2.3.5, scipy 1.17.1,
transformers 5.8.1, pandas 3.0.3). PyRosetta channels sometimes ship
with incompatible deps (notably an older numpy on some quarterly
builds), and a shared dev env should not carry a licensed binary.

So we run PyRosetta in a separate conda env. The adapter shells out to
``<env>/bin/python -m vibeproteinagent.adapters.pyrosetta._runner``.

## Bootstrap (operator-side)

```bash
# 1. Obtain a licence at https://els2.comotion.uw.edu/product/pyrosetta
#    and grab the conda credentials from the licence email.
export ROSETTA_USER=<your_username>
export ROSETTA_PASS=<your_token>

# 2. Build a fresh env (Python 3.11 matches our vpa baseline; PyRosetta
#    has wheels for 3.9-3.12).
conda create -y -n pyrosetta-vpa python=3.11
conda activate pyrosetta-vpa

# 3. Install via the conda channel (preferred — wheels are pre-built).
conda install -y -c "https://${ROSETTA_USER}:${ROSETTA_PASS}@conda.rosettacommons.org" pyrosetta

# 3b. OR via the pip index, if you prefer pip:
# pip install pyrosetta-installer && \
#   python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"

# 4. Verify.
python -c "import pyrosetta; pyrosetta.init(extra_options='-mute all'); print('ok')"

# 5. Tell the adapter where the env lives (optional — default discovery
#    looks at /opt/conda/envs/pyrosetta-vpa and `conda env list`).
export VPA_PYROSETTA_ENV=$(conda info --base)/envs/pyrosetta-vpa
```

Once the env import test passes, every PyRosetta adapter call (and the
``vpa daemon warm pyrosetta`` healthcheck) flips to True automatically.

## Env vars consumed by the adapter

| var | default | role |
| --- | --- | --- |
| `VPA_PYROSETTA_ENV` | `/opt/conda/envs/pyrosetta-vpa` or `conda env list` lookup | path to the dedicated env root |
| `CONDA_EXE` | first `conda` on PATH | used for env discovery only |

## Capabilities exposed

The adapter wraps four primitives. All share a single ``PyRosettaInput``
shape; ``mode`` selects the dispatch.

| mode | output | notes |
| --- | --- | --- |
| `score` | `total_score` (REF2015 REU) | the default |
| `interface` | `total_score` + `interface_metrics` (dG_separated, dSASA, sc, packstat, …) | requires `chains_a` + `chains_b` |
| `relax` | `total_score` + `relaxed_pdb_path` | FastRelax(`max_relax_cycles`) |
| `pack` | `total_score` + `relaxed_pdb_path` | RestrictToRepacking + PackRotamersMover |

## Promotion to VERIFIED

The adapter ships as PARTIAL until both of these hold:

1. ``<VPA_PYROSETTA_ENV>/bin/python -c "import pyrosetta"`` exits 0
2. The slow test ``tests/test_adapter_pyrosetta_real.py::test_score_mode_on_fixture``
   passes against ``tests/fixtures/6MRR.pdb``

No code changes are required for promotion — only operator-side env
provisioning.
