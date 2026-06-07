---
name: vpa-protein-design-orchestration
description: Guide external-agent protein design using objective thresholds, VPA tool families, ranking, submission, and reports without VPA rounds. Use when the user asks to design, validate, rank, optimize, submit, or report protein candidates. Not for environment install (vpa-install-and-readiness) or inspecting finished jobs (vpa-state-reporting).
---

# VPA Protein Design Orchestration

Use this when the user asks VPA to design, validate, rank, optimize, submit, or
report protein candidates.

## Objective First

Infer or ask for:

- target and design family;
- constraints such as length, chain, motif, ligand, organism, family, or assay;
- measurable acceptance thresholds, for example confidence, interface score,
  affinity proxy, sequence QC, novelty, or exact functional evidence;
- stopping condition: accepted candidate or no acceptable candidate.

Keep criteria explicit and conservative when the query is underspecified.

## Tool Families

Before committing to a long design run, confirm the tools you'll need are ready
with `vpa tools --json` and `vpa tool info <tool> --json`; read the response
under `data.tools` / `data.tool`. Availability is included in those outputs.
For schema details without live probes, use
`vpa tool info <tool> --schema-only --json`. Gated models (e.g. ESM3's
`HF_TOKEN`) or missing weights otherwise fail mid-run. Repair them up front via
vpa-install-and-readiness. If those checks already show the tools available,
skip installation and go straight to execution — only invoke
vpa-install-and-readiness when something is missing or broken.

Think in families. The current set and exact specs come from
`vpa tools --json` and `vpa tool info <tool> --schema-only --json` — treat those as truth, not
this list (names below are illustrative examples, not exhaustive):

- Retrieval / function grounding (e.g. `search_sequence`, `template_select`,
  `family_profile`).
- Backbone generation (e.g. `rfdiffusion`, `rfpeptides`).
- Inverse folding (e.g. `proteinmpnn`, `ligandmpnn`).
- Sequence generation / masked infilling (e.g. `esm2`).
- Sequence & backbone co-generation / infilling (e.g. `esm3`).
- Sequence engineering (e.g. `sequence_edit`).
- Binder generation (e.g. `pxdesign`, `bindcraft`, `boltzgen`, `proteina_complexa`).
- Evidence — structure prediction, affinity, QC (e.g. `boltz2_affinity`,
  `protein_qc`, `ipsae`, `prodigy`, `tmalign`).
- Candidate-stream control: `merge_candidates` (dedup/union only),
  `rank_designs`, `submit_candidate`.

These are hints, not a fixed script. Choose the optimal tool set that can
maximize the objective.

Before nontrivial params, read the tool schema. Do not infer enum values from
tool names (for example, an MSA tool name is not necessarily a valid backend
value), and check mode-specific required fields such as `conditional_required`.

## Representation State

Respect what each candidate actually contains:

- sequence-only candidates need folding / structure prediction before
  structure or interface scoring;
- backbone-only candidates need inverse folding before sequence QC or sequence
  conditioned folding;
- full-structure candidates can go to structure analysis, interface scoring,
  ranking, or submission;
- Protenix -> ipSAE needs two wires: pass `candidates` for the structure
  (`structure_path` may be mmCIF) and pass `pae_path` separately from the
  Protenix output. Do not assume the candidate stream alone carries the PAE
  input. Concrete `${outputs...}` wiring belongs in vpa-run-and-workspace;
- schema field names such as `pdb_path` can still accept mmCIF when the tool
  schema says so. Check the tool schema instead of converting formats by name;
- use `merge_candidates` only to combine canonical candidate streams. It does
  not create evidence, improve designs, or choose a winner.

## Posture: explore to maximize

Lean toward exploration. For the hard steps, try more than one tool or approach
(e.g. several generators or scorers), compare them with `rank_designs`, filter, and
iterate — regenerate, re-score, adjust constraints — to push the objective
higher. Stop when the evidence plateaus or thresholds are clearly met. VPA is
stateless per job, so you own the loop; run as many jobs as the objective
warrants. Independent attempts can run in parallel — submit alternatives
together and wait on them as a batch rather than serializing.

Use one `traj_` trajectory for each user query. You may choose direct `vpa run`
calls or an explicit `vpa workflow run` block at any point; the choice is open.
Do not treat workflow as required for protein design. Use it only when you decide
a fixed sub-DAG is useful, then continue the adaptive loop from its results.

When a job fails (OOM, CUDA, bad params), don't drop the objective: read
`vpa logs`, fix params and retry or swap to another tool, and fold the failure
back into the loop.

## Grounding

Quote sequences, scores, confidence, and pass/fail only from tool outputs. Do
not invent numbers or acceptance verdicts from model knowledge.

## Execution Shape

Rough pseudo-flow (a guide, not a fixed script):

```text
objective -> criteria -> trajectory -> try tools/workflows -> evidence jobs -> merge/rank
   ^                                                                |
   +--- gap to objective? try more / adjust constraints and loop ---+
-> submit_candidate -> report
```

Optional progress checklist — drop or reorder it if a better path fits the
objective:

```text
- [ ] objective + measurable acceptance criteria stated
- [ ] one trajectory tracks this user query
- [ ] candidate tools/workflows tried, with reason
- [ ] evidence jobs run -> rank_designs when comparison helps
- [ ] verified against criteria (qc / structure_analysis / rank) -> if short, fix and re-run
- [ ] explored alternatives / iterated while it improves the objective
- [ ] submit_candidate (accepted | no_acceptable_candidate)
- [ ] one HTML trajectory report rendered
```

If no candidate passes, call `submit_candidate` with
`outcome=no_acceptable_candidate` and explain the failed thresholds.

## Example (one shape, not the only one)

Goal: a binder for protein target X.

-> define objective + measurable thresholds (e.g. pLDDT >= 80, ipSAE >= 0.7)
-> pxdesign (generate binders vs X; emits canonical candidates)
-> protenix_v2 (folds pxdesign candidates -> pLDDT + PAE)
-> ipSAE (evidence from PAE)
-> rank_designs (pass job_ids for the predictor/scorer/QC jobs when available)
-> analyze the room for improvement + try other tools, or mask some
   structure/sequence and regenerate
...
-> submit_candidate (accepted if a design clears thresholds,
   else no_acceptable_candidate)
-> run the submit_candidate next_report_command
