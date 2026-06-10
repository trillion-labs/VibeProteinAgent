---
name: vpa-protein-design-orchestration
description: Guide external-agent protein design using objective thresholds, VPA tool families, ranking, submission, and reports without VPA rounds. Use when the user asks to design, validate, rank, optimize, submit, or report protein candidates. Not for environment install (vpa-install-and-readiness) or inspecting finished jobs (vpa-state-reporting).
---

# VPA Protein Design Orchestration

Use this when the user asks VPA to design, validate, rank, optimize, submit, or
report protein candidates.

## Objective First

Re-run this step for every request. Whether it is a fresh-session start or
another query later in the same session, define the objective and acceptance
criteria anew — do not reuse or assume the previous query's. Each query gets its
own `traj_` trajectory started with its own `--objective`.

Infer or ask for:

- target and design family;
- constraints such as length, chain, motif, ligand, organism, family, or assay;
- measurable acceptance thresholds, for example confidence, interface score,
  affinity proxy, sequence QC, novelty, or exact functional evidence;
- stopping condition: accepted candidate or no acceptable candidate.

Keep criteria explicit and conservative when the query is underspecified.
When starting the trajectory, pass the objective explicitly:
`vpa trajectory start --workspace <workspace> --user-query "<original request>" --objective "<objective>"`.
Record measurable acceptance criteria in workflow params, run params, or the
final `submit_candidate` reasoning before rendering the report; do not rely on
the report to reconstruct them from generic prose.

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
  `objective_rank`, `submit_candidate`.

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
- when validation tools enrich a logical design with structure or scoring
  evidence, keep submitting and ranking the enriched canonical candidate stream.
  Submitting an earlier sequence-only candidate after downstream evidence exists
  can make the final report dossier appear to lack structure or score evidence.

## Posture: explore to maximize

Lean toward exploration. For the hard steps, try more than one tool or approach
(e.g. several generators or scorers), compare them with `objective_rank` using
explicit gates and a score formula, filter, and
iterate — regenerate, re-score, adjust constraints — to push the objective
higher. Stop when the evidence plateaus or thresholds are clearly met. VPA is
stateless per job, so you own the loop; run as many jobs as the objective
warrants. Independent attempts can run in parallel — submit alternatives
together and wait on them as a batch rather than serializing.

For expensive downstream validation, you may evaluate only a top subset. Before
doing so, make the subset rule explicit and, when the tool output supports it,
record a candidate selection/filter event with the input count, selected ids,
skipped ids, selection reason, selection metrics, and downstream tool. The final
report must be able to distinguish generated-only candidates from evaluated,
ranked, and submitted candidates.

Use one `traj_` trajectory for each user query. You may choose direct `vpa run`
calls or an explicit `vpa workflow run` block at any point; the choice is open.
Start that trajectory with `--objective`, and keep the same objective wording in
downstream params when tools expose `objective`, `design_objective`,
`acceptance_criteria`, `success_criteria`, or `thresholds`. Do not treat workflow
as required for protein design. Use it only when you decide a fixed sub-DAG is
useful, then continue the adaptive loop from its results.

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
- [ ] one trajectory tracks this user query and was started with `--objective`
- [ ] candidate tools/workflows tried, with reason
- [ ] any top-k/subset validation rule recorded
- [ ] evidence jobs run -> objective_rank with explicit gates/formula run by
      default before submit/report (even for a single candidate) so the report
      renders ranking + score from tool output
- [ ] verified against criteria (qc / structure_analysis / rank) -> if short, fix and re-run
- [ ] explored alternatives / iterated while it improves the objective
- [ ] submit_candidate (accepted | no_acceptable_candidate), preferably
      referencing the enriched/ranked candidate that carries final evidence
- [ ] accepted submit_candidate includes selection_summary for the report
- [ ] acceptance criteria recorded before final report rendering
- [ ] one HTML trajectory report rendered
```

If `objective_rank` returns fewer ranked rows than the distinct designs you
validated, treat that as a candidate-identity contract failure rather than a
normal tie. Inspect the per-tool outputs, proceed only with an explicit caveat
when the evidence is otherwise decisive, and prefer fixing candidate wiring or
rerunning with canonical candidate streams before final report rendering.

If no candidate passes, call `submit_candidate` with
`outcome=no_acceptable_candidate` and explain the failed thresholds.
If you stated an acceptance threshold and the measured tool evidence does not
meet it, do not submit `accepted` unless the user explicitly relaxed the
criteria or asked for design hypotheses despite the failed threshold.

## Example (one shape, not the only one)

Goal: a binder for protein target X.

-> define objective + measurable thresholds (e.g. pLDDT >= 80, ipSAE >= 0.7)
-> pxdesign (generate binders vs X; emits canonical candidates)
-> protenix_v2 (folds pxdesign candidates -> pLDDT + PAE)
-> ipSAE (evidence from PAE)
-> objective_rank (pass job_ids plus objective_name, gates, score_formula, and tie_breakers)
-> analyze the room for improvement + try other tools, or mask some
   structure/sequence and regenerate
...
-> submit_candidate (prefer rank=1 from the objective_rank job; accepted if a
   design clears thresholds, else no_acceptable_candidate)
-> run the submit_candidate next_report_command
