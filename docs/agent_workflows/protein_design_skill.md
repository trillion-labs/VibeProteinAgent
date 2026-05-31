# Protein Design Workflow Skill

This runbook is for agents executing VPA protein workflows. Keep ordinary
scientific conversation flexible; use this document when the user wants an
executable retrieval, design, evaluation, ranking, submission, or report.

## Execution Checklist

1. Restate the protein task as a workflow family: semantic retrieval, binder
   design, de novo design, or mixed optimization.
2. Identify required user assets: target PDB/CIF, sequence, motif, constraints,
   design objective, workspace, provider/model, GPU budget, and max rounds.
3. Prefer a plan-only or deterministic preflight when a live run is expensive or
   likely to fail due to credentials, missing tools, or GPU pressure.
4. Execute with canonical VPA tools and let compiler/runtime own executor
   wiring, candidate streams, and derived artifacts.
5. Inspect workflow/node/job status, final decision, submitted candidates, and
   report artifacts before calling a run successful.

## Semantic Retrieval Trajectory

Use this family when the user asks to find, retrieve, compare, or analyze known
proteins or sequence-like candidates.

Canonical shape:

```text
search_sequence
-> sequence_analysis
-> merge_candidates
-> rank_designs
-> submit_candidate or report
```

Notes:

- Retrieval tools produce candidate records or candidate-like rows.
- `sequence_analysis` contributes sequence features and liability metrics.
- `merge_candidates` is the normalization point before ranking.
- `rank_designs` may omit optional auxiliary streams such as predictions,
  analyses, and QC results when upstream tools did not produce them.

## Binder Design Trajectory

Use this family when the user provides a target structure or binding objective.

Canonical shape:

```text
pxdesign or binder generator
-> protein_qc and/or structure_analysis
-> merge_candidates
-> rank_designs
-> submit_candidate or report
```

Notes:

- Treat target structures and user constraints as root inputs.
- Downstream structure/QC/ranking nodes should consume canonical candidates from
  upstream nodes, not planner-authored artifact paths.
- PXDesign preview fallback must use real chain provenance from the generated
  preview artifact; binder-only preview mode should not invent a conditioning
  chain.

## De Novo Design Trajectory

Use this family when the user asks to generate new proteins without a target
binding interface.

Canonical shape:

```text
rfdiffusion or backbone generation
-> proteinmpnn or sequence generation
-> esm3/protenix_v2 or structure prediction/refinement
-> protein_qc
-> structure_analysis
-> rank_designs
-> submit_candidate or report
```

Notes:

- Keep expensive GPU fan-out explicit through workflow or compiler-supported
  batching, not through hidden prompt assumptions.
- If a downstream predictor can derive its input JSON or structure artifact from
  upstream candidates, the LLM should not provide that artifact as a literal.
- Use ranking/submission only after real outputs exist.

## Prior-Round Candidates

When an optimization round needs prior candidates, describe the relationship as
a candidate-source upstream in planner IR. The executor-facing value
`${outputs.prior_round.candidates}` is materialized by `CandidateArchive` from
the exact survivor snapshot.

Do not use raw strings, compact summaries, report tables, or prompt digests as
candidate pools. If a model emits shorthand such as `prior_round`, the compiler
may normalize known shorthand to canonical candidate-source wiring; unknown
planner literals remain invalid.

## Submission And Reports

Final submission is a post-round decision. Planner IR should not add
`submit_candidate` as a planned workflow node. After execution:

- submit only candidates that came through the archive/materialized workflow;
- resolve submit job aliases to real `job_...` identifiers;
- create a `vpa report` for successful design/retrieval runs;
- collect PASS evidence with workflow id, provider/model, report path, and final
  candidate/answer summary.

