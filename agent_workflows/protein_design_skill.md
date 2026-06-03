# Protein Design Workflow Skill

This runbook is for agents executing VPA protein workflows. Keep ordinary
scientific conversation flexible; use this document when the user wants an
executable retrieval, design, evaluation, ranking, submission, or report.
For VPA installation, repair, external binaries, sidecars, and model weights,
read [vpa_installation.md](vpa_installation.md).
For portable workspace commands, execution, tracking, and report commands, pair
this with [vpa_execution_environment.md](vpa_execution_environment.md).

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
6. Keep terminal actions out of the planned workflow DAG. `submit_candidate` is
   executed by the controller after a completed round and a valid decision.

## Semantic Retrieval Trajectory

Use this family when the user asks to find, retrieve, compare, or analyze known
proteins or sequence-like candidates.

Canonical shape:

```text
search_sequence
-> sequence_analysis
-> merge_candidates
-> rank_designs
-> controller decision or report
```

Notes:

- Retrieval tools produce candidate records or candidate-like rows.
- `sequence_analysis` contributes sequence features and liability metrics.
- `merge_candidates` is the normalization point before ranking.
- `rank_designs` may omit optional auxiliary streams such as predictions,
  analyses, and QC results when upstream tools did not produce them.
- Add `structure_analysis` when the user provides structures, asks for fold,
  stability, topology, secondary-structure, structural similarity, or
  structure-backed functional comparisons. Do not add structure analysis for
  sequence-only retrieval unless it changes the requested decision/report.
- For ontology or family-language requests, resolve identifiers before sequence
  retrieval when needed: `function_term_resolver`, `go_term_search`,
  `ontology_lookup`, `go_sequences`, `interpro_entry_search`,
  `interpro_entry_lookup`, and `interpro_sequences` are first-class retrieval
  tools.
- Use `template_select`, `foldseek`, or `mmseqs2_gpu` for template, structural,
  or sequence-neighbor retrieval when the user asks for similarity evidence
  rather than free-form semantic search.

## Binder Design Trajectory

Use this family when the user provides a target structure or binding objective.

Canonical shape:

```text
binder generator (pxdesign, boltzgen, bindcraft, proteina_complexa,
                  rfdiffusion binder modes, rfpeptides)
-> Optional full structure prediction if the binder generator emits backbone structure only
-> quality metric (ipsae, protein_qc) and/or structure_analysis and/or sequence_analysis
-> merge_candidates
-> rank_designs
-> controller decision or report
```

Notes:

- Treat target structures and user constraints as root inputs.
- Downstream structure/ipsae/QC/ranking nodes should consume canonical candidates from
  upstream nodes, not planner-authored artifact paths.
- Add a structure-prediction/refinement step when a
  selected tool does not produce full-atom binder sidechain coordinates.
- PXDesign is planner-facing as a full-structure binder route, but the compiler
  may normalize PXDesign intent into the full-structure tail
  `pxdesign -> protenix_v2 -> protein_qc -> rank_designs` when downstream
  consumers require full-structure candidates.
- PXDesign preview fallback must use real chain provenance from the generated
  preview artifact; binder-only preview mode should not invent a conditioning
  chain.
- For ligand or small-molecule binding objectives, prefer ligand-aware branches
  such as `ligandmpnn`, `proteina_complexa` ligand-binder mode, `vina`,
  `boltz2_affinity`, and structure/QC scoring as requested by the deliverable.

## De Novo Design Trajectory

Use this family when the user asks to generate new proteins without a target
binding interface.

Canonical shape:

```text
rfdiffusion, rfdiffusion3, rfpeptides, or other backbone generation
-> proteinmpnn or sequence generation
-> protenix_v2 for structure prediction
-> protein_qc
-> sequence_analysis and structure_analysis
-> rank_designs
-> controller decision or report
```

Notes:

- Keep expensive GPU fan-out explicit through workflow or compiler-supported
  batching, not through hidden prompt assumptions.
- If a downstream predictor can derive its input JSON or structure artifact from
  upstream candidates, the LLM should not provide that artifact as a literal.
- Use `esm2`/`esm3` for language-model sequence generation or refinement when
  the request is primarily sequence/family semantic rather than fixed-backbone
  inverse folding. Use `proteinmpnn`/`ligandmpnn` when there is a real backbone
  or ligand-aware structural context.
- Add refinement/scoring tools only when they change the deliverable:
  `pyrosetta`, `ipsae`, `prodigy`, `pxmeter`, `tmalign`, `vina`,
  `boltz2_affinity`, `protein_qc`, `sequence_analysis`, and
  `structure_analysis` are evidence producers for ranking/reporting, not
  substitutes for a missing generator.
- For semantic, language-interfaced functional constraints such as GO/InterPro-like descriptions,
  ground the design objective before ranking: retrieve natural exemplars with
  `search_sequence`, build a family reference with `family_profile`, pass that
  reference as `profile_source` into supported generators/refiners for soft
  `profile_guidance`, then run `family_profile` again with generated candidates
  as `candidate_source` and the reference profile as `profile_source` before
  `rank_designs`.
- Prefer sequence-model guidance for semantic family/function constraints:
  `search_sequence -> family_profile(build) -> esm2/esm3(profile_guidance) ->
  family_profile(score) -> rank_designs`. This is the default when the request
  describes function, family, domain, GO, InterPro, enzyme class, or motif-like
  sequence evidence without a required backbone.
- When de novo backbone generation is also needed, use the profile as soft
  evidence rather than a hard structural motif:
  `search_sequence -> family_profile(build) -> rfdiffusion(length/advisory) ->
  proteinmpnn(normal or cautious bias) -> family_profile(score) ->
  rank_designs`. RFDiffusion should only receive length/advisory guidance from
  sequence-only profiles.
- Keep natural family evidence and generated design candidates on separate
  Planner IR relationships: reference evidence flows as `profile_source`, while
  executable designs flow as `candidate_source`.
- `family_profile(build)` emits profile evidence only; run
  `family_profile(score)` before ranking so `rank_designs` sees annotated
  candidate scores rather than raw profile evidence.
- Profile guidance into `esm2`, `esm3`, `rfdiffusion`, or `proteinmpnn` is soft
  evidence. Sequence-only profiles must not be converted into structural motif
  constraints unless real structure evidence is also present.
- ProteinMPNN profile bias is a refiner-level aid, not the primary semantic
  grounding tool. Apply it only when the designed chain or segment length maps
  clearly to the profile consensus; otherwise preserve normal ProteinMPNN
  behavior and rely on `family_profile(score)` for ranking evidence.
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
- create `vpa report arun_<id>` for successful agent-driven
  design/retrieval runs so the report includes all workflow rounds for the user
  query, unless the user explicitly asked for a job- or workflow-specific report;
- collect PASS evidence with agent run id, workflow ids, provider/model, report
  path, and final candidate/answer summary.
