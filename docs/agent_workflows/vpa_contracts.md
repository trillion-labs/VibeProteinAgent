# VPA Agent Contracts

These are the operational contracts agents must preserve while planning,
executing, debugging, or documenting VPA workflows.

## Memory, Candidates, Prompts

`MemoryBank` is a lossless raw ledger. It stores raw workflow summaries, node and
job payloads, LLM events, decisions, candidate extraction events, validation
events, and snapshot ids. It must not depend on compact display sentinels such
as `"<dict>"` or `"<list>"`.

`CandidateArchive` is the typed materialized source for executor candidate
inputs. It owns candidate records, exact version snapshot refs, rejected records,
merge audits, survivor snapshots, and executor materialization for prior-round
candidates.

`PromptContext` is a lossy digest for planner/tool selector/decision LLMs. It
may include bounded candidate cards, capability summaries, failure cards,
trajectory cards, and next-move hints. It must not include raw candidate
objects, full sequences, raw nested scores, full artifact dicts, or structure
paths.

## Non-Sources

The following surfaces are not executor candidate sources:

- `ResultSummarizer` output;
- `outcome.summary`;
- `agent_summaries`;
- report display summaries or HTML compaction;
- prompt text or LLM reasoning;
- legacy helper output unless explicitly in a report-only old-workspace fallback.

If a candidate pool is needed for execution, hydrate it from
`CandidateArchive.materialize(...)` or
`CandidateArchive.materialize_executor_outputs(...)`.

## Planner And Compiler Ownership

Planner LLMs choose semantic topology and intent. Compiler/runtime code owns
schema-aware executor wiring.

Rules:

- Candidate-stream fields are not planner-owned payloads.
- Optional `rank_designs.predictions`, `rank_designs.analyses`, and
  `rank_designs.qc_results` may be omitted when empty planner literals appear;
  non-empty planner literal candidate streams remain invalid.
- Downstream source artifacts that can be derived from upstream candidates, such
  as a predictor input JSON, should not pass through as planner literals.
- User-provided, known-input, compiled-wiring, runtime-derived, upstream, and
  adapter-default values must not be deleted merely because they are empty.
- Root nodes may legitimately use literal source artifacts supplied by the user.

## Prior-Round Contract

Prior-round executor candidates come only from `CandidateArchive` survivor
snapshot materialization. Planner/tool selector/decision prompts may receive the
prompt context for the prior round, but not raw survivor candidate arrays or
snapshot internals.

Allowed planner expression:

```json
{"node": "prior_round", "relationship": "candidate_source"}
```

Do not rely on compact candidate summaries or report candidate tables for
next-round execution.

## Decision Contract

The decision stage chooses one of the supported post-round actions, such as
submitting a candidate, moving to a next round with or without candidates, or
ending with no acceptable candidate. Decision LLM payloads should receive
`PromptContext`, success criteria, and local deterministic decision context, not
raw candidate pools.

`submit_candidate.job_id` must resolve to a real `job_...` id from the executed
workflow. Human-readable node aliases are acceptable only if controller logic
resolves them before validation.

## Failure Guardrails

- Compact candidate poisoning: a value like `scores: "<dict>"` is invalid for
  executor materialization and should be rejected before merge/rank execution.
- Empty optional rank streams: planner-owned `[]` for auxiliary candidate-stream
  fields should be omitted, while non-empty planner-owned payloads stay invalid.
- Source artifact ownership: downstream artifacts derived from candidates belong
  to adapters/runtime, not LLM literals.
- Prior-round shorthand: known shorthand can be normalized by the compiler, but
  unknown candidate-stream strings must fail before executor validation.
- Report fallback: old archive snapshots may be hydrated for report
  compatibility, but `agent_summaries` remain transcript/timeline only.

