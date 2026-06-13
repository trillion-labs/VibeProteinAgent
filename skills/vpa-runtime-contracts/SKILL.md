---
name: vpa-runtime-contracts
description: Preserve deterministic VPA runtime contracts for candidate sources, workflow dataflow, reporting, and audits. Use when editing VPA code, authoring workflows, or debugging candidate flow; not for running a design objective (use vpa-protein-design-orchestration).
---

# VPA Runtime Contracts

Use this when editing VPA code, writing workflows, debugging candidate flow, or
checking report/audit behavior.

## Candidate Sources

Valid candidate sources are adapter outputs, workflow node outputs, candidate
records, merge/rank outputs, and explicit `submit_candidate` payloads. LLM
summaries, HTML report summaries, display cards, prompts, and transcript text
are not executable candidate sources.

## Dataflow

Workflow YAML owns deterministic dataflow:

```text
producer candidate output -> consumer candidate input
```

Candidate streams carry canonical candidates, not every side artifact a
consumer may need. Example: Protenix -> ipSAE needs both the structure
candidate and the matching PAE/confidence file; concrete YAML wiring belongs in
vpa-run-and-workspace.

Use `merge_candidates` when multiple streams must become one stream. Run
`objective_rank` before final selection by default once scoring/structure
evidence exists (even for a single candidate, so the report renders ranking and
score from tool output), and provide explicit gates, score_formula, direction,
and tie-breakers.

## Reportable Evidence

Adaptive top-k validation is allowed for expensive tools, but the filter must
be reportable: record the input count, selected/skipped candidate ids, selection
reason, selection metrics, and downstream tool in a candidate selection/filter
event when possible. Reports should show generated, evaluated, ranked, and
submitted populations separately.

Evaluator tools should consume canonical `candidates` whenever available. Raw
paths are acceptable for one-off diagnostics, but do not use an unrecorded
manual table or transcript-only calculation as the sole final-selection
evidence. Put evidence in tool output or a structured recorded artifact.
Raw manual complex JSON/path evidence is diagnostic until it is reattached to
the logical candidate through canonical `candidates` output and `objective_rank`.
Evaluator-local ids are not global design ids when they can collide across
jobs. Preserve upstream candidate identity, source job ids, and material
identity through merge, rank, and submit.

## Reporting

Reports are read-only over jobs, workflows, artifacts, candidate records, and
legacy history. They must not create agent state, launch enrichment jobs by
default, invent final selections, or be hand-edited as HTML.
Final report preflight should be able to join accepted submissions to recorded
`objective_rank` rows when rank evidence exists. If the join is ambiguous,
record a better identifier through `submit_candidate` instead of explaining the
match in prose.

## Audit

Preserve candidate IDs, source tool/node/job, score provenance, artifacts,
merge/rank events, parameter resolution events, warnings, and submission
justification. Prefer compact pseudo-code in docs over copying implementation
details.
