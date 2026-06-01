# Planner IR And Canonical Candidates

Planner IR is the boundary between LLM intent and executable VPA workflows. The
planner describes the scientific topology; VPA code compiles that topology into
schema-aware workflow nodes, parameter wiring, candidate streams, and runtime
jobs.

## Responsibility Split

| Component | Responsibility |
|---|---|
| Planner LLM | Choose tools, modes, assumptions, missing inputs, and semantic upstream relationships. |
| Compiler | Convert semantic relationships into executor-safe workflow wiring using tool descriptors and compatibility rules. |
| Executor | Submit jobs, resolve `${params.*}` / `${outputs.*}`, track cache keys, and collect outputs. |
| CandidateArchive | Store and materialize typed candidate records, including prior-round survivor snapshots. |
| Decision LLM | Inspect bounded prompt context and choose answer, next round, submit candidate, or no acceptable candidate. |
| Report layer | Render stored job/workflow/agent evidence without becoming an execution source. |

The planner should not author brittle executor details such as raw
`${outputs.node.field}` candidate payloads, downstream artifact paths that can
be derived by adapters, or final `submit_candidate` nodes before real outputs
exist.

## Planner IR Shape

Planner IR v1 is semantic. A node says what tool should run and why. Edges use
relationships such as `candidate_source` instead of hard-coded executor field
references. The compiler owns mode validation, canonical candidate wiring,
merge insertion, and parameter ownership checks.

For tools with explicit modes, such as `rfdiffusion`, `esm2`, `foldseek`,
`pxdesign`, `sequence_edit`, `pyrosetta`, and `proteina_complexa`, the planner
must select an allowed mode when the tool schema requires one. Invalid or
missing modes are returned to the planner as structured fixups.

## Canonical Candidate Contract

Canonical Candidate v1 is the typed handoff shape for design, retrieval, QC,
ranking, and submission. Candidate identity is provenance-aware: two producers
can emit identical sequences as different candidates because source matters.

Scores and artifacts are namespaced by producer tool:

```json
{
  "schema_version": "candidate_v1",
  "candidate_id": "cand_123",
  "source": {"tool": "protein_qc", "local_id": "qc_0000"},
  "sequence": "ACDEFGHIKLMNPQRSTVWY",
  "scores": {
    "protein_qc": {
      "gravy": -0.49,
      "pi": 6.78
    }
  },
  "artifacts": {
    "protein_qc": {
      "summary_json": "..."
    }
  }
}
```

Execution surfaces must hydrate candidate inputs from `CandidateArchive` or
adapter outputs, not from prompt summaries, report tables, or compact display
strings.

## Prior Rounds And Reports

Prior-round candidates are materialized from exact survivor snapshots in
`CandidateArchive`. Prompt summaries can guide the next decision, but they are
not candidate pools.

Reports are read-only. They may hydrate legacy archive snapshots for display
compatibility, but report summaries and `agent_summaries` remain transcript and
visualization surfaces only.
