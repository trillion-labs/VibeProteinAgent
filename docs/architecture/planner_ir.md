# Planner IR and Canonical Candidates

VPA is moving away from asking LLMs to write executor-ready workflow JSON
directly. Exact executor references such as `${outputs.node.field}`,
`${jobs.node}`, `foreach`, and tool-specific field wiring are brittle model
outputs. They are better handled by code that can inspect schemas and compile a
semantic plan into a safe workflow.

Planner IR v1 is the LLM-facing contract for that separation. The Planner LLM
describes the goal, assumptions, missing inputs, candidate tool nodes, and
semantic upstream relationships. It says that one node depends on another as a
`candidate_source`; it does not spell out executor references or workflow engine
details. A later compiler will turn this IR into executor `workflow_plan`
nodes by using adapter schemas, output contracts, and compatibility rules.

The intended responsibility split is:

- Planner LLM: choose semantic intent and topology from the available tools.
- Compiler code: convert that intent into schema-aware executor-safe wiring.
- Executor: run the compiled jobs.
- Decision LLM: inspect actual outputs and choose `submit_candidate`,
  `next_round`, or `no_acceptable_candidate`.

Planner IR deliberately does not include executor references, `foreach`,
`max_parallel`, task-specific workflow templates, `workflow_ref`,
`submit_final`, or `submit_candidate` as a planned node. Final submission is a
post-round decision made after real outputs exist.

Canonical Candidate v1 is the future producer output shape for protein-design
style candidates. Candidate identity is producer-rooted: `source.local_id` is
required, while `source.native_id` is optional and only applies to external
database entries. The same sequence emitted by two different producer nodes can
be two different candidates because provenance matters.

Scores and artifacts are namespaced by producer tool:

```json
{
  "scores": {
    "rank_designs": {
      "score": 0.91
    }
  },
  "artifacts": {
    "boltzgen": {
      "structure_path": "design_0001.pdb"
    }
  }
}
```

This avoids unscoped top-level drift while leaving each adapter room to expose
its own metrics and files. Later PRs can use canonical candidates to let the
compiler connect candidate producers, QC tools, rankers, and submission
decisions without asking the LLM to guess field names.

This PR only adds the contracts, validation helpers, tests, and this document.
It does not change planner prompts, controller behavior, adapters, executor
runtime behavior, or live provider validation.
