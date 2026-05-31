# Protein Design Workflow Operating Protocol v1

This is the shared operating protocol for Codex, Claude Code, and other agentic
assistants working in this VPA repository. It is not a prompt library. Use it to
decide when a protein conversation becomes an executable VPA workflow and which
contracts must hold once execution starts.

## Mode Boundary

Stay in normal conversation mode for protein Q&A, brainstorming, literature
discussion, troubleshooting, and design critique. In that mode, answer directly
and do not force a workflow unless the user asks to run, validate, rank, report,
or submit candidates.

Switch to workflow execution mode when the user asks for end-to-end retrieval,
design, evaluation, ranking, live matrix validation, report generation, or a
concrete VPA run. In execution mode, use VPA canonical workflows and preserve the
compiler/executor ownership contracts. Do not treat LLM summaries, display
summaries, prompts, or report summaries as executable candidate sources.

## Canonical Workflow Families

- Semantic retrieval: search or retrieve sequences, analyze them, merge
  canonical candidates, rank designs, then submit or report evidence.
- Binder design: generate binder/backbone candidates for a target, evaluate
  structure and QC outputs, merge candidates, rank designs, then submit/report.
- De novo design: generate backbones/sequences, predict or refine structures,
  QC and analyze outputs, rank designs, then submit/report.

For detailed trajectory patterns, read
[docs/agent_workflows/protein_design_skill.md](docs/agent_workflows/protein_design_skill.md).

## Hard Contracts

- `MemoryBank` is a lossless raw ledger. It preserves raw workflow, node, job,
  LLM, decision, candidate extraction, and validation events.
- `CandidateArchive` is the only typed source for executor candidate
  materialization, including `${outputs.prior_round.candidates}`.
- `PromptContext` is a lossy LLM digest. It may contain candidate cards,
  capability summaries, failure cards, and next-move hints, but not raw
  candidate objects, full sequences, raw score dicts, or structure paths.
- `ResultSummarizer`, `agent_summaries`, report display summaries, and HTML
  compaction are display/transcript surfaces only.
- Planner/tool selector/decision LLMs describe intent and choose actions; they
  do not own candidate-stream payloads or downstream source artifacts that the
  compiler/runtime can derive from upstream candidates.

For implementation guardrails, read
[docs/agent_workflows/vpa_contracts.md](docs/agent_workflows/vpa_contracts.md).

## Evidence Protocol

For successful executable runs, produce a VPA report or a report set that points
to the exact workflow/run evidence. A pass is not just "the CLI exited": record
the provider, workflow family, run directory/workspace, final decision, submitted
candidate or answer, report path, and any environment-only caveats.

For live matrix operations, preflight credentials and provider CLIs first, track
phase-level status while the run is active, and classify failures by phase before
proposing fixes. See
[docs/agent_workflows/live_matrix_validation.md](docs/agent_workflows/live_matrix_validation.md).

