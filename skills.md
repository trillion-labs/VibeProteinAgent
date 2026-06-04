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
design, evaluation, ranking, validation, report generation, or a
concrete VPA run. In execution mode, use VPA canonical workflows and preserve the
compiler/executor ownership contracts. Do not treat LLM summaries, display
summaries, prompts, or report summaries as executable candidate sources.
When the user asks to install, repair, or prepare VPA or a VPA tool, read
[agent_workflows/vpa_installation.md](agent_workflows/vpa_installation.md) and
perform deterministic installer/preflight steps before execution.

The skills-aware assistant is the route chooser. Pick an explicit VPA command
before calling the runtime:

- use `vpa run <tool>` for a known single-tool/debug run;
- use `vpa status`, `vpa wait`, `vpa results`, `vpa logs`, `vpa artifacts`, or
  `vpa report` for existing `job_`, `wf_`, or `arun_` state;
- use `vpa workflow run` when the user supplies or requests a concrete workflow
  descriptor;
- use `vpa agent run` only for an LLM-authored scientific workflow plan.

`vpa agent run` is not a natural-language router for direct tool/status/report
or explicit workflow operations. Tool names inside an agent request, including
phrases like "include rfdiffusion", are planner context rather than direct
adapter dispatch.

## Repository And Command Map

- Python package and CLI live under `src/vibeproteinagent/`; installed commands
  are `vpa` and `vibeproteinagent`.
- CLI entry points are wired in `src/vibeproteinagent/cli/main.py`: `tools`,
  `tool`, `workspace`, `workflow`, `daemon`, `mcp`, `agent`, `install`, plus
  top-level `run`, `status`, `wait`, `logs`, `results`, `cancel`, `retry`,
  `artifacts`, `download`, `export`, `doctor`, `visualize`, and `report`.
- Built-in tool descriptors/adapters live in `src/vibeproteinagent/adapters/`.
  Important groups include retrieval/ontology, sequence editing, backbone and
  binder design, structure prediction, QC/scoring/refinement, docking/affinity,
  candidate merge/rank/submission, and converters.
- Agent planning/runtime lives in `src/vibeproteinagent/agents/`; workflow DAG
  loading/execution lives in `src/vibeproteinagent/workflows/`; durable state is
  in `src/vibeproteinagent/workspace/`; workers/sidecars are in
  `src/vibeproteinagent/workers/`; reports are in
  `src/vibeproteinagent/reporting/`.
- Prefer installed `vpa` commands. From a source checkout, set
  `PYTHONPATH=$PROJECT_ROOT/src` and run
  `python -m vibeproteinagent.cli.main ...`.

## Canonical Workflow Families

- Semantic retrieval: search or retrieve sequences, analyze them, merge
  canonical candidates, rank designs, then make a post-round
  `submit_candidate` decision or report evidence. GO/InterPro/function requests
  may use `function_term_resolver`, `go_term_search`, `ontology_lookup`,
  `go_sequences`, `interpro_entry_search`, `interpro_entry_lookup`, and
  `interpro_sequences` before `search_sequence`/`family_profile`.
- Template and structural retrieval: retrieve templates or structural neighbors
  with `template_select`, `foldseek`, or `mmseqs2_gpu`, then analyze, merge,
  rank, and report evidence.
- Sequence engineering/refinement: mutate or refine existing sequences with
  `sequence_edit`, `esm2`, or `esm3`, then analyze/QC, merge, rank, and report.
- Binder design: generate binder/backbone candidates for a target, evaluate
  structure and QC outputs, merge candidates, rank designs, then make a
  post-round `submit_candidate` decision or report. Supported routes include
  `pxdesign`, `boltzgen`, `bindcraft`, `proteina_complexa`, RFDiffusion binder
  modes, `rfpeptides`, and ligand-aware `ligandmpnn`/`vina`/affinity variants
  when the user asks for ligand or small-molecule binding.
- De novo design: generate backbones/sequences, predict or refine structures,
  QC and analyze outputs, rank designs, then make a post-round
  `submit_candidate` decision or report. Supported generators/refiners include
  `rfdiffusion`, `rfdiffusion3`, `rfpeptides`, `proteinmpnn`, `ligandmpnn`,
  `esm2`, `esm3`, `protenix_v2`, `boltz2_affinity`, and refinement/scoring
  tools such as `pyrosetta`, `protein_qc`, `ipsae`, `prodigy`, `pxmeter`,
  `tmalign`, and `structure_analysis`.
- Semantic de novo design should first ground the function/family objective with
  `search_sequence -> family_profile(build)`. Prefer `esm2`/`esm3`
  `profile_guidance` for sequence-family semantics; use RFDiffusion only for
  length/advisory guidance from sequence-only profiles, and use ProteinMPNN
  profile bias only when chain/segment mapping is clear.

For detailed trajectory patterns, read
[agent_workflows/protein_design_skill.md](agent_workflows/protein_design_skill.md).
For portable workspace setup, command execution, and report generation, read
[agent_workflows/vpa_execution_environment.md](agent_workflows/vpa_execution_environment.md).

## Hard Contracts

- `MemoryBank` is a lossless raw ledger. It preserves raw workflow, node, job,
  LLM, decision, candidate extraction, and validation events.
- `CandidateArchive` is the only typed source for executor candidate
  materialization, including `${outputs.prior_round.candidates}`.
- `profile_source` carries family reference evidence, not executable
  candidates. Generated candidates must stay on `candidate_source` and be
  annotated with `family_profile(score)` before ranking.
- Planner IR expresses dataflow with `upstream` relationships such as
  `candidate_source`, `profile_source`, `term_source`, and `identifier_source`.
  Do not author executor references such as `${outputs...}`, `${jobs...}`, or
  `${item...}` in Planner IR except where compiler fixup explicitly normalizes a
  canonical candidate-stream shorthand.
- Terminal controller actions such as `submit_candidate` are post-round
  decisions. They are not Planner IR workflow nodes.
- `PromptContext` is a lossy LLM digest. It may contain candidate cards,
  capability summaries, failure cards, and next-move hints, but not raw
  candidate objects, full sequences, raw score dicts, or structure paths.
- `ResultSummarizer`, `agent_summaries`, report display summaries, and HTML
  compaction are display/transcript surfaces only.
- Planner/tool selector/decision LLMs describe intent and choose actions; they
  do not own candidate-stream payloads or downstream source artifacts that the
  compiler/runtime can derive from upstream candidates.

For implementation guardrails, read
[agent_workflows/vpa_contracts.md](agent_workflows/vpa_contracts.md).

## Evidence Protocol

For successful executable runs, produce a VPA report or a report set that points
to the exact user-query/run evidence. A pass is not just "the CLI exited":
record the provider, workflow family, run directory/workspace, agent run id,
final decision, submitted candidate or answer, report path, and any
environment-only caveats. Unless the user explicitly asks for a job- or
workflow-specific report, report on the full `arun_...` agent run rather than a
single workflow round or job.
