# Agent Instructions

For protein or VPA work, read `skills/vpa-command-router/SKILL.md` first.

VPA is a deterministic tool substrate. External Codex/Claude agents choose the
objective, acceptance criteria, tool/workflow boundary, analysis, and next
action.

Default posture: explore to maximize the objective. Try alternative tools and
approaches, compare the evidence, and iterate as the objective warrants. VPA is
stateless and does not model rounds, so the external agent owns the exploration
loop and runs as many jobs as it needs.

## Routing map

Claude Code auto-selects a skill from its description. Codex does not, so use
this map directly:

- Install / repair / CUDA / weights / binaries / credentials ->
  `skills/vpa-install-and-readiness/SKILL.md`
- Protein design objective (design, rank, submit, optimize) ->
  `skills/vpa-protein-design-orchestration/SKILL.md`
- Launch a known tool or explicit YAML workflow ->
  `skills/vpa-run-and-workspace/SKILL.md`
- Inspect existing `traj_`/`job_`/`wf_` state or render a report ->
  `skills/vpa-state-reporting/SKILL.md`
- Edit VPA code, author workflows, debug candidate flow ->
  `skills/vpa-runtime-contracts/SKILL.md`
- Expose tools over MCP -> `vpa mcp serve`, then let the agent choose tools.

Follow references one level deep; do not chain skill to skill to skill.
