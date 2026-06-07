---
name: vpa-command-router
description: Choose the VPA command boundary for install, tool, workflow, MCP, report, and protein design execution. Use first for VPA/protein work.
---

# VPA Command Router

VPA is a deterministic tool substrate. The external agent owns objective
extraction, threshold choice, tool selection, execution monitoring, analysis,
and next action.

## Route

- Install, repair, CUDA, weights, binaries, credentials: use
  `skills/vpa-install-and-readiness/SKILL.md`.
- Known direct tool/debug run: use `vpa run <tool> ...`.
- User-provided or agent-authored YAML workflow: use `vpa workflow run <file>`.
- Existing state: use `vpa trajectory status`, `vpa status`, `vpa wait`,
  `vpa results`, `vpa logs`, `vpa artifacts`, or `vpa report`.
- MCP integration: expose VPA tools through `vpa mcp serve`; let the external
  agent choose and call tools.
- Protein design objective: use
  `skills/vpa-protein-design-orchestration/SKILL.md`, then execute through
  direct tools or explicit workflows.

## Locked vs open

Locked (do not improvise):

- State the objective and measurable acceptance criteria before any run. If the
  query lacks criteria, ask once or pick conservative explicit ones and report
  them.
- Final selection comes only from `submit_candidate` output (`accepted` or
  `no_acceptable_candidate`), justified against the objective. Never infer a
  final candidate from a report or summary alone.
- Track each user query with one `traj_` trajectory and finish with one HTML
  trajectory report.
- Quote sequences, scores, confidence, and pass/fail only from tool outputs, not
  from model knowledge.

Open (your judgement):

- Which tools/workflows to explore. Try alternatives and iterate to maximize the
  objective.
- How long to keep iterating. VPA is stateless and does not model rounds, so you
  own the loop — keep each job/workflow/report contract explicit.

Use `rank_designs` to compare evidence when useful.
