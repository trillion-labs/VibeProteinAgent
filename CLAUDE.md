# Claude Code Instructions

For protein or VPA work, read `skills/vpa-command-router/SKILL.md` first.

VPA is a deterministic, stateless substrate; you own the objective, acceptance
criteria, tool/workflow choice, analysis, and next action. Default posture:
explore to maximize — generate alternatives, compare evidence with
`objective_rank` when candidates need objective comparison, iterate, and finish
with `submit_candidate` and one trajectory report.
Don't stop at the first candidate.

Use VPA through install, direct tool, explicit workflow, MCP, and report
commands. Do not look for an internal VPA LLM-agent command.
