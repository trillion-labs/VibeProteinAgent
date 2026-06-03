---
description: Install or run VibeProteinAgent from a natural-language protein request
argument-hint: "<install or protein workflow request>"
---

Read `AGENTS.md` and `skills.md` first.

If the user asks to install or prepare VPA, follow
`agent_workflows/vpa_installation.md`. Use `/vpa` as the default VPA home when
the user did not specify another path:

```bash
export VPA_HOME=/vpa
```

If the user asks to run, validate, rank, submit, report, or otherwise execute a
protein workflow, follow the VPA Protein Design Workflow Operating Protocol in
`skills.md`, using `agent_workflows/protein_design_skill.md`,
`agent_workflows/vpa_execution_environment.md`, and
`agent_workflows/vpa_contracts.md` as needed.

User request:

```text
$ARGUMENTS
```
