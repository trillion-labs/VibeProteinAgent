---
description: Use VibeProteinAgent from a natural-language install or protein request
argument-hint: "<install, report, workflow, or protein design request>"
---

Read `skills/vpa-command-router/SKILL.md` first.

Respect the active installed runtime first. If `VPA_HOME` is already set and
`$VPA_HOME/env.sh` exists, source it before checking `vpa`. If `vpa` is then on
PATH, use it as-is. Only choose an install root when the user asks for
install/repair or no working VPA command exists.

Fallback install root, only when one is needed:

```bash
export VPA_HOME=${VPA_HOME:-/vpa}
test -f "$VPA_HOME/env.sh" && source "$VPA_HOME/env.sh"
```

Route the request through VPA install, direct tool, explicit workflow, MCP, or
report commands. The external Claude Code session owns objective extraction,
thresholds, tool selection, execution analysis, and next action.

User request:

```text
$ARGUMENTS
```
