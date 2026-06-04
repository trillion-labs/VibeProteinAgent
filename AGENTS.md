# Agent Instructions

For protein or VPA work, read [skills.md](skills.md) first.

Use normal conversation for open-ended protein discussion. When the user asks to
run, validate, rank, submit, report, inspect existing `job_`/`wf_`/`arun_`
state, or otherwise execute protein work, switch to the VPA Protein Design
Workflow Operating Protocol in `skills.md`.

The assistant chooses the VPA command boundary before invoking the runtime:
direct tool/debug work uses explicit `vpa run` and job/status/result commands;
explicit workflow descriptors use `vpa workflow run`; reports use `vpa report`;
LLM-authored scientific workflow planning uses `vpa agent run`. Do not rely on
`vpa agent run` to infer direct tool, status, report, or workflow mode from free
text.
