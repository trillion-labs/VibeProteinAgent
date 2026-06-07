---
name: vpa-install-and-readiness
description: Install, repair, and verify VPA runtime readiness including CUDA, sidecars, binaries, weights, and credentials. Use for environment setup and doctor checks; not for launching tools or designing proteins.
---

# VPA Install And Readiness

Use this when the task is installation, repair, doctor checks, missing weights,
CUDA mismatch, missing binaries, credentials, or tool readiness.

## Approach

- Prefer `vpa install doctor` before changing the environment.
- Use `vpa install plan --profile full --cuda-profile auto` to inspect impact.
- Use `vpa install --profile full --cuda-profile auto --yes` for ordinary setup.
- Use `--tools <tool,...>` for focused repair when a specific tool is missing.
- Re-run `vpa install doctor` after changes and report remaining caveats.

## Readiness Checks

Check the exact failure surface first:

```text
vpa version
vpa install doctor --profile full
vpa tools --json
vpa tool info <tool> --json
```

For gated models or remote downloads, verify the relevant token or local cache.
For GPU tools, verify CUDA profile, driver compatibility, sidecar env, and
required binaries/weights.

Do not route install work through an internal LLM agent. The external agent
chooses the repair action and VPA executes deterministic installer steps.
