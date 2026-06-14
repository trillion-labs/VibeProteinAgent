---
name: vpa-install-and-readiness
description: Install, repair, and verify VPA runtime readiness including CUDA, sidecars, binaries, weights, and credentials. Use for environment setup and doctor checks; not for launching tools or designing proteins.
---

# VPA Install And Readiness

Use this when the task is installation, repair, doctor checks, missing weights,
CUDA mismatch, missing binaries, credentials, or tool readiness.

## Approach

- If VPA is not already installed and ready, install with:

```bash
npx -y vibeproteinagent install --profile full --cuda-profile auto --yes
```

- If ESM3 is required, set `HF_TOKEN` in the same environment before running
  that command.
- Do not ask users to manually install sidecar envs or checkpoints for tools in
  the full profile.
- Use `vpa install doctor --profile full` after install or repair.

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
