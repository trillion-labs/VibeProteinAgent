# Contributing

Thank you for considering a contribution to VibeProteinAgent.

## License And Contribution Terms

VibeProteinAgent first-party code is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE). By submitting a pull request,
patch, issue attachment, or other contribution to this repository, you agree
that:

- you have the right to submit the contribution;
- you grant the project maintainers a perpetual, worldwide, royalty-free,
  irrevocable, sublicensable license to use, copy, modify, distribute,
  publicly display, publicly perform, and relicense your contribution as part
  of VibeProteinAgent;
- your contribution may be offered under the repository's current
  non-commercial license and under separate commercial licenses;
- your contribution does not include third-party code, model weights, datasets,
  or other materials unless their license permits inclusion here and you clearly
  identify the source and license.

If you cannot agree to these terms, please open an issue before submitting code.

## Third-Party Code And Models

Many protein-design tools, model weights, and vendored or pointer-vendored
components have their own licenses. Do not copy new upstream code, weights, or
datasets into this repository without adding provenance notes and the relevant
license files. When in doubt, prefer a pointer-vendored integration that
downloads or discovers the dependency outside the repository.

## Commercial Use

Commercial use of VibeProteinAgent first-party code requires a separate license
from the project maintainers. Third-party components may impose additional
terms, including academic-only, non-commercial, gated-download, or
non-redistribution requirements.

## Development Basics

- Keep changes scoped to the task at hand.
- Add or update tests for behavioral changes.
- Run focused tests for the code you changed.
- Do not rewrite vendored upstream license text.
