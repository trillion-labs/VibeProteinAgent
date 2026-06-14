# Third-Party Notices

VibeProteinAgent includes vendored JavaScript assets for self-contained report
and visualization HTML output.

## 3Dmol.js

- Files:
  - `src/vibeproteinagent/reporting/assets/3Dmol-min.js`
  - `src/vibeproteinagent/visualization/assets/3Dmol-min.js`
- License: BSD-3-Clause, with bundled notices for GLmol, Three.js, and jQuery.
- Upstream: https://github.com/3dmol/3Dmol.js
- License text:
  - `src/vibeproteinagent/reporting/assets/3Dmol-min.js.LICENSE.txt`
  - `src/vibeproteinagent/visualization/assets/3Dmol-min.js.LICENSE.txt`

## Tool source and sidecars

The repository also records vendored or pointer-vendored tool integrations under
`vendor/`. Each integration keeps its upstream license text and provenance in
`vendor/<tool>/LICENSE`, `vendor/<tool>/UPSTREAM.md`, or, for multi-license
trees, the local `licenses/` directory.

The Python wheel/sdist does not bundle the full `vendor/` tree. The npm
bootstrap package includes the VPA source, installer scripts, and vendored tool
source/provenance needed for source-fallback installs. Local model weights,
checkpoints, generated environments, caches, and large example/data payloads are
excluded from npm packaging by `vendor/.npmignore`.
