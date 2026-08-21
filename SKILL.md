---
name: open-motion-bridge
description: Analyze a local video into reviewable pose tracking data, bind text or image assets to named landmarks through a declarative EditSpec, generate HyperFrames or SVG sketch projects, and verify an externally rendered result by reprojecting resolved coordinates onto the rendered pixels. Use for video-to-code motion reconstruction, coordinate-verified overlays, and tracked sketch animation. This Skill does not render video and does not upload or publish media.
---

# Open Motion Bridge

Use this Skill when a user wants an existing local video reconstructed as code-controlled motion graphics, tracked SVG sketch animation, or both. The goal is a reviewable data-to-code workflow, not a claim that a generated visual exactly recreates unverified footage.

**Scope boundary — no rendering.** This Skill produces data and generated source (Tracking IR, resolved bindings, a HyperFrames project, SVG). Turning the generated project into an mp4 is the responsibility of an external renderer (for example the HyperFrames CLI) invoked by the calling agent, outside this Skill. The Skill's own `verify` command then measures that externally rendered file; it never claims a render it did not inspect.

## Before invoking tools

1. Confirm the input is a local path and the user has permission to process it.
2. Do not add the video, frames, outputs, model weights, or credentials to git.
3. Ask for the desired output profile only if it cannot be inferred: source, YouTube 16:9, YouTube Shorts 9:16, Instagram Reel 9:16, Instagram Feed 4:5, or custom.
4. Convert any natural-language direction into a candidate EditSpec (`edits.spec.json`). Show the resolved JSON to the user; surface unresolved targets, conflicts, and unsupported directions instead of guessing.
5. Verify `ffmpeg` and the selected providers are locally available. An unavailable optional provider is a reported limitation, not a reason to fabricate output.

## Tool chain

Implemented commands, in order:

```text
analyze -> author EditSpec -> generate (resolves bindings)
-> [external render, outside this Skill]
-> verify (reprojection against the rendered file) -> report
```

- `python -m open_motion_bridge analyze <video> --output <dir>` creates the source manifest and immutable `tracking.ir.json`. Never mutate that file afterwards.
- Author `edits.spec.json` from the user's direction. Bindings attach `text` or `image` assets to named landmarks with `onLowConfidence` (fade/hide/hold), bbox-relative `scale`, landmark-pair `rotate`, time `range`, and `maxSpeed` clamping. See [EditSpec](docs/edit-spec.md).
- `python -m open_motion_bridge generate <tracking-ir> --source-video <video> --output <project> --edit-spec <spec> --overlay bindings` writes the HyperFrames project plus `bindings.resolved.json` (the auditable per-frame transform table) and `render.tracking.ir.json`. Use `--overlay skeleton` for diagnostics, `both` for review. `--target sketch-svg` emits the exact SVG trace.
- Rendering the generated project is not part of this Skill. Hand the project directory to the external renderer chosen by the user or calling workflow.
- `python -m open_motion_bridge verify <project> --rendered-video <mp4> --output <report.json>` re-measures where staged image assets actually landed in the rendered pixels against `bindings.resolved.json` and writes a pass/fail report with per-sample pixel errors. Text bindings are reported as not measurable; verify them with an inspected extracted frame instead of implying a measured pass.

## Review policy

Manual review is the default. `auto-review` is allowed only for tracks satisfying the project thresholds: confidence at least 0.80, continuity at least 0.95, no drift warning, no occlusion/ID change in range, and camera confidence at least 0.75 when stabilization is used.

If a condition fails, mark the affected range `needs-review`. Offer re-analysis, ROI selection, exclusion, or a clearly recorded manual correction. Review the resolved binding stats (`visibleFrames`, `clampedFrames`, `missingLandmarkFrames`, state counts) before hand-off; a binding whose anchor is mostly `missing-landmark` or `hidden-low-confidence` needs a different anchor or provider, not a render.

## Output selection

- **HyperFrames:** primary target for deterministic HTML composition; preserve generated source maps back to IR/patch operations.
- **SVG sketch:** use exact tracked paths first, then apply rough/freehand styling as a separate, configurable pass.
- **Remotion:** optional adapter only; never replace the HyperFrames source of truth.

Choose `screenSpace` for graphics that move with the original frame and `stabilizedSpace` for world-pinned graphics. If camera confidence is insufficient, retain screen-space behavior and disclose the limitation.

## Completion report

Report the source hash, selected providers, review policy/result, resolved binding stats, profile, generated paths, warnings, and — when an externally rendered file was verified — the reprojection summary (`measurableBindings`, `passedBindings`, `maxErrorPx`, tolerance). Never claim a render happened inside this Skill, and never claim a visual match without a completed `verify` run or an inspected frame.

## References

- Read [Tracking IR](docs/ir-schema.md) when mapping providers or coordinates.
- Read [EditSpec](docs/edit-spec.md) when interpreting a natural-language request.
- Read [Validation](docs/validation.md) before approval or hand-off to an external renderer.
- Read [Risks and licensing](docs/risks-and-licenses.md) when enabling a provider, handling real media, or preparing a public release.
