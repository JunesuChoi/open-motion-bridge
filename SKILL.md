---
name: open-motion-bridge
description: Analyze a local video into reviewable pose, object, and camera tracking data, then generate approved HyperFrames or SVG sketch projects through the open-motion-bridge CLI. Use for video-to-code motion reconstruction, coordinate-verified overlays, and tracked sketch animation; do not use to upload or publish media.
---

# Open Motion Bridge

Use this Skill when a user wants an existing local video reconstructed as code-controlled motion graphics, tracked SVG sketch animation, or both. The goal is a reviewable data-to-code workflow, not a claim that a generated visual exactly recreates unverified footage.

## Before invoking tools

1. Confirm the input is a local path and the user has permission to process it.
2. Do not add the video, frames, outputs, model weights, or credentials to git.
3. Ask for the desired output profile only if it cannot be inferred: source, YouTube 16:9, YouTube Shorts 9:16, Instagram Reel 9:16, Instagram Feed 4:5, or custom.
4. Convert any natural-language direction into a candidate EditSpec. Show unresolved targets, conflicts, and unsupported directions instead of guessing.
5. Verify `ffmpeg` and the selected providers are locally available. An unavailable optional provider is a reported limitation, not a reason to fabricate output.

## Tool chain

When the implementation is available, use this order:

```text
preflight -> analyze -> validate IR -> verify overlay -> resolve edits
-> review or auto-review -> generate target project -> target check/snapshots
-> user approval -> final render
```

- Run `omb analyze <video>` to create source manifest and immutable tracking IR.
- Run `omb verify <tracking-ir>` to open or prepare the overlay review.
- Store user/agent-approved changes as `edits.patch.json`; never replace `tracking.ir.json`.
- Run `omb edit` to create a reproducible `resolved.ir.json`.
- Run `omb generate` separately for HyperFrames and/or SVG sketch. Use the same resolved IR so outputs remain comparable.

## Review policy

Manual review is the default. `auto-review` is allowed only for tracks satisfying the project thresholds: confidence at least 0.80, continuity at least 0.95, no drift warning, no occlusion/ID change in range, and camera confidence at least 0.75 when stabilization is used.

If a condition fails, mark the affected range `needs-review`. Offer re-analysis, ROI selection, exclusion, or a clearly recorded manual correction. Do not call final rendering complete before required user approval.

## Output selection

- **HyperFrames:** primary target for deterministic HTML composition; preserve generated source maps back to IR/patch operations.
- **SVG sketch:** use exact tracked paths first, then apply rough/freehand styling as a separate, configurable pass.
- **Remotion:** optional adapter only; never replace the HyperFrames source of truth.

Choose `screenSpace` for graphics that move with the original frame and `stabilizedSpace` for world-pinned graphics. If camera confidence is insufficient, retain screen-space behavior and disclose the limitation.

## Completion report

Report the source hash, selected providers, review policy/result, approved and unresolved tracks, profile, generated paths, warnings, and validation evidence. Do not claim an executable render, successful model use, or visual match without the relevant completed command and inspected output.

## References

- Read [Tracking IR](docs/ir-schema.md) when mapping providers or coordinates.
- Read [EditSpec](docs/edit-spec.md) when interpreting a natural-language request.
- Read [Validation](docs/validation.md) before approval or final render.
- Read [Risks and licensing](docs/risks-and-licenses.md) when enabling a provider, handling real media, or preparing a public release.
