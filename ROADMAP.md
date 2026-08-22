# Roadmap

This roadmap intentionally puts data contracts, reviewability, and test fixtures before model breadth or generated visual polish. A phase is not complete until its stated evidence exists.

Direction note (2026-08): the core video loop — analyze, EditSpec asset binding, deterministic generation, and render reprojection verification — is implemented. Still-image analysis, MotionSpec generation, and progressive ink-to-color drawing with Lab-grouped, real RGB coordinate strokes and selectable close-up modes are also implemented. The older source-reveal mask remains a compatibility mode, while `sampled-strokes` is the recommended path and `hybrid-paint` limits source pixels to a subtle texture finish. Video rendering itself is explicitly out of scope for this project: generated projects are handed to an external renderer, and this project verifies the externally rendered result. The phases below are re-prioritized around that boundary: verification depth first, then subject-aware still-image layers, expressive breadth, and skill packaging.

## Next priorities

1. **Text binding verification.** Rasterize text bindings to transparent PNGs at generate time so `verify` can template-match them like images, closing the "not measurable" gap and removing browser font dependence from the visual result.
2. **Synthetic fixture end-to-end tests.** Programmatically generated motion clips (known ground-truth coordinates, no real people) driving analyze -> generate -> verify in CI, asserting pixel error bounds numerically.
3. **Failure diagnostics.** On an out-of-tolerance `verify` sample, save a side-by-side diagnostic image (expected vs. measured position) next to the report.
4. **Subject-aware photo layers.** Add opt-in local segmentation and depth ordering so hair, face, clothing, and background can receive independent parallax, brush order, and occlusion-aware close-ups without pretending whole-image effects isolate regions.
5. **Drawing verification.** Extend the deterministic sampled-stroke fixtures with pixel-level monotonic painted-area, color-error, final full-frame camera release, and brush/tool alignment measurements in addition to inspected snapshots.
6. **Camera motion provider.** Optical-flow/homography estimation to enable `stabilizedSpace` (world-pinned graphics).
7. **Binding kinds.** Landmark-pair lines/arrows, joint-angle labels, and motion trails — all derivable from the existing IR.
8. **Multi-person tracks.** Persist multiple IoU-matched tracks so an EditSpec can target `person-002`.
9. **Workspace convention and one-command flow.** `runs/<source-hash>/` caching plus a single orchestration command covering analyze -> generate (verify stays a separate, post-external-render step).
10. **NL -> EditSpec/MotionSpec example corpus.** Korean/English instruction-to-spec pairs under `integrations/` so Codex and Claude translate equivalent direction consistently.

## Phase 0 — Repository and reproducible contracts

**Outcome:** A Windows-first monorepo scaffold that can be installed on macOS/Linux where dependencies permit, with no real media in version control.

Issues:

- Create Python and TypeScript workspace manifests and a single version policy.
- Add MIT license, media ignore rules, contributor media policy, and optional-provider license registry.
- Publish JSON Schema for Tracking IR, EditSpec, patch history, and provenance.
- Add synthetic fixture manifests and expected outputs; no real photos, faces, or video frames.
- Add CI jobs for schema validation, formatting, type checks, and fixture-only tests.

Exit evidence:

- A clean clone contains no user video or model weight.
- Every fixture validates against its schema on Windows CI and one Unix CI runner.
- The repository documents an explicit local `ffmpeg` discovery policy.

Recovery: revert schema changes by versioning a new minor schema; do not mutate an already-published schema in place.

## Phase 1 — Ingest, timing, and immutable IR

**Outcome:** A local video can be inspected, frame-addressed, and represented without tracking.

Issues:

- Implement `omb analyze` ingest preflight: path, codec, duration, dimensions, variable-frame-rate detection, audio stream, and source hash.
- Implement ffmpeg frame/timestamp extraction with original presentation timestamps preserved.
- Produce an empty but valid Tracking IR and source manifest.
- Implement profile coordinate transforms for source, 16:9, 9:16, and 4:5 canvases.
- Add fixture tests for constant-frame-rate, variable-frame-rate, rotated, and no-audio manifests.

Exit evidence:

- A selected source frame can be mapped from timestamp to IR frame index and back within one source-frame tolerance.
- Source IR remains byte-identical after an edit workflow.

Recovery: preserve raw ffprobe and timestamp tables in provenance so an ingest adapter can be corrected without losing source evidence.

## Phase 2 — Provider-based pose, object, and camera analysis

**Outcome:** Independent providers produce the common Tracking IR contract.

Issues:

- Implement the `PoseProvider`, `ObjectProvider`, and `CameraMotionProvider` interfaces.
- Add MediaPipe full-body pose as the default provider: 33 landmarks, bounding box, track lifecycle, and confidence.
- Add automatic object detection/tracking and manual ROI tracking contracts.
- Record occlusion, re-identification, track-ID changes, and unsupported results rather than hiding them.
- Add optical-flow/feature/homography camera estimator that outputs `screenSpace` and `stabilizedSpace` transforms with confidence.
- Register YOLO/RTMPose-class providers as optional integrations with independent licensing/install checks.

Exit evidence:

- Synthetic motion fixtures cover continuous tracks, occlusion, re-entry, and camera-pan transforms.
- All providers either emit a valid track or an explicit failure status with provenance.

Recovery: providers may be disabled independently. No optional provider may block MediaPipe-only analysis.

## Phase 3 — Browser verification and patch review

**Outcome:** A user can see what the system believes before generation.

Issues:

- Build a TypeScript Canvas/SVG overlay viewer with frame-accurate seeking.
- Show source video/frame, pose skeleton, bounding boxes, object paths, camera transform, confidence, and uncertainty flags.
- Provide track actions: approve, exclude, request re-analysis, specify ROI, and manually correct coordinates.
- Store every change as an ordered `edits.patch.json`; implement compare and rollback.
- Implement `auto-review` scoring and escalation according to [validation policy](docs/validation.md).

Exit evidence:

- A manual correction changes `resolved.ir.json` without changing `tracking.ir.json`.
- A failed auto-review item cannot enter a final-render-ready state without an explicit override recorded in the patch.

Recovery: users can discard a patch or roll back to any valid patch revision.

## Phase 4 — Deterministic code generators

**Outcome:** Approved data produces inspectable code, not opaque video effects.

Issues:

- Compile approved tracks into a HyperFrames project with explicit dimensions, duration, clips/tracks, and seek-safe timeline data.
- Compile paths and pose geometry into precise SVG and a second-pass rough/freehand style layer.
- Make coordinate-space selection, profile reframing, and edit effects explicit in generated source.
- Generate source maps from generated elements back to IR track IDs, frame ranges, and patch operations.
- Add a minimal Remotion adapter behind a separate package boundary.

Exit evidence:

- Generated source passes schema/type/lint checks and can be regenerated identically from the same resolved IR.
- Rendered reference frames retain the approved track overlay within configured tolerance.

Recovery: generator templates are versioned and selected through provenance; old resolved IR can always re-run with its recorded generator version.

## Phase 5 — Render verification and output profiles

**Outcome:** Generated projects are checked against their source intent and can be reframed for common social-video layouts.

Issues:

- Implement profile-aware safe areas, subject-aware crop suggestions, and layout constraints.
- Add `source`, YouTube 16:9, YouTube Shorts 9:16, Instagram Reel 9:16, Instagram Feed 4:5, and custom profiles.
- Render deterministic preview frames at review timestamps.
- Add visual regression fixtures for paths, safe areas, reframe transforms, and sketch stroke output.
- Add report output with quality scores, unresolved warnings, asset paths, and approvals.

Exit evidence:

- Profile transformations preserve normalized source-to-output mapping in fixture tests.
- Render comparison catches intended fixture drift and passes approved baselines.

Recovery: visual baseline updates require a recorded reason and an independent review; no blanket snapshot update.

## Phase 6 — Skill packaging and public release readiness

**Outcome:** Codex and Claude can reliably orchestrate the same CLI without vendor lock-in.

Issues:

- Finalize root `SKILL.md` and mode-specific references.
- Add Codex and Claude integration examples that convert natural language to EditSpec before invocation.
- Add quick validation for Skill metadata, CLI help text, and schema compatibility.
- Write security, privacy, media-rights, model-license, and contributor documentation.
- Publish a release checklist and a fixture-only end-to-end smoke test.

Exit evidence:

- A fresh agent can follow the Skill with a synthetic fixture and obtain the expected planning/review actions.
- Codex and Claude examples produce the same valid EditSpec for equivalent fixture instructions.

Recovery: integrations remain documentation/adapters; a vendor-specific failure cannot change the core CLI or data schema.

## Cross-phase review rule

Every phase needs two independent checks before merge:

1. automated contract, type, lint, schema, or render check appropriate to the change;
2. a human or independent-agent review of the changed behavior and its failure path.

Passing a command is not evidence of visual correctness. Changes that affect generated visuals additionally require inspected reference frames or a documented reason why visual inspection is not applicable.
