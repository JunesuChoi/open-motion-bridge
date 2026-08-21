# Validation and review policy

## Two gates, not one

Every run has a data-quality gate and a render-quality gate.

1. **Data-quality gate:** verifies the source manifest, IR schemas, tracking continuity, coordinates, camera estimate, and patch semantics.
2. **Render-quality gate:** verifies generated source, deterministic render settings, profile mapping, and visual agreement at selected timestamps.

A passing tracker does not prove a correct render. A successful render does not prove a correct track.

## Manual review

The verification UI displays the original frame/video beside or over:

- pose skeleton and landmark confidence;
- object bbox/path/ROI;
- screen-space and stabilized-space positions;
- camera transform confidence;
- profile crop and safe area;
- HyperFrames and SVG preview reference frames.

For each track or time range, a reviewer can approve, exclude, request re-analysis, create a ROI, or add a manual coordinate correction. All changes are patch operations with an undoable revision.

## Auto-review eligibility

The default policy recommends auto-approval only when all conditions hold:

| Metric | Threshold |
| --- | --- |
| Average track confidence | `>= 0.80` |
| Track continuity | `>= 0.95` |
| Coordinate drift | no warning |
| Occlusion | none in selected range |
| Track ID change | none in selected range |
| Camera estimate, if used | `>= 0.75` |

Anything else is `needs-review`. An explicit human override may proceed, but must state why in the patch and report. The first version must not treat an override as a new automatic baseline.

## Failure handling

| Signal | System behavior | Review path |
| --- | --- | --- |
| fps/timestamp mismatch | stop profile/render generation; retain raw timing evidence | re-ingest with corrected timestamp adapter |
| coordinate drift | flag affected range; do not auto-approve | re-analyze, choose another provider, or patch manually |
| track ID change | split lifecycle and flag re-identification | merge only through explicit reviewer operation |
| occlusion | mark missing/occluded observations; do not fabricate certainty | exclude, interpolate with labelled confidence, or re-track |
| motion blur | reduce confidence and surface provider evidence | sample differently or request manual correction |
| camera estimate failure | retain screen-space data; disable stabilized actions | use source-following effects or review manually |
| unavailable model/license | do not fall back invisibly | report missing provider and choose an available approved provider |

For the optional MMPose provider, require local pose and detector configs/checkpoints, 133 named WholeBody landmarks, and a recorded provider/version/device. A missing runtime, missing model asset, invalid model output, or uncertain primary-subject association is a visible failure or review signal; it must never silently become a MediaPipe analysis.

## Temporal-resolution policy

- Keep the provider output in `tracking.ir.json` unchanged.
- Apply smoothing only in a derived render IR, recording its profile, cutoff values, requested render FPS, visibility threshold, and gap policy.
- Interpolate only across a configured short gap with available endpoint observations.
- Hide a landmark after the allowed gap and mark it for manual correction instead of extrapolating a confident position.
- Validate both raw continuity and the render IR's interpolation/hidden-gap counts before automatic approval.

## Render checks

For each generated target:

1. Validate resolved IR, EditSpec, profile, and generator input schemas.
2. Run TypeScript/Python checks relevant to the generated package.
3. Run target-specific lint/check. For HyperFrames, this includes the framework validation command once the generator exists.
4. Render or snapshot agreed review timestamps: start, every edit boundary, middle of each active track, and end.
5. Overlay approved coordinates over the generated frame using source maps.
6. Compare with a versioned baseline for synthetic fixtures; inspect differences rather than accepting all updates.
7. Require human approval before calling a production render complete.

## Acceptance criteria for an implementation change

- Schema/type/lint tests pass.
- A relevant fixture covers the normal path and at least one failure path.
- A second independent review examines the behavior and its failure handling.
- Any changed output profile or generator checks visual references.
- No test artifact contains real-person media unless separately approved and licensed.
