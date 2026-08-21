# Risks and licensing

## Repository license boundary

The repository uses MIT. This covers repository-authored code and documentation only. It does not grant rights to:

- input video, audio, logos, or images;
- people appearing in video;
- third-party model weights or checkpoints;
- third-party detection/tracking packages with their own licenses;
- commercial platform SDKs or hosted inference services.

## Provider policy

MediaPipe is the default pose-provider direction because the architecture must remain practical for an MIT repository. This is not legal clearance for a particular version, runtime, model asset, or downstream use. Each release must record the exact provider/version/license notice it ships or instructs users to download.

YOLO-, RTMPose-, or other model-backed integrations are optional provider packages. They must:

1. be installable separately from the core;
2. declare package, model, weight, and runtime licenses;
3. not be required for fixture CI or basic MediaPipe workflow;
4. surface their terms before a user enables them;
5. avoid bundling restricted weights without explicit distribution permission.

## Media privacy and rights

- Core processing is local by default.
- The repository ignores local media, model caches, output folders, and secret files.
- Do not commit `pinball.mp4`, any actual source video, extracted frames, faces, or identifying tracking data.
- A contributor who adds media must document permission, source, allowed redistribution, retention, and removal path.
- The project must not state that it can determine consent, ownership, or a person's identity from a video.

## Technical risks

| Risk | Mitigation |
| --- | --- |
| Variable frame rate | Preserve presentation timestamps and test VFR manifests |
| Camera parallax / insufficient features | Record low confidence; keep screen-space output usable |
| Occlusion and re-identification | Record lifecycle events; require review rather than concealing gaps |
| GPU/CPU variability | Provider capability checks, deterministic fixture mode, clear fallback status |
| ffmpeg installation variance | Preflight discovers/report binary and version; document local install steps |
| Generated visual drift | Source maps, reference snapshots, overlay comparison, approval gate |
| Ambiguous natural language | EditSpec preview, semantic validation, explicit unsupported-request status |
| Profile cropping hides a subject | Safe-area visualization and tracked-subject reframe review |

## Release checklist

- Confirm no ignored media has been force-added.
- Run a repository scan for video/image files beyond approved synthetic fixtures.
- Verify notices for all optional providers exposed in the release.
- Run fixture-only CI and the documented cross-platform smoke tests.
- Confirm documentation distinguishes planned, supported, optional, and experimental functionality.
