# open-motion-bridge

> Local-first, Skill-First video-to-code harness for tracked motion graphics and sketch animation.

`open-motion-bridge` turns a locally supplied video into a reviewable motion-data project:

```text
local video
  -> ffmpeg ingest
  -> pose + object + camera analysis
  -> immutable Tracking IR
  -> browser overlay review and patch history
  -> HyperFrames project + SVG sketch project
  -> optional Remotion adapter
```

It is designed for AI agents to use through a general `SKILL.md`, while the actual work remains explicit, local, and reproducible through a CLI. Natural-language direction is converted by the calling agent into a validated `EditSpec`; the CLI never silently interprets or ignores free text.

## Status

The first executable vertical slice is included:

- local ffprobe/OpenCV ingest;
- MediaPipe full-body pose sampling and an opt-in MMPose RTMPose-L WholeBody provider into immutable Tracking IR;
- native-FPS analysis, One Euro temporal smoothing, confidence-aware gap handling, and render-FPS interpolation into a separate render Tracking IR;
- deterministic HyperFrames source generation with a source-video pose overlay;
- exact SVG skeleton trace export.

It is intentionally narrow. Automatic object tracking, camera stabilization, browser patch review, generic EditSpec resolution, Remotion export, and social-platform reframing remain roadmap items. The implemented commands are labelled below; unsupported commands must fail explicitly rather than imply that they worked.

## Product guarantees

- Local video is the default input; no cloud upload is required by the core workflow.
- Original analysis data is immutable. Edits live in a separate patch file.
- Every generated project retains provenance back to its source manifest, IR, and edit patch.
- Manual review is the default. `auto-review` may approve only tracks that pass defined quality gates.
- Real user footage, including local development clips such as `pinball.mp4`, is never committed to this repository.

## Intended outputs

| Target | Purpose |
| --- | --- |
| HyperFrames | Primary HTML-based, deterministic motion-graphics project |
| SVG sketch | Exact tracked paths followed by a rough/freehand styling pass |
| Remotion | Optional adapter for consumers who need a React-based export |

The generator supports `source`, `youtube-16x9`, `youtube-shorts-9x16`, `instagram-reel-9x16`, `instagram-feed-4x5`, and fully custom canvas profiles. Profiles define canvas geometry and safe-area/reframing constraints; they are configurable rather than a claim of current platform certification.

## CLI

```powershell
python -m open_motion_bridge analyze <local-video-path> --output <analysis-dir>
python -m open_motion_bridge generate <analysis-dir>/tracking.ir.json `
  --source-video <local-video-path> --output <hyperframes-project-dir> --target both `
  --render-fps 30 --smoothing-profile balanced --force
```

For high-precision face and hand anchors, use the opt-in MMPose provider with **local** model assets. It never downloads a model or silently falls back to MediaPipe:

```powershell
python -m open_motion_bridge analyze <local-video-path> --output <analysis-dir> `
  --pose-provider mmpose-rtmpose-l-wholebody `
  --mmpose-pose-config <local-rtmpose-wholebody-config.py> `
  --mmpose-pose-weights <local-rtmpose-wholebody-checkpoint.pth> `
  --mmpose-detector-config <local-person-detector-config.py> `
  --mmpose-detector-weights <local-person-detector-checkpoint.pth> `
  --mmpose-device cuda:0
```

Install the Python package first:

```powershell
pip install -e .\packages\analyzer-python
```

MMPose needs a compatible PyTorch/MMCV/MMDetection environment and is deliberately optional. See [MMPose provider setup](docs/mmpose-provider.md) before installing it.

The generator stages a copy of the supplied source into the output project so the rendered project is self-contained. Keep generated projects outside a public repository unless you have redistribution rights to that source media.

If decoded frame PTS extends past the container's declared A/V streams, the Tracking IR preserves the complete timing evidence while the generated composition uses the shortest declared A/V duration. This prevents a silent or frozen tail in the rendered deliverable.

`analyze` samples every decodable source frame by default (`--sample-fps 0`). `generate` never alters that raw IR: it writes `render.tracking.ir.json` with configurable One Euro smoothing, short-gap holds, long-gap hiding, and interpolation at the requested render FPS. Use `responsive`, `balanced` (default), or `stable` according to the needed motion response.

For a natural-language request, a Codex/Claude integration will first write and validate an `EditSpec`, show the resolved request to the user where required, then call the future deterministic patch/resolution commands. The core CLI has no hidden LLM dependency and does not treat arbitrary text as executable instruction.

## Repository layout

```text
packages/
  analyzer-python/          # planned Python ingest and analysis core
  verifier-web/             # planned TypeScript overlay/review UI
  generator-hyperframes/    # planned TypeScript HyperFrames compiler
  generator-sketch-svg/     # planned TypeScript SVG compiler
  export-remotion/          # planned optional adapter
  shared-schema/            # planned generated schema/types package
docs/                       # contracts, validation, risk policy
fixtures/                   # synthetic, non-identifying test inputs only
integrations/               # Codex and Claude integration guides
```

## Local media policy

Put development video outside the repository or in an ignored local path. `pinball.mp4` is intentionally ignored. CI uses synthetic IR and SVG fixtures, never a real-person sample video.

Before analyzing, distributing, or rendering footage, ensure that you have the rights and permissions to use the video and everyone identifiable in it. See [risks and licenses](docs/risks-and-licenses.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Tracking IR schema](docs/ir-schema.md)
- [Natural-language EditSpec contract](docs/edit-spec.md)
- [Validation and review policy](docs/validation.md)
- [MMPose provider setup](docs/mmpose-provider.md)
- [Risks and licensing](docs/risks-and-licenses.md)
- [Implementation roadmap](ROADMAP.md)
- [Agent entry point](SKILL.md)

## License

[MIT](LICENSE). The repository license does not automatically apply to optional model weights, model runtimes, third-party media, or platform SDKs.
