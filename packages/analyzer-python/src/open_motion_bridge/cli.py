from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import MMPoseOptions, analyze_video, generate_projects, verify_render


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omb",
        description="Local MediaPipe pose analysis and HyperFrames/SVG trace generation.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    analyze = subcommands.add_parser("analyze", help="Create immutable source manifest and Tracking IR.")
    analyze.add_argument("video", type=_path)
    analyze.add_argument("--output", type=_path, required=True)
    analyze.add_argument(
        "--sample-fps",
        type=float,
        default=0.0,
        help="Analysis FPS; use 0 (default) to preserve every decodable source frame.",
    )
    analyze.add_argument(
        "--pose-provider",
        choices=("mediapipe", "mmpose-rtmpose-l-wholebody"),
        default="mediapipe",
        help="Pose provider. MMPose requires explicit local config and checkpoint files.",
    )
    analyze.add_argument("--mmpose-pose-config", type=_path, help="Local RTMPose-L WholeBody config path.")
    analyze.add_argument("--mmpose-pose-weights", type=_path, help="Local RTMPose-L WholeBody checkpoint path.")
    analyze.add_argument("--mmpose-detector-config", type=_path, help="Local person-detector config path.")
    analyze.add_argument("--mmpose-detector-weights", type=_path, help="Local person-detector checkpoint path.")
    analyze.add_argument("--mmpose-device", default="cuda:0", help="MMPose device, for example cuda:0 or cpu.")
    analyze.add_argument("--force", action="store_true")

    generate = subcommands.add_parser(
        "generate", help="Generate a HyperFrames project and/or SVG skeleton trace from Tracking IR."
    )
    generate.add_argument("ir", type=_path)
    generate.add_argument("--source-video", type=_path, required=True)
    generate.add_argument("--output", type=_path, required=True)
    generate.add_argument("--target", choices=("hyperframes", "sketch-svg", "both"), default="both")
    generate.add_argument("--profile", default="source", choices=("source", "youtube-shorts-9x16", "instagram-reel-9x16"))
    generate.add_argument("--render-fps", type=float, default=30.0)
    generate.add_argument("--smoothing-profile", choices=("responsive", "balanced", "stable"), default="balanced")
    generate.add_argument("--visibility-threshold", type=float, default=0.2)
    generate.add_argument("--max-gap-ms", type=float, default=250.0)
    generate.add_argument(
        "--edit-spec",
        type=_path,
        help="Declarative asset binding spec. Never mutates the analysis IR.",
    )
    generate.add_argument(
        "--overlay",
        choices=("skeleton", "bindings", "both"),
        default="skeleton",
        help="What the composition draws: tracking skeleton, approved bindings, or both for review.",
    )
    generate.add_argument("--force", action="store_true")

    verify = subcommands.add_parser(
        "verify",
        help="Measure where bindings actually landed in a rendered file against their resolved coordinates.",
    )
    verify.add_argument("project", type=_path, help="Generated HyperFrames project directory.")
    verify.add_argument("--rendered-video", type=_path, required=True)
    verify.add_argument("--output", type=_path, required=True)
    verify.add_argument("--samples", type=int, default=8)
    verify.add_argument("--tolerance-px", type=float, default=24.0)
    verify.add_argument("--force", action="store_true")

    analyze_image = subcommands.add_parser(
        "analyze-image", help="Create a single-observation Tracking IR from a still photo."
    )
    analyze_image.add_argument("image", type=_path)
    analyze_image.add_argument("--output", type=_path, required=True)
    analyze_image.add_argument("--force", action="store_true")

    generate_photo = subcommands.add_parser(
        "generate-photo",
        help="Generate a HyperFrames motion-graphics project from a photo IR and a MotionSpec.",
    )
    generate_photo.add_argument("ir", type=_path)
    generate_photo.add_argument("--source-image", type=_path, required=True)
    generate_photo.add_argument("--motion-spec", type=_path, required=True)
    generate_photo.add_argument("--output", type=_path, required=True)
    generate_photo.add_argument("--force", action="store_true")

    sketch_image = subcommands.add_parser(
        "sketch-image",
        help="Vectorize a photo into ordered strokes and generate a self-drawing sketch project.",
    )
    sketch_image.add_argument("image", type=_path)
    sketch_image.add_argument("--output", type=_path, required=True)
    sketch_image.add_argument("--duration-ms", type=float, default=10000.0)
    sketch_image.add_argument("--fps", type=float, default=30.0)
    sketch_image.add_argument("--hold-ms", type=float, default=2000.0)
    sketch_image.add_argument("--photo-fade-ms", type=float, default=1500.0)
    sketch_image.add_argument("--max-strokes", type=int, default=900)
    sketch_image.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        mmpose_options = None
        if args.pose_provider == "mmpose-rtmpose-l-wholebody":
            required_assets = {
                "--mmpose-pose-config": args.mmpose_pose_config,
                "--mmpose-pose-weights": args.mmpose_pose_weights,
                "--mmpose-detector-config": args.mmpose_detector_config,
                "--mmpose-detector-weights": args.mmpose_detector_weights,
            }
            missing_assets = [flag for flag, path in required_assets.items() if path is None]
            if missing_assets:
                raise ValueError(
                    "MMPose provider requires explicit local assets: " + ", ".join(missing_assets)
                )
            mmpose_options = MMPoseOptions(
                pose_config=args.mmpose_pose_config,
                pose_weights=args.mmpose_pose_weights,
                detector_config=args.mmpose_detector_config,
                detector_weights=args.mmpose_detector_weights,
                device=args.mmpose_device,
            )
        analyze_video(
            args.video,
            args.output,
            sample_fps=args.sample_fps,
            force=args.force,
            pose_provider=args.pose_provider,
            mmpose_options=mmpose_options,
        )
        return 0
    if args.command == "generate":
        generate_projects(
            args.ir,
            args.source_video,
            args.output,
            args.target,
            args.profile,
            args.force,
            render_fps=args.render_fps,
            smoothing_profile=args.smoothing_profile,
            visibility_threshold=args.visibility_threshold,
            max_gap_ms=args.max_gap_ms,
            edit_spec=args.edit_spec,
            overlay=args.overlay,
        )
        return 0
    if args.command == "verify":
        report = verify_render(
            args.project,
            args.rendered_video,
            args.output,
            samples=args.samples,
            tolerance_px=args.tolerance_px,
            force=args.force,
        )
        return 0 if report["summary"]["passed"] else 1
    if args.command == "analyze-image":
        from .photomotion import analyze_image as run_analyze_image

        run_analyze_image(args.image, args.output, force=args.force)
        return 0
    if args.command == "generate-photo":
        from .photomotion import generate_photo_project

        generate_photo_project(
            args.ir, args.source_image, args.motion_spec, args.output, force=args.force
        )
        return 0
    if args.command == "sketch-image":
        from .sketchdraw import generate_sketch_project

        plan = generate_sketch_project(
            args.image,
            args.output,
            duration_ms=args.duration_ms,
            fps=args.fps,
            hold_ms=args.hold_ms,
            photo_fade_ms=args.photo_fade_ms,
            max_strokes=args.max_strokes,
            force=args.force,
        )
        print(f"strokes={plan['vectorization']['strokeCount']} inkPx={plan['vectorization']['totalInkPx']}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")
