from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import analyze_video, generate_projects


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
    generate.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        analyze_video(args.video, args.output, sample_fps=args.sample_fps, force=args.force)
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
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")
