from __future__ import annotations

import hashlib
import html
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

SCHEMA_VERSION = "0.1.0"
POSE_CONNECTIONS = tuple((int(a), int(b)) for a, b in mp.solutions.pose.POSE_CONNECTIONS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _run_ffprobe(video: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,duration,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def _reliable_playback_duration_ms(probe: dict[str, Any], fallback_ms: float) -> tuple[float, str]:
    """Choose a render duration that cannot outlive a declared A/V stream.

    Frame PTS can legitimately extend beyond a container's declared duration. The
    analysis IR keeps that evidence, while the generated composition stops at
    the shortest declared audio/video stream so it cannot add a silent or frozen
    tail to an otherwise synchronized source.
    """
    stream_durations: dict[str, float] = {}
    for stream in probe.get("streams", []):
        kind = stream.get("codec_type")
        if kind not in {"audio", "video"} or kind in stream_durations:
            continue
        try:
            duration_ms = float(stream.get("duration", 0.0)) * 1000.0
        except (TypeError, ValueError):
            continue
        if duration_ms > 0:
            stream_durations[kind] = duration_ms

    if stream_durations:
        return min(stream_durations.values()), "shortest-declared-av-stream"
    return fallback_ms, "container-duration-fallback"


def _frame_timestamps(video: Path) -> list[float]:
    """Read presentation timestamps from ffprobe instead of trusting OpenCV clock state."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    timestamps: list[float] = []
    for raw_line in completed.stdout.splitlines():
        candidate = raw_line.strip().split(",", 1)[0]
        try:
            timestamps.append(float(candidate))
        except ValueError:
            continue
    if not timestamps:
        raise RuntimeError(f"ffprobe returned no video frame timestamps: {video}")
    return timestamps


def _write_json(path: Path, value: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}. Use --force for generated output.")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bbox(landmarks: list[dict[str, Any]]) -> dict[str, float]:
    visible = [point for point in landmarks if point["visibility"] >= 0.2]
    points = visible or landmarks
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return {
        "x": max(0.0, min(xs)),
        "y": max(0.0, min(ys)),
        "width": min(1.0, max(xs)) - max(0.0, min(xs)),
        "height": min(1.0, max(ys)) - max(0.0, min(ys)),
    }


def analyze_video(video: Path, output_dir: Path, sample_fps: float, force: bool) -> None:
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than zero")
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = _run_ffprobe(video)
    frame_timestamps = _frame_timestamps(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not decode: {video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or sample_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(source_fps / sample_fps))

    frames: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    detection_count = 0
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step:
                index += 1
                continue
            timestamp_ms = (
                frame_timestamps[index] * 1000.0
                if index < len(frame_timestamps)
                else index / source_fps * 1000.0
            )
            frames.append(
                {
                    "frameIndex": index,
                    "sourceTimeMs": round(timestamp_ms, 3),
                    "decodeStatus": "decoded",
                }
            )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                landmarks = [
                    {
                        "name": mp.solutions.pose.PoseLandmark(number).name.lower(),
                        "x": round(float(item.x), 6),
                        "y": round(float(item.y), 6),
                        "z": round(float(item.z), 6),
                        "visibility": round(float(item.visibility), 6),
                    }
                    for number, item in enumerate(result.pose_landmarks.landmark)
                ]
                mean_visibility = sum(item["visibility"] for item in landmarks) / len(landmarks)
                observations.append(
                    {
                        "frameIndex": index,
                        "sourceTimeMs": round(timestamp_ms, 3),
                        "confidence": round(mean_visibility, 6),
                        "occlusion": "none" if mean_visibility >= 0.5 else "partial",
                        "screenSpace": {"bbox": _bbox(landmarks), "keypoints": landmarks},
                        "quality": {
                            "interpolated": False,
                            "driftWarning": False,
                            "manualCorrectionRequired": mean_visibility < 0.5,
                        },
                    }
                )
                detection_count += 1
            index += 1
    finally:
        pose.close()
        capture.release()

    container_duration_ms = 0.0
    try:
        container_duration_ms = float(probe.get("format", {}).get("duration", 0.0)) * 1000.0
    except (TypeError, ValueError):
        container_duration_ms = 0.0
    frame_interval_ms = 1000.0 / source_fps
    frame_pts_duration_ms = (frame_timestamps[-1] * 1000.0) + frame_interval_ms
    duration_ms = max(container_duration_ms, frame_pts_duration_ms)
    render_duration_ms, render_duration_source = _reliable_playback_duration_ms(probe, container_duration_ms or duration_ms)

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": _utc_now(),
        "source": {
            "sourceHash": _sha256(video),
            "fileName": video.name,
            "displayWidth": width,
            "displayHeight": height,
            "nominalFps": round(source_fps, 6),
            "durationMs": round(duration_ms, 3),
            "renderDurationMs": round(render_duration_ms, 3),
            "frameCountEstimate": frame_count,
            "ffprobe": probe,
            "timingEvidence": {
                "source": "ffprobe-best_effort_timestamp_time",
                "framePtsCount": len(frame_timestamps),
                "containerDurationMs": round(container_duration_ms, 3),
                "framePtsDurationMs": round(frame_pts_duration_ms, 3),
                "renderDurationSource": render_duration_source,
            },
        },
        "sampling": {"requestedFps": sample_fps, "sourceFrameStep": step, "sampledFrames": len(frames)},
    }
    continuity = detection_count / len(frames) if frames else 0.0
    ir = {
        "schemaVersion": SCHEMA_VERSION,
        "source": manifest["source"],
        "coordinateSystem": {
            "unit": "normalized",
            "screenSpace": "x-right-y-down, [0,1] relative to display orientation",
            "stabilizedSpace": "not-produced-in-pose-only-provider",
        },
        "frames": frames,
        "tracks": [
            {
                "id": "person-001",
                "type": "pose",
                "provider": {
                    "name": "mediapipe-pose",
                    "version": getattr(mp, "__version__", "unknown"),
                    "modelIdentifier": "Pose(model_complexity=1)",
                    "licenseHint": "Verify MediaPipe and model distribution terms before release.",
                },
                "lifecycle": {
                    "firstFrame": observations[0]["frameIndex"] if observations else None,
                    "lastFrame": observations[-1]["frameIndex"] if observations else None,
                    "reidentifiedFrom": [],
                    "idChanges": [],
                },
                "observations": observations,
            }
        ],
        "cameraMotion": [],
        "analysis": {
            "status": "completed" if observations else "completed-with-no-pose-detections",
            "sampledFrames": len(frames),
            "poseDetections": detection_count,
            "continuity": round(continuity, 6),
            "limitations": [
                "Object tracking and camera stabilization are not implemented in this vertical slice.",
                "stabilizedSpace is intentionally unavailable rather than fabricated.",
            ],
        },
        "provenance": [
            {
                "kind": "analysis",
                "createdAt": _utc_now(),
                "tool": "open-motion-bridge",
                "toolVersion": SCHEMA_VERSION,
                "sourceHash": manifest["source"]["sourceHash"],
            }
        ],
    }
    _write_json(output_dir / "source-manifest.json", manifest, force)
    _write_json(output_dir / "tracking.ir.json", ir, force)


def _coordinate(value: float, pixels: int) -> str:
    return f"{value * pixels:.2f}"


def _skeleton_markup(observation: dict[str, Any], width: int, height: int) -> str:
    points = observation["screenSpace"]["keypoints"]
    by_index = {index: point for index, point in enumerate(points)}
    lines = []
    for start, end in POSE_CONNECTIONS:
        a, b = by_index.get(start), by_index.get(end)
        if not a or not b or min(a["visibility"], b["visibility"]) < 0.2:
            continue
        lines.append(
            '<line x1="{}" y1="{}" x2="{}" y2="{}" />'.format(
                _coordinate(a["x"], width),
                _coordinate(a["y"], height),
                _coordinate(b["x"], width),
                _coordinate(b["y"], height),
            )
        )
    dots = [
        '<circle cx="{}" cy="{}" r="5" />'.format(_coordinate(point["x"], width), _coordinate(point["y"], height))
        for point in points
        if point["visibility"] >= 0.35
    ]
    return "".join(lines + dots)


def _render_html(ir: dict[str, Any], pose_frames: list[dict[str, Any]], profile: str) -> str:
    source = ir["source"]
    width, height = int(source["displayWidth"]), int(source["displayHeight"])
    duration = max(0.1, float(source.get("renderDurationMs", source["durationMs"])) / 1000.0)
    if profile in {"youtube-shorts-9x16", "instagram-reel-9x16"}:
        width, height = 1080, 1920
    pose_json = json.dumps(pose_frames, ensure_ascii=False, separators=(",", ":"))
    connections_json = json.dumps(POSE_CONNECTIONS, separators=(",", ":"))
    return f'''<!doctype html>
<html lang="en" data-resolution="portrait">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={width}, height={height}" />
    <title>Open Motion Bridge pose trace</title>
    <script src="./node_modules/gsap/dist/gsap.min.js"></script>
    <style>
      * {{ box-sizing: border-box; }}
      html, body {{ margin: 0; width: {width}px; height: {height}px; overflow: hidden; background: #17130f; }}
      #root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; font-family: Arial, sans-serif; }}
      .clip {{ position: absolute; inset: 0; }}
      #source-video {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }}
      #pose-canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }}
      #hud-backplate {{ position: absolute; top: 42px; left: 36px; width: 720px; height: 148px; background: rgba(23, 19, 15, 0.82); border-left: 6px solid #ffb45b; }}
      #hud-inner {{ position: absolute; top: 64px; left: 64px; color: #fff9ed; }}
      #hud-label {{ font-size: 24px; font-weight: 700; letter-spacing: 0.16em; }}
      #hud-detail {{ margin-top: 10px; font-size: 18px; opacity: 0.88; letter-spacing: 0.04em; }}
      #corner {{ position: absolute; right: 56px; bottom: 62px; width: 112px; height: 112px; border-right: 4px solid #ffb45b; border-bottom: 4px solid #ffb45b; opacity: 0.9; }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{duration:.3f}" data-width="{width}" data-height="{height}" data-fps="30">
      <video id="source-video" src="assets/source.mp4" muted playsinline data-start="0" data-duration="{duration:.3f}" data-track-index="1"></video>
      <audio id="source-audio" src="assets/source.mp4" data-start="0" data-track-index="2" data-volume="1"></audio>
      <canvas id="pose-canvas" class="clip" data-start="0" data-duration="{duration:.3f}" data-track-index="3" width="{source["displayWidth"]}" height="{source["displayHeight"]}"></canvas>
      <section id="hud" class="clip" data-start="0" data-duration="2.1" data-track-index="4">
        <div id="hud-backplate"></div>
        <div id="hud-inner"><div id="hud-label">POSE TRACE</div><div id="hud-detail">local MediaPipe → Tracking IR → HyperFrames</div></div>
        <div id="corner"></div>
      </section>
    </div>
    <script>
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true }});
      const poseFrames = {pose_json};
      const poseConnections = {connections_json};
      const poseCanvas = document.getElementById('pose-canvas');
      const poseContext = poseCanvas.getContext('2d');
      const drawPose = (time) => {{
        poseContext.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
        let low = 0;
        let high = poseFrames.length - 1;
        while (low < high) {{
          const middle = Math.ceil((low + high) / 2);
          if (poseFrames[middle].time <= time) low = middle; else high = middle - 1;
        }}
        const frame = poseFrames[low];
        if (!frame) return;
        poseContext.lineCap = 'round';
        poseContext.lineJoin = 'round';
        poseContext.strokeStyle = '#ffb45b';
        poseContext.lineWidth = 9;
        for (const [from, to] of poseConnections) {{
          const a = frame.points[from];
          const b = frame.points[to];
          if (!a || !b || a[2] < 0.2 || b[2] < 0.2) continue;
          poseContext.beginPath();
          poseContext.moveTo(a[0] * poseCanvas.width, a[1] * poseCanvas.height);
          poseContext.lineTo(b[0] * poseCanvas.width, b[1] * poseCanvas.height);
          poseContext.stroke();
        }}
        for (const point of frame.points) {{
          if (point[2] < 0.35) continue;
          poseContext.beginPath();
          poseContext.arc(point[0] * poseCanvas.width, point[1] * poseCanvas.height, 5, 0, Math.PI * 2);
          poseContext.fillStyle = '#fff2d9';
          poseContext.fill();
          poseContext.lineWidth = 3;
          poseContext.strokeStyle = '#ff8d3d';
          poseContext.stroke();
        }}
      }};
      const poseState = {{ time: 0 }};
      drawPose(0);
      tl.to(poseState, {{ time: {duration:.3f}, duration: {duration:.3f}, ease: 'none', onUpdate: () => drawPose(poseState.time) }}, 0);
      tl.fromTo('#hud-inner', {{ opacity: 0, y: -24 }}, {{ opacity: 1, y: 0, duration: 0.45, ease: 'power3.out' }}, 0.12);
      tl.to('#hud-inner', {{ opacity: 0, duration: 0.35, ease: 'power2.in' }}, 1.65);
      tl.set('#hud-inner', {{ opacity: 0 }}, 1.95);
      tl.fromTo('#corner', {{ opacity: 0, scale: 0.8 }}, {{ opacity: 0.9, scale: 1, duration: 0.45, ease: 'power3.out' }}, 0.28);
      window.__timelines['main'] = tl;
    </script>
  </body>
</html>
'''


def _render_svg(ir: dict[str, Any], skeletons: list[tuple[float, float, str]]) -> str:
    source = ir["source"]
    width, height = int(source["displayWidth"]), int(source["displayHeight"])
    groups = "".join(
        f'<g id="frame-{index}" data-start="{start:.3f}" data-duration="{duration:.4f}">{markup}</g>'
        for index, (start, duration, markup) in enumerate(skeletons)
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <metadata>Open Motion Bridge exact pose trace; source observations remain in tracking.ir.json.</metadata>
  <g fill="#fff2d9" stroke="#ff8d3d" stroke-linecap="round" stroke-linejoin="round">{groups}</g>
</svg>
'''


def generate_projects(ir_path: Path, source_video: Path, output_dir: Path, target: str, profile: str, force: bool) -> None:
    if not ir_path.is_file():
        raise FileNotFoundError(f"Tracking IR not found: {ir_path}")
    if not source_video.is_file():
        raise FileNotFoundError(f"Source video not found: {source_video}")
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    if ir.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported IR schema: {ir.get('schemaVersion')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = ir.get("tracks", [{}])[0].get("observations", [])
    total_duration = float(ir["source"]["durationMs"]) / 1000.0
    skeletons: list[tuple[float, float, str]] = []
    for index, observation in enumerate(observations):
        start = round(float(observation["sourceTimeMs"]) / 1000.0, 3)
        next_time = float(observations[index + 1]["sourceTimeMs"]) / 1000.0 if index + 1 < len(observations) else total_duration
        next_start = round(next_time, 3)
        skeletons.append((start, max(0.001, next_start - start), _skeleton_markup(observation, int(ir["source"]["displayWidth"]), int(ir["source"]["displayHeight"]))))
    if not skeletons:
        raise RuntimeError("No pose observations were produced; refusing to generate an unverified overlay project.")
    if target in {"hyperframes", "both"}:
        html_path = output_dir / "index.html"
        if html_path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {html_path}; use --force for generated output.")
        assets = output_dir / "assets"
        assets.mkdir(exist_ok=True)
        staged_video = assets / "source.mp4"
        if staged_video.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {staged_video}; use --force for generated output.")
        shutil.copy2(source_video, staged_video)
        if force:
            compositions_dir = output_dir / "compositions"
            if compositions_dir.exists():
                for stale_file in compositions_dir.glob("pose-chunk-*.html"):
                    stale_file.unlink()
        pose_frames = [
            {
                "time": round(float(observation["sourceTimeMs"]) / 1000.0, 6),
                "points": [
                    [round(float(point["x"]), 6), round(float(point["y"]), 6), round(float(point["visibility"]), 6)]
                    for point in observation["screenSpace"]["keypoints"]
                ],
            }
            for observation in observations
        ]
        html_path.write_text(_render_html(ir, pose_frames, profile), encoding="utf-8")
        _write_json(
            output_dir / "open-motion-bridge.generated.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "createdAt": _utc_now(),
                "sourceIr": str(ir_path.name),
                "sourceHash": ir["source"]["sourceHash"],
                "profile": profile,
                "target": "hyperframes",
                "trackId": "person-001",
                "observationCount": len(skeletons),
                "compositionDurationMs": round(float(ir["source"].get("renderDurationMs", ir["source"]["durationMs"])), 3),
            },
            force,
        )
    if target in {"sketch-svg", "both"}:
        svg_path = output_dir / "sketch-pose-trace.svg"
        if svg_path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {svg_path}; use --force for generated output.")
        svg_path.write_text(_render_svg(ir, skeletons), encoding="utf-8")
