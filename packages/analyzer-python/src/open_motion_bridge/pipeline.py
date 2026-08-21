from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import html
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from .bindings import Binding, load_edit_spec, resolve_bindings

SCHEMA_VERSION = "0.1.0"

# Overlay modes decide what the generated composition draws on top of the source frame.
# "skeleton" keeps the diagnostic tracking view, "bindings" shows only approved attached
# assets, and "both" is the review view that proves assets sit on the tracked landmarks.
OVERLAY_MODES = ("skeleton", "bindings", "both")

# On Windows, console child processes (ffmpeg/ffprobe) otherwise flash a visible
# terminal window per invocation; verify extracts many frames, so dozens of windows
# would steal foreground focus. All subprocess calls must go through _run_quiet.
_SUBPROCESS_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _run_quiet(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run a console tool without spawning a visible window or stealing focus."""
    kwargs.setdefault("creationflags", _SUBPROCESS_NO_WINDOW)
    return subprocess.run(command, **kwargs)

# Render verification thresholds. A binding sample is only measurable at full opacity,
# and a template match below the minimum score is reported as unreliable instead of
# being converted into a false pass/fail distance.
_VERIFY_MIN_OPACITY = 0.999
_VERIFY_MIN_MATCH_SCORE = 0.55

POSE_CONNECTIONS = tuple((int(a), int(b)) for a, b in mp.solutions.pose.POSE_CONNECTIONS)
POSE_CONNECTION_NAMES = tuple(
    (
        mp.solutions.pose.PoseLandmark(start).name.lower(),
        mp.solutions.pose.PoseLandmark(end).name.lower(),
    )
    for start, end in POSE_CONNECTIONS
)

# COCO-WholeBody stores 23 body/foot, 68 face, and 21 keypoints per hand.
# Names deliberately stay provider-neutral so render and patch consumers can anchor
# graphics by landmark name instead of an opaque provider-specific array index.
COCO_WHOLEBODY_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    *(f"face_{index:02d}" for index in range(68)),
    *(f"left_hand_{index:02d}" for index in range(21)),
    *(f"right_hand_{index:02d}" for index in range(21)),
)


@dataclass(frozen=True)
class TemporalSmoothingConfig:
    """Deterministic, renderer-facing temporal processing settings."""

    profile: str
    render_fps: float
    min_cutoff: float
    beta: float
    derivative_cutoff: float
    visibility_threshold: float
    max_gap_ms: float


@dataclass(frozen=True)
class MMPoseOptions:
    """Explicit local assets for the opt-in RTMPose-L WholeBody provider."""

    pose_config: Path
    pose_weights: Path
    detector_config: Path
    detector_weights: Path
    device: str


_SMOOTHING_PROFILES: dict[str, tuple[float, float]] = {
    "responsive": (1.4, 0.08),
    "balanced": (1.0, 0.035),
    "stable": (0.7, 0.015),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _temporal_smoothing_config(
    profile: str,
    render_fps: float,
    visibility_threshold: float,
    max_gap_ms: float,
) -> TemporalSmoothingConfig:
    if profile not in _SMOOTHING_PROFILES:
        raise ValueError(f"Unknown smoothing profile: {profile}")
    if render_fps <= 0:
        raise ValueError("--render-fps must be greater than zero")
    if not 0 <= visibility_threshold <= 1:
        raise ValueError("--visibility-threshold must be between 0 and 1")
    if max_gap_ms <= 0:
        raise ValueError("--max-gap-ms must be greater than zero")
    min_cutoff, beta = _SMOOTHING_PROFILES[profile]
    return TemporalSmoothingConfig(
        profile=profile,
        render_fps=render_fps,
        min_cutoff=min_cutoff,
        beta=beta,
        derivative_cutoff=1.0,
        visibility_threshold=visibility_threshold,
        max_gap_ms=max_gap_ms,
    )


def _sample_step(source_fps: float, requested_fps: float) -> int:
    """Return a deterministic decode step; zero requests every source frame."""
    if requested_fps < 0:
        raise ValueError("--sample-fps must be zero (native) or greater")
    if requested_fps == 0:
        return 1
    return max(1, round(source_fps / requested_fps))


def _bounded_score(value: Any) -> float:
    """Convert provider confidence to the Tracking IR's normalized visibility range."""
    return max(0.0, min(1.0, float(value)))


def _as_python_sequence(value: Any) -> list[Any]:
    """Normalize lists and NumPy-like values without adding NumPy as a hard dependency."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("MMPose returned a non-sequence keypoint payload.")
    return list(value)


def _unwrap_single_batch(value: Any) -> list[Any]:
    """MMPose may return a singleton batch dimension for one input frame."""
    result = _as_python_sequence(value)
    if len(result) == 1 and isinstance(result[0], (list, tuple)):
        return _as_python_sequence(result[0])
    return result


def _mmpose_landmarks(
    raw_keypoints: Any,
    raw_scores: Any,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Map an MMPose COCO-WholeBody prediction to canonical named screen-space points."""
    keypoints = _unwrap_single_batch(raw_keypoints)
    scores = _unwrap_single_batch(raw_scores)
    expected_count = len(COCO_WHOLEBODY_KEYPOINT_NAMES)
    if len(keypoints) != expected_count or len(scores) != expected_count:
        raise RuntimeError(
            "Expected exactly 133 COCO-WholeBody keypoints from the MMPose provider; "
            f"received keypoints={len(keypoints)}, scores={len(scores)}."
        )
    landmarks: list[dict[str, Any]] = []
    for name, coordinates, score in zip(COCO_WHOLEBODY_KEYPOINT_NAMES, keypoints, scores, strict=True):
        xy = _as_python_sequence(coordinates)
        if len(xy) < 2:
            raise RuntimeError(f"MMPose keypoint {name!r} does not contain x/y coordinates.")
        landmarks.append(
            {
                "name": name,
                "x": round(float(xy[0]) / width, 6),
                "y": round(float(xy[1]) / height, 6),
                "z": 0.0,
                "visibility": round(_bounded_score(score), 6),
            }
        )
    return landmarks


def _mmpose_candidates(result: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    """Extract all frame candidates while keeping raw provider ordering out of the IR contract."""
    predictions = result.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise RuntimeError("MMPose returned an unexpected prediction batch for one input frame.")
    instances = predictions[0]
    if not isinstance(instances, list):
        raise RuntimeError("MMPose returned an invalid per-frame instance list.")
    candidates: list[dict[str, Any]] = []
    for provider_index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise RuntimeError("MMPose returned a non-object instance prediction.")
        landmarks = _mmpose_landmarks(
            instance.get("keypoints"),
            instance.get("keypoint_scores"),
            width,
            height,
        )
        confidence = sum(point["visibility"] for point in landmarks) / len(landmarks)
        bbox = _bbox(landmarks)
        candidates.append(
            {
                "providerIndex": provider_index,
                "landmarks": landmarks,
                "confidence": round(confidence, 6),
                "bbox": bbox,
            }
        )
    return candidates


def _bbox_area(bbox: dict[str, float]) -> float:
    return max(0.0, bbox["width"]) * max(0.0, bbox["height"])


def _bbox_iou(first: dict[str, float], second: dict[str, float]) -> float:
    first_right, first_bottom = first["x"] + first["width"], first["y"] + first["height"]
    second_right, second_bottom = second["x"] + second["width"], second["y"] + second["height"]
    intersection_width = max(0.0, min(first_right, second_right) - max(first["x"], second["x"]))
    intersection_height = max(0.0, min(first_bottom, second_bottom) - max(first["y"], second["y"]))
    intersection = intersection_width * intersection_height
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _select_mmpose_subject(
    candidates: list[dict[str, Any]], previous_bbox: dict[str, float] | None
) -> tuple[dict[str, Any], float | None]:
    """Keep a primary subject stable using IoU after the initial confident/largest selection."""
    if not candidates:
        raise ValueError("Cannot select an MMPose subject from an empty candidate list.")
    if previous_bbox is None:
        return max(candidates, key=lambda candidate: (candidate["confidence"], _bbox_area(candidate["bbox"]))), None
    selected = max(
        candidates,
        key=lambda candidate: (
            _bbox_iou(previous_bbox, candidate["bbox"]),
            candidate["confidence"],
            _bbox_area(candidate["bbox"]),
        ),
    )
    return selected, round(_bbox_iou(previous_bbox, selected["bbox"]), 6)


def _create_mmpose_inferencer(options: MMPoseOptions) -> tuple[Any, str]:
    """Create MMPose only when all local assets are explicitly supplied."""
    for label, path in (
        ("pose config", options.pose_config),
        ("pose checkpoint", options.pose_weights),
        ("detector config", options.detector_config),
        ("detector checkpoint", options.detector_weights),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"MMPose {label} is not a local file: {path}")
    try:
        import mmpose
        from mmpose.apis import MMPoseInferencer
    except ImportError as error:
        raise RuntimeError(
            "MMPose provider requested but its runtime is unavailable. Install the optional "
            "MMPose dependencies and a supported PyTorch build; Open Motion Bridge will not "
            "silently fall back to MediaPipe."
        ) from error
    inferencer = MMPoseInferencer(
        pose2d=str(options.pose_config),
        pose2d_weights=str(options.pose_weights),
        det_model=str(options.detector_config),
        det_weights=str(options.detector_weights),
        det_cat_ids=[0],
        device=options.device,
    )
    return inferencer, str(getattr(mmpose, "__version__", "unknown"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _run_ffprobe(video: Path) -> dict[str, Any]:
    completed = _run_quiet(
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
    completed = _run_quiet(
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


def _mediapipe_landmarks(result: Any) -> list[dict[str, Any]]:
    if not result.pose_landmarks:
        return []
    return [
        {
            "name": mp.solutions.pose.PoseLandmark(number).name.lower(),
            "x": round(float(item.x), 6),
            "y": round(float(item.y), 6),
            "z": round(float(item.z), 6),
            "visibility": round(float(item.visibility), 6),
        }
        for number, item in enumerate(result.pose_landmarks.landmark)
    ]


def analyze_video(
    video: Path,
    output_dir: Path,
    sample_fps: float,
    force: bool,
    pose_provider: str = "mediapipe",
    mmpose_options: MMPoseOptions | None = None,
) -> None:
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if sample_fps < 0:
        raise ValueError("--sample-fps must be zero (native) or greater")
    if pose_provider not in {"mediapipe", "mmpose-rtmpose-l-wholebody"}:
        raise ValueError(f"Unsupported pose provider: {pose_provider}")
    if pose_provider == "mmpose-rtmpose-l-wholebody" and mmpose_options is None:
        raise ValueError("MMPose provider requires explicit local model and detector assets.")
    mmpose_inferencer: Any | None = None
    mmpose_version: str | None = None
    if pose_provider == "mmpose-rtmpose-l-wholebody":
        if mmpose_options is None:
            raise AssertionError("MMPose options were not validated.")
        # Fail before ingest or artifact creation when optional local runtime/assets are unavailable.
        mmpose_inferencer, mmpose_version = _create_mmpose_inferencer(mmpose_options)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = _run_ffprobe(video)
    frame_timestamps = _frame_timestamps(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not decode: {video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or (sample_fps if sample_fps > 0 else 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = _sample_step(source_fps, sample_fps)

    frames: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    detection_count = 0
    pose: Any | None = None
    previous_mmpose_bbox: dict[str, float] | None = None
    mmpose_multi_candidate_frames = 0
    mmpose_low_iou_frames: list[int] = []
    if pose_provider == "mediapipe":
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        provider_metadata = {
            "name": "mediapipe-pose",
            "version": getattr(mp, "__version__", "unknown"),
            "modelIdentifier": "Pose(model_complexity=1)",
            "licenseHint": "Verify MediaPipe and model distribution terms before release.",
        }
        analysis_limitations = [
            "Object tracking and camera stabilization are not implemented in this vertical slice.",
            "stabilizedSpace is intentionally unavailable rather than fabricated.",
        ]
    else:
        if mmpose_options is None or mmpose_version is None:
            raise AssertionError("MMPose provider was not initialized.")
        provider_metadata = {
            "name": "mmpose-rtmpose-l-wholebody",
            "version": mmpose_version,
            "modelIdentifier": "RTMPose-L COCO-WholeBody (133 keypoints)",
            "licenseHint": "MMPose runtime is Apache-2.0; verify local checkpoint and detector licenses before release.",
            "assets": {
                "poseConfig": mmpose_options.pose_config.name,
                "poseCheckpoint": mmpose_options.pose_weights.name,
                "detectorConfig": mmpose_options.detector_config.name,
                "detectorCheckpoint": mmpose_options.detector_weights.name,
                "device": mmpose_options.device,
            },
        }
        analysis_limitations = [
            "Object tracking and camera stabilization are not implemented in this vertical slice.",
            "stabilizedSpace is intentionally unavailable rather than fabricated.",
            "The MMPose provider emits one primary subject selected by bbox IoU after the initial confident/largest instance; full multi-person persistent track export is not implemented.",
        ]
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
            selection_iou: float | None = None
            if pose is not None:
                landmarks = _mediapipe_landmarks(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                if mmpose_inferencer is None:
                    raise AssertionError("MMPose provider was not initialized.")
                mmpose_result = next(mmpose_inferencer(frame, return_vis=False))
                candidates = _mmpose_candidates(mmpose_result, width, height)
                if len(candidates) > 1:
                    mmpose_multi_candidate_frames += 1
                if candidates:
                    selected, selection_iou = _select_mmpose_subject(candidates, previous_mmpose_bbox)
                    previous_mmpose_bbox = selected["bbox"]
                    landmarks = selected["landmarks"]
                else:
                    landmarks = []
            if landmarks:
                mean_visibility = sum(item["visibility"] for item in landmarks) / len(landmarks)
                uncertain_subject_match = selection_iou is not None and selection_iou < 0.05
                if uncertain_subject_match:
                    mmpose_low_iou_frames.append(index)
                observations.append(
                    {
                        "frameIndex": index,
                        "sourceTimeMs": round(timestamp_ms, 3),
                        "confidence": round(mean_visibility, 6),
                        "occlusion": "none" if mean_visibility >= 0.5 else "partial",
                        "screenSpace": {"bbox": _bbox(landmarks), "keypoints": landmarks},
                        "quality": {
                            "interpolated": False,
                            "driftWarning": uncertain_subject_match,
                            "manualCorrectionRequired": mean_visibility < 0.5 or uncertain_subject_match,
                        },
                    }
                )
                detection_count += 1
            index += 1
    finally:
        if pose is not None:
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
        "sampling": {
            "requestedFps": "source-native" if sample_fps == 0 else sample_fps,
            "effectiveFps": round(source_fps / step, 6),
            "sourceFrameStep": step,
            "sampledFrames": len(frames),
        },
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
                "provider": provider_metadata,
                "lifecycle": {
                    "firstFrame": observations[0]["frameIndex"] if observations else None,
                    "lastFrame": observations[-1]["frameIndex"] if observations else None,
                    "reidentifiedFrom": [],
                    "idChanges": [
                        {"frameIndex": frame_index, "reason": "low-iou-primary-subject-match"}
                        for frame_index in mmpose_low_iou_frames
                    ],
                },
                "observations": observations,
            }
        ],
        "cameraMotion": [],
        "sampling": manifest["sampling"],
        "analysis": {
            "status": "completed" if observations else "completed-with-no-pose-detections",
            "sampledFrames": len(frames),
            "poseDetections": detection_count,
            "continuity": round(continuity, 6),
            "limitations": analysis_limitations,
            "mmposeSubjectSelection": (
                {
                    "policy": "initial-confidence-area, then bbox-iou",
                    "multipleCandidateFrames": mmpose_multi_candidate_frames,
                    "lowIouMatchFrames": mmpose_low_iou_frames,
                }
                if pose_provider == "mmpose-rtmpose-l-wholebody"
                else None
            ),
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
    by_name = {str(point["name"]): point for point in points}
    lines = []
    for start, end in POSE_CONNECTION_NAMES:
        a, b = by_name.get(start), by_name.get(end)
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


def _binding_elements_markup(bindings_payload: dict[str, Any] | None) -> str:
    """Emit one inspectable DOM node per binding so generated source maps back to a binding id."""
    if not bindings_payload:
        return ""
    elements: list[str] = []
    for binding in bindings_payload["bindings"]:
        identifier = html.escape(str(binding["id"]))
        style = binding.get("style", {})
        if binding["kind"] == "text":
            declarations = ";".join(
                (
                    f'color:{html.escape(str(style.get("color", "#fff2d9")))}',
                    f'font-weight:{html.escape(str(style.get("fontWeight", "800")))}',
                    f'font-family:{html.escape(str(style.get("fontFamily", "Arial, sans-serif")))}',
                    f'letter-spacing:{html.escape(str(style.get("letterSpacing", "0.04em")))}',
                    f'-webkit-text-stroke:{html.escape(str(style.get("strokeWidth", "6px")))}'
                    f' {html.escape(str(style.get("strokeColor", "rgba(23,19,15,0.88)")))}',
                )
            )
            elements.append(
                f'<div class="omb-binding omb-binding-text" data-binding-id="{identifier}" '
                f'style="{declarations}">{html.escape(str(binding["text"]))}</div>'
            )
        else:
            elements.append(
                f'<img class="omb-binding omb-binding-image" data-binding-id="{identifier}" '
                f'src="{html.escape(str(binding["source"]))}" alt="" />'
            )
    return "".join(elements)


def _binding_render_payload(bindings_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Reduce the resolved binding table to the fields a composition actually reads."""
    if not bindings_payload:
        return []
    return [
        {
            "id": binding["id"],
            "kind": binding["kind"],
            "aspect": binding["aspect"],
            "frames": [
                {
                    "t": frame["t"],
                    "x": frame["x"],
                    "y": frame["y"],
                    "size": frame["size"],
                    "rotation": frame["rotation"],
                    "opacity": frame["opacity"],
                }
                for frame in binding["frames"]
            ],
        }
        for binding in bindings_payload["bindings"]
    ]


def _stage_binding_assets(
    bindings_payload: dict[str, Any], edit_spec: Path | None, assets_dir: Path
) -> None:
    """Copy referenced image assets next to the composition and rewrite their source paths.

    Image sources are resolved relative to the EditSpec file so a spec stays portable.
    The true pixel aspect ratio replaces any declared value, because a wrong aspect
    silently distorts every attached asset and cannot be caught by coordinate checks.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    base_dir = edit_spec.parent if edit_spec is not None else Path.cwd()
    for binding in bindings_payload["bindings"]:
        if binding["kind"] != "image":
            continue
        raw_source = str(binding["source"])
        candidate = Path(raw_source)
        resolved = candidate if candidate.is_absolute() else (base_dir / candidate)
        resolved = resolved.expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Binding {binding['id']!r} references a missing image asset: {resolved}"
            )
        staged_name = f"binding-{binding['id']}{resolved.suffix.lower()}"
        shutil.copy2(resolved, assets_dir / staged_name)
        image = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Binding {binding['id']!r} image could not be decoded: {resolved}")
        pixel_height, pixel_width = image.shape[0], image.shape[1]
        if pixel_height <= 0:
            raise RuntimeError(f"Binding {binding['id']!r} image has an invalid height: {resolved}")
        binding["source"] = f"assets/{staged_name}"
        binding["aspect"] = round(pixel_width / pixel_height, 6)
        binding["assetEvidence"] = {
            "originalPath": str(resolved),
            "sha256": _sha256(resolved),
            "pixelWidth": int(pixel_width),
            "pixelHeight": int(pixel_height),
        }


def _render_html(
    ir: dict[str, Any],
    pose_frames: list[dict[str, Any]],
    profile: str,
    bindings_payload: dict[str, Any] | None = None,
    overlay: str = "skeleton",
) -> str:
    if overlay not in OVERLAY_MODES:
        raise ValueError(f"Unknown overlay mode: {overlay}")
    source = ir["source"]
    source_width, source_height = int(source["displayWidth"]), int(source["displayHeight"])
    width, height = source_width, source_height
    duration = max(0.1, float(source.get("renderDurationMs", source["durationMs"])) / 1000.0)
    source_track = next((track for track in ir.get("tracks", []) if track.get("type") == "pose"), {})
    provider_name = html.escape(str(source_track.get("provider", {}).get("name", "local-pose-provider")))
    if profile in {"youtube-shorts-9x16", "instagram-reel-9x16"}:
        width, height = 1080, 1920
    show_skeleton = overlay in {"skeleton", "both"}
    show_bindings = overlay in {"bindings", "both"}
    hud_display = "" if show_skeleton else "display:none;"
    skeleton_display = "" if show_skeleton else "display:none;"
    binding_markup = _binding_elements_markup(bindings_payload) if show_bindings else ""
    bindings_json = json.dumps(
        _binding_render_payload(bindings_payload) if show_bindings else [],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    size_scale = round(width / source_width, 6)
    pose_json = json.dumps(pose_frames, ensure_ascii=False, separators=(",", ":"))
    connections_json = json.dumps(POSE_CONNECTION_NAMES, separators=(",", ":"))
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
      #pose-canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; {skeleton_display} }}
      #binding-layer {{ position: absolute; inset: 0; pointer-events: none; }}
      .omb-binding {{ position: absolute; left: 0; top: 0; opacity: 0; transform-origin: 50% 50%; will-change: transform, opacity; white-space: nowrap; }}
      .omb-binding-text {{ line-height: 1; paint-order: stroke fill; }}
      .omb-binding-image {{ object-fit: contain; }}
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
      <div id="binding-layer" class="clip" data-start="0" data-duration="{duration:.3f}" data-track-index="4">{binding_markup}</div>
      <section id="hud" class="clip" data-start="0" data-duration="2.1" data-track-index="5" style="{skeleton_display}">
        <div id="hud-backplate"></div>
        <div id="hud-inner"><div id="hud-label">POSE TRACE</div><div id="hud-detail">local {provider_name} → Tracking IR → HyperFrames</div></div>
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
      const interpolatePose = (time) => {{
        let low = 0;
        let high = poseFrames.length - 1;
        while (low < high) {{
          const middle = Math.ceil((low + high) / 2);
          if (poseFrames[middle].time <= time) low = middle; else high = middle - 1;
        }}
        const left = poseFrames[low];
        const right = poseFrames[Math.min(low + 1, poseFrames.length - 1)];
        if (!left || !right || left === right || right.time <= left.time) return left;
        const ratio = Math.max(0, Math.min(1, (time - left.time) / (right.time - left.time)));
        return {{
          points: Object.fromEntries(Object.entries(left.points).map(([name, point]) => {{
            const next = right.points[name] || point;
            return [name, point.map((value, axis) => value + (next[axis] - value) * ratio)];
          }})),
        }};
      }};
      const drawPose = (time) => {{
        poseContext.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
        const frame = interpolatePose(time);
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
        for (const point of Object.values(frame.points)) {{
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
      const bindingTracks = {bindings_json};
      const bindingSizeScale = {size_scale};
      const bindingNodes = bindingTracks.map((track) => {{
        const node = document.querySelector('[data-binding-id="' + track.id + '"]');
        if (node && track.kind === 'text') node.style.fontSize = '100px';
        return {{ track: track, node: node, measured: 0 }};
      }});
      const bindingFrameAt = (frames, time) => {{
        let low = 0;
        let high = frames.length - 1;
        while (low < high) {{
          const middle = Math.ceil((low + high) / 2);
          if (frames[middle].t <= time) low = middle; else high = middle - 1;
        }}
        const left = frames[low];
        const right = frames[Math.min(low + 1, frames.length - 1)];
        if (!left || !right || left === right || right.t <= left.t) return left;
        const ratio = Math.max(0, Math.min(1, (time - left.t) / (right.t - left.t)));
        const mix = (a, b) => a + (b - a) * ratio;
        return {{
          t: time,
          x: mix(left.x, right.x),
          y: mix(left.y, right.y),
          size: mix(left.size, right.size),
          rotation: mix(left.rotation, right.rotation),
          opacity: mix(left.opacity, right.opacity),
        }};
      }};
      const drawBindings = (time) => {{
        for (const entry of bindingNodes) {{
          if (!entry.node) continue;
          const frame = bindingFrameAt(entry.track.frames, time);
          if (!frame) continue;
          if (frame.opacity <= 0) {{ entry.node.style.opacity = '0'; continue; }}
          const sizePx = frame.size * bindingSizeScale;
          let width = sizePx;
          let height = sizePx / (entry.track.aspect || 1);
          if (entry.track.kind === 'text') {{
            if (!entry.measured) entry.measured = entry.node.getBoundingClientRect().width / 100 || 1;
            entry.node.style.fontSize = sizePx.toFixed(3) + 'px';
            width = sizePx * entry.measured;
            height = sizePx;
          }} else {{
            entry.node.style.width = width.toFixed(3) + 'px';
            entry.node.style.height = height.toFixed(3) + 'px';
          }}
          const left = frame.x * {width} - width / 2;
          const top = frame.y * {height} - height / 2;
          entry.node.style.opacity = String(frame.opacity);
          entry.node.style.transform =
            'translate(' + left.toFixed(3) + 'px,' + top.toFixed(3) + 'px) rotate(' + frame.rotation.toFixed(3) + 'deg)';
        }}
      }};
      const bindingState = {{ time: 0 }};
      drawBindings(0);
      if (bindingTracks.length) {{
        tl.to(bindingState, {{ time: {duration:.3f}, duration: {duration:.3f}, ease: 'none', onUpdate: () => drawBindings(bindingState.time) }}, 0);
      }}
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


def _low_pass(value: float, previous: float, cutoff: float, delta_seconds: float) -> float:
    rate = 2.0 * 3.141592653589793 * cutoff * delta_seconds
    alpha = rate / (rate + 1.0)
    return alpha * value + (1.0 - alpha) * previous


def _smooth_pose_observations(
    observations: list[dict[str, Any]], config: TemporalSmoothingConfig
) -> list[dict[str, Any]]:
    """Apply One Euro filtering without ever mutating the immutable source IR."""
    states: dict[str, dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for observation in observations:
        timestamp_ms = float(observation["sourceTimeMs"])
        input_points = observation["screenSpace"]["keypoints"]
        filtered_points: list[dict[str, Any]] = []
        held_for_occlusion = False
        for point in input_points:
            point_name = str(point["name"])
            state = states.setdefault(
                point_name,
                {"last_time_ms": None, "last_raw": {}, "last_filtered": {}, "last_derivative": {}, "last_good_ms": None},
            )
            visibility = float(point["visibility"])
            valid = visibility >= config.visibility_threshold
            filtered_point = {"name": point_name}
            if valid:
                previous_time = state["last_time_ms"]
                delta_seconds = max(1.0 / 240.0, (timestamp_ms - previous_time) / 1000.0) if previous_time is not None else 0.0
                for axis in ("x", "y", "z"):
                    raw_value = float(point[axis])
                    if previous_time is None:
                        filtered_value = raw_value
                        derivative = 0.0
                    else:
                        raw_derivative = (raw_value - state["last_raw"][axis]) / delta_seconds
                        derivative = _low_pass(
                            raw_derivative,
                            state["last_derivative"][axis],
                            config.derivative_cutoff,
                            delta_seconds,
                        )
                        cutoff = config.min_cutoff + config.beta * abs(derivative)
                        filtered_value = _low_pass(raw_value, state["last_filtered"][axis], cutoff, delta_seconds)
                    state["last_raw"][axis] = raw_value
                    state["last_filtered"][axis] = filtered_value
                    state["last_derivative"][axis] = derivative
                    filtered_point[axis] = round(filtered_value, 6)
                state["last_time_ms"] = timestamp_ms
                state["last_good_ms"] = timestamp_ms
                filtered_point["visibility"] = round(visibility, 6)
            else:
                gap_ms = timestamp_ms - state["last_good_ms"] if state["last_good_ms"] is not None else float("inf")
                if state["last_filtered"] and gap_ms <= config.max_gap_ms:
                    for axis in ("x", "y", "z"):
                        filtered_point[axis] = round(float(state["last_filtered"][axis]), 6)
                    # The render IR marks this as an inferred short gap; raw visibility remains untouched in the source IR.
                    filtered_point["visibility"] = round(config.visibility_threshold, 6)
                    held_for_occlusion = True
                else:
                    for axis in ("x", "y", "z"):
                        filtered_point[axis] = round(float(point[axis]), 6)
                    filtered_point["visibility"] = 0.0
            filtered_points.append(filtered_point)

        confidence = sum(point["visibility"] for point in filtered_points) / len(filtered_points)
        output.append(
            {
                "frameIndex": observation.get("frameIndex"),
                "sourceTimeMs": round(timestamp_ms, 3),
                "confidence": round(confidence, 6),
                "occlusion": "interpolated-short-gap" if held_for_occlusion else observation.get("occlusion", "none"),
                "screenSpace": {"bbox": _bbox(filtered_points), "keypoints": filtered_points},
                "quality": {
                    "interpolated": held_for_occlusion,
                    "temporalSmoothingApplied": True,
                    "manualCorrectionRequired": bool(observation.get("quality", {}).get("manualCorrectionRequired", False)),
                },
            }
        )
    return output


def _interpolate_render_observations(
    observations: list[dict[str, Any]], duration_ms: float, config: TemporalSmoothingConfig
) -> list[dict[str, Any]]:
    """Resample filtered observations to the renderer FPS, hiding long uncertain gaps."""
    if not observations:
        return []
    source_times = [float(observation["sourceTimeMs"]) for observation in observations]
    render_frame_count = max(1, int((duration_ms / 1000.0) * config.render_fps + 0.999999))
    rendered: list[dict[str, Any]] = []
    for render_frame_index in range(render_frame_count):
        timestamp_ms = render_frame_index * 1000.0 / config.render_fps
        right_index = bisect_right(source_times, timestamp_ms)
        left_index = right_index - 1
        if left_index < 0:
            left_index = 0
        if right_index >= len(observations):
            right_index = left_index
        left = observations[left_index]
        right = observations[right_index]
        left_time = source_times[left_index]
        right_time = source_times[right_index]
        gap_ms = right_time - left_time
        ratio = 0.0 if gap_ms <= 0 else (timestamp_ms - left_time) / gap_ms
        can_interpolate = left_index != right_index and 0 < gap_ms <= config.max_gap_ms
        outside_short_hold = not can_interpolate and abs(timestamp_ms - left_time) > config.max_gap_ms
        points: list[dict[str, Any]] = []
        right_points = {str(point["name"]): point for point in right["screenSpace"]["keypoints"]}
        for left_point in left["screenSpace"]["keypoints"]:
            point_name = str(left_point["name"])
            right_point = right_points.get(point_name, left_point)
            point = {"name": point_name}
            if can_interpolate:
                for axis in ("x", "y", "z", "visibility"):
                    point[axis] = round(float(left_point[axis]) + (float(right_point[axis]) - float(left_point[axis])) * ratio, 6)
            else:
                for axis in ("x", "y", "z", "visibility"):
                    point[axis] = round(float(left_point[axis]), 6)
            if outside_short_hold:
                point["visibility"] = 0.0
            points.append(point)
        confidence = sum(point["visibility"] for point in points) / len(points)
        rendered.append(
            {
                "frameIndex": left.get("frameIndex"),
                "renderFrameIndex": render_frame_index,
                "sourceTimeMs": round(timestamp_ms, 3),
                "confidence": round(confidence, 6),
                "occlusion": "long-gap-hidden" if outside_short_hold else left.get("occlusion", "none"),
                "screenSpace": {"bbox": _bbox(points), "keypoints": points},
                "quality": {
                    "interpolated": can_interpolate,
                    "temporalSmoothingApplied": True,
                    "manualCorrectionRequired": outside_short_hold or bool(left.get("quality", {}).get("manualCorrectionRequired", False)),
                },
            }
        )
    return rendered


def _build_render_tracking_ir(ir: dict[str, Any], config: TemporalSmoothingConfig) -> dict[str, Any]:
    source_track = next((track for track in ir.get("tracks", []) if track.get("type") == "pose"), None)
    if not source_track:
        raise RuntimeError("No pose track is available for render-time temporal processing.")
    filtered = _smooth_pose_observations(source_track.get("observations", []), config)
    duration_ms = float(ir["source"].get("renderDurationMs", ir["source"]["durationMs"]))
    rendered = _interpolate_render_observations(filtered, duration_ms, config)
    if not rendered:
        raise RuntimeError("No pose observations were produced; refusing to generate an unverified overlay project.")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "source": ir["source"],
        "coordinateSystem": ir["coordinateSystem"],
        "tracks": [
            {
                "id": f'{source_track["id"]}-render',
                "type": "pose",
                "provider": {
                    "name": "open-motion-bridge-temporal-resolver",
                    "modelIdentifier": f"one-euro/{config.profile}",
                    "sourceTrackId": source_track["id"],
                    "sourceProvider": source_track.get("provider", {}).get("name", "unknown"),
                },
                "lifecycle": source_track.get("lifecycle", {}),
                "observations": rendered,
            }
        ],
        "temporalProcessing": {
            "profile": config.profile,
            "renderFps": config.render_fps,
            "algorithm": "one-euro-filter + confidence-aware linear interpolation",
            "minCutoff": config.min_cutoff,
            "beta": config.beta,
            "derivativeCutoff": config.derivative_cutoff,
            "visibilityThreshold": config.visibility_threshold,
            "maxGapMs": config.max_gap_ms,
            "rawIrMutable": False,
        },
        "provenance": [
            *ir.get("provenance", []),
            {
                "kind": "temporal-resolution",
                "createdAt": _utc_now(),
                "tool": "open-motion-bridge",
                "toolVersion": SCHEMA_VERSION,
                "sourceTrackId": source_track["id"],
            },
        ],
    }


def generate_projects(
    ir_path: Path,
    source_video: Path,
    output_dir: Path,
    target: str,
    profile: str,
    force: bool,
    render_fps: float = 30.0,
    smoothing_profile: str = "balanced",
    visibility_threshold: float = 0.2,
    max_gap_ms: float = 250.0,
    edit_spec: Path | None = None,
    overlay: str = "skeleton",
) -> None:
    if not ir_path.is_file():
        raise FileNotFoundError(f"Tracking IR not found: {ir_path}")
    if not source_video.is_file():
        raise FileNotFoundError(f"Source video not found: {source_video}")
    if overlay not in OVERLAY_MODES:
        raise ValueError(f"Unknown overlay mode: {overlay}; expected one of {list(OVERLAY_MODES)}")
    if overlay in {"bindings", "both"} and edit_spec is None:
        raise ValueError(f"Overlay mode {overlay!r} requires --edit-spec; refusing to render an empty asset layer.")
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    if ir.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported IR schema: {ir.get('schemaVersion')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_config = _temporal_smoothing_config(
        profile=smoothing_profile,
        render_fps=render_fps,
        visibility_threshold=visibility_threshold,
        max_gap_ms=max_gap_ms,
    )
    render_ir = _build_render_tracking_ir(ir, temporal_config)
    observations = render_ir["tracks"][0]["observations"]
    total_duration = float(ir["source"].get("renderDurationMs", ir["source"]["durationMs"])) / 1000.0
    source_width = int(ir["source"]["displayWidth"])
    source_height = int(ir["source"]["displayHeight"])

    bindings_payload: dict[str, Any] | None = None
    if edit_spec is not None:
        bindings = load_edit_spec(edit_spec)
        bindings_payload = resolve_bindings(bindings, render_ir, source_width, source_height)
        bindings_payload["sourceEditSpec"] = edit_spec.name
        bindings_payload["sourceHash"] = ir["source"]["sourceHash"]

    skeletons: list[tuple[float, float, str]] = []
    for index, observation in enumerate(observations):
        start = round(float(observation["sourceTimeMs"]) / 1000.0, 3)
        next_time = float(observations[index + 1]["sourceTimeMs"]) / 1000.0 if index + 1 < len(observations) else total_duration
        next_start = round(next_time, 3)
        skeletons.append((start, max(0.001, next_start - start), _skeleton_markup(observation, int(ir["source"]["displayWidth"]), int(ir["source"]["displayHeight"]))))
    if not skeletons:
        raise RuntimeError("No pose observations were produced; refusing to generate an unverified overlay project.")
    _write_json(output_dir / "render.tracking.ir.json", render_ir, force)
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
        if bindings_payload is not None:
            _stage_binding_assets(bindings_payload, edit_spec, assets)
        if force:
            compositions_dir = output_dir / "compositions"
            if compositions_dir.exists():
                for stale_file in compositions_dir.glob("pose-chunk-*.html"):
                    stale_file.unlink()
        pose_frames = [
            {
                "time": round(float(observation["sourceTimeMs"]) / 1000.0, 6),
                "points": {
                    str(point["name"]): [
                        round(float(point["x"]), 6),
                        round(float(point["y"]), 6),
                        round(float(point["visibility"]), 6),
                    ]
                    for point in observation["screenSpace"]["keypoints"]
                },
            }
            for observation in observations
        ]
        html_path.write_text(
            _render_html(ir, pose_frames, profile, bindings_payload, overlay), encoding="utf-8"
        )
        if bindings_payload is not None:
            _write_json(output_dir / "bindings.resolved.json", bindings_payload, force)
        _write_json(
            output_dir / "open-motion-bridge.generated.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "createdAt": _utc_now(),
                "sourceIr": str(ir_path.name),
                "sourceHash": ir["source"]["sourceHash"],
                "profile": profile,
                "target": "hyperframes",
                "overlay": overlay,
                "trackId": render_ir["tracks"][0]["id"],
                "observationCount": len(skeletons),
                "compositionDurationMs": round(float(ir["source"].get("renderDurationMs", ir["source"]["durationMs"])), 3),
                "temporalProcessing": render_ir["temporalProcessing"],
                "bindings": [
                    {"id": binding["id"], "kind": binding["kind"], "stats": binding["stats"]}
                    for binding in (bindings_payload or {}).get("bindings", [])
                ],
            },
            force,
        )
    if target in {"sketch-svg", "both"}:
        svg_path = output_dir / "sketch-pose-trace.svg"
        if svg_path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {svg_path}; use --force for generated output.")
        svg_path.write_text(_render_svg(ir, skeletons), encoding="utf-8")


def _rendered_video_dimensions(video: Path) -> tuple[int, int]:
    probe = _run_ffprobe(video)
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width > 0 and height > 0:
            return width, height
    raise RuntimeError(f"ffprobe reported no usable video dimensions: {video}")


def _extract_frame(video: Path, timestamp_seconds: float, destination: Path) -> None:
    """Decode one frame at a presentation time so measurements use real rendered pixels."""
    _run_quiet(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp_seconds:.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    if not destination.is_file():
        raise RuntimeError(f"ffmpeg produced no frame at {timestamp_seconds:.3f}s from {video}")


def _sample_binding_frames(binding: dict[str, Any], samples: int) -> list[dict[str, Any]]:
    """Pick evenly spaced frames where the binding is fully opaque.

    Faded frames are deliberately excluded: a partially transparent asset cannot be
    localized reliably, so including it would weaken the measurement instead of the render.
    """
    visible = [frame for frame in binding["frames"] if float(frame["opacity"]) >= _VERIFY_MIN_OPACITY]
    if not visible:
        return []
    if len(visible) <= samples:
        return list(visible)
    step = (len(visible) - 1) / (samples - 1) if samples > 1 else 0.0
    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    for index in range(samples):
        position = int(round(index * step))
        if position in used:
            continue
        used.add(position)
        picked.append(visible[position])
    return picked


def _measure_binding_in_frame(
    numpy_module: Any,
    rendered_frame: Any,
    project_dir: Path,
    binding: dict[str, Any],
    frame: dict[str, Any],
    pixel_scale: float,
    rendered_width: int,
    rendered_height: int,
    tolerance_px: float,
) -> dict[str, Any]:
    """Locate the staged asset near its resolved position and report the pixel error."""
    expected_x = float(frame["x"]) * rendered_width
    expected_y = float(frame["y"]) * rendered_height
    result: dict[str, Any] = {
        "t": round(float(frame["t"]), 6),
        "expected": {"x": round(expected_x, 3), "y": round(expected_y, 3)},
        "measured": None,
        "errorPx": None,
        "matchScore": None,
        "status": "not-measured",
    }

    asset_path = project_dir / str(binding["source"])
    if not asset_path.is_file():
        result["status"] = "missing-staged-asset"
        return result
    asset = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    if asset is None:
        result["status"] = "undecodable-staged-asset"
        return result

    template_width = max(2, int(round(float(frame["size"]) * pixel_scale)))
    aspect = float(binding.get("aspect") or 1.0)
    template_height = max(2, int(round(template_width / aspect)))
    resized = cv2.resize(asset, (template_width, template_height), interpolation=cv2.INTER_AREA)
    mask = None
    if resized.ndim == 2:
        template = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    elif resized.shape[2] == 4:
        template = numpy_module.ascontiguousarray(resized[:, :, :3])
        alpha = numpy_module.ascontiguousarray(resized[:, :, 3])
        mask = cv2.merge([alpha, alpha, alpha])
    else:
        template = numpy_module.ascontiguousarray(resized[:, :, :3])

    # Search a bounded neighbourhood so a visually similar area elsewhere in the frame
    # cannot masquerade as a correct placement.
    margin = max(tolerance_px * 4.0, 48.0)
    half_width = template_width / 2.0 + margin
    half_height = template_height / 2.0 + margin
    x0 = int(max(0, round(expected_x - half_width)))
    y0 = int(max(0, round(expected_y - half_height)))
    x1 = int(min(rendered_frame.shape[1], round(expected_x + half_width)))
    y1 = int(min(rendered_frame.shape[0], round(expected_y + half_height)))
    region = rendered_frame[y0:y1, x0:x1]
    if region.shape[0] < template_height or region.shape[1] < template_width:
        result["status"] = "expected-position-outside-measurable-area"
        return result

    scores = cv2.matchTemplate(region, template, cv2.TM_CCORR_NORMED, mask=mask)
    scores = numpy_module.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, max_score, _, max_location = cv2.minMaxLoc(scores)
    measured_x = x0 + max_location[0] + template_width / 2.0
    measured_y = y0 + max_location[1] + template_height / 2.0
    error = ((measured_x - expected_x) ** 2 + (measured_y - expected_y) ** 2) ** 0.5

    result["measured"] = {"x": round(measured_x, 3), "y": round(measured_y, 3)}
    result["errorPx"] = round(error, 3)
    result["matchScore"] = round(float(max_score), 6)
    result["templateSize"] = {"width": template_width, "height": template_height}
    result["searchRegion"] = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
    if float(max_score) < _VERIFY_MIN_MATCH_SCORE:
        result["status"] = "unreliable-match"
    elif error <= tolerance_px:
        result["status"] = "measured"
    else:
        result["status"] = "measured-out-of-tolerance"
    return result


def verify_render(
    project_dir: Path,
    rendered_video: Path,
    output_path: Path,
    samples: int = 8,
    tolerance_px: float = 24.0,
    force: bool = False,
) -> dict[str, Any]:
    """Re-measure rendered asset placement against the resolved binding table.

    This is the only step that can claim a generated video matches its approved data,
    because it reads the rendered pixels instead of trusting the generator.
    """
    import numpy

    if not project_dir.is_dir():
        raise FileNotFoundError(f"Generated project directory not found: {project_dir}")
    if not rendered_video.is_file():
        raise FileNotFoundError(f"Rendered video not found: {rendered_video}")
    if samples < 1:
        raise ValueError("--samples must be at least 1")
    if tolerance_px <= 0:
        raise ValueError("--tolerance-px must be greater than zero")

    bindings_path = project_dir / "bindings.resolved.json"
    if not bindings_path.is_file():
        raise FileNotFoundError(
            f"No resolved binding table in {project_dir}; run generate with --edit-spec before verify."
        )
    payload = json.loads(bindings_path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported resolved binding schema: {payload.get('schemaVersion')!r}")

    space = payload["coordinateSpace"]
    source_width = int(space["pixelWidth"])
    source_height = int(space["pixelHeight"])
    rendered_width, rendered_height = _rendered_video_dimensions(rendered_video)
    pixel_scale = rendered_width / source_width

    generated_path = project_dir / "open-motion-bridge.generated.json"
    generated = json.loads(generated_path.read_text(encoding="utf-8")) if generated_path.is_file() else {}

    plan: dict[float, list[tuple[int, dict[str, Any]]]] = {}
    reports: list[dict[str, Any]] = []
    for index, binding in enumerate(payload["bindings"]):
        entry: dict[str, Any] = {
            "id": binding["id"],
            "kind": binding["kind"],
            "anchorLandmarks": binding.get("anchorLandmarks", []),
            "status": "pending",
            "samples": [],
        }
        if binding["kind"] != "image":
            # Text glyph localization is not implemented; reporting it as unmeasured keeps
            # the summary honest instead of implying a verified placement.
            entry["status"] = "not-measurable-non-image-binding"
        else:
            picked = _sample_binding_frames(binding, samples)
            if not picked:
                entry["status"] = "not-measurable-never-fully-visible"
            else:
                for frame in picked:
                    plan.setdefault(round(float(frame["t"]), 6), []).append((index, frame))
        reports.append(entry)

    with tempfile.TemporaryDirectory(prefix="omb-verify-") as temporary:
        temporary_dir = Path(temporary)
        for order, timestamp in enumerate(sorted(plan)):
            frame_path = temporary_dir / f"frame-{order:04d}.png"
            _extract_frame(rendered_video, timestamp, frame_path)
            rendered_frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if rendered_frame is None:
                raise RuntimeError(f"Extracted frame could not be decoded: {frame_path}")
            for binding_index, frame in plan[timestamp]:
                reports[binding_index]["samples"].append(
                    _measure_binding_in_frame(
                        numpy,
                        rendered_frame,
                        project_dir,
                        payload["bindings"][binding_index],
                        frame,
                        pixel_scale,
                        rendered_width,
                        rendered_height,
                        tolerance_px,
                    )
                )

    warnings: list[str] = []
    measurable_count = 0
    passed_count = 0
    overall_max_error = 0.0
    for entry in reports:
        if not entry["samples"]:
            if entry["status"] == "pending":
                entry["status"] = "not-measurable-no-samples"
            warnings.append(f"binding {entry['id']!r}: {entry['status']}")
            continue
        measurable_count += 1
        errors = [sample["errorPx"] for sample in entry["samples"] if sample["errorPx"] is not None]
        failed = [sample for sample in entry["samples"] if sample["status"] != "measured"]
        entry["measuredSamples"] = len(entry["samples"])
        entry["maxErrorPx"] = round(max(errors), 3) if errors else None
        entry["meanErrorPx"] = round(sum(errors) / len(errors), 3) if errors else None
        entry["status"] = "passed" if not failed else "failed"
        if entry["status"] == "passed":
            passed_count += 1
        else:
            for sample in failed:
                warnings.append(
                    f"binding {entry['id']!r} at {sample['t']:.3f}s: {sample['status']}"
                    + (f" ({sample['errorPx']}px)" if sample["errorPx"] is not None else "")
                )
        if errors:
            overall_max_error = max(overall_max_error, max(errors))

    if measurable_count == 0:
        warnings.append("No binding could be measured; this report is not evidence of a correct render.")

    report = {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": _utc_now(),
        "tool": "open-motion-bridge",
        "project": str(project_dir),
        "renderedVideo": {
            "path": str(rendered_video),
            "sha256": _sha256(rendered_video),
            "width": rendered_width,
            "height": rendered_height,
        },
        "sourceSpace": {"width": source_width, "height": source_height, "pixelScale": round(pixel_scale, 6)},
        "generated": {
            "sourceHash": generated.get("sourceHash"),
            "profile": generated.get("profile"),
            "overlay": generated.get("overlay"),
            "temporalProcessing": generated.get("temporalProcessing"),
        },
        "policy": {
            "requestedSamples": samples,
            "tolerancePx": tolerance_px,
            "minMatchScore": _VERIFY_MIN_MATCH_SCORE,
            "minOpacity": _VERIFY_MIN_OPACITY,
            "method": "masked template matching (TM_CCORR_NORMED) inside a bounded neighbourhood",
        },
        "bindings": reports,
        "summary": {
            "measurableBindings": measurable_count,
            "passedBindings": passed_count,
            "maxErrorPx": round(overall_max_error, 3),
            "passed": measurable_count > 0 and passed_count == measurable_count,
            "warnings": warnings,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, report, force)
    return report
