"""Declarative asset binding: attach text or image assets to tracked landmarks.

An EditSpec never mutates the analysis IR. It is resolved against the render-time
Tracking IR into an explicit per-frame transform table so the generated composition
and the render verifier read exactly the same numbers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1.0"

_SCALE_MODES = {"bbox-diagonal", "bbox-width", "bbox-height", "absolute-px"}
_OFFSET_MODES = {"normalized", "size-relative"}
_ROTATE_MODES = {"none", "fixed", "landmark-pair"}
_LOW_CONFIDENCE_POLICIES = {"fade", "hide", "hold"}
_KINDS = {"text", "image"}

# Anchor confidence bands. Above _TRACKED_CONFIDENCE an attachment is treated as
# reliably placed; at or below _HIDDEN_CONFIDENCE no position is trustworthy enough to draw.
_TRACKED_CONFIDENCE = 0.5
_HIDDEN_CONFIDENCE = 0.2


@dataclass(frozen=True)
class Anchor:
    landmarks: tuple[str, ...]


@dataclass(frozen=True)
class Binding:
    id: str
    kind: str
    anchor: Anchor
    text: str | None
    source: str | None
    offset_mode: str
    offset_x: float
    offset_y: float
    scale_mode: str
    scale_value: float
    aspect: float
    rotate_mode: str
    rotate_from: str | None
    rotate_to: str | None
    rotate_degrees: float
    on_low_confidence: str
    start_ms: float
    end_ms: float | None
    max_speed: float
    style: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid EditSpec: {message}")


def _parse_binding(raw: Any, index: int) -> Binding:
    _require(isinstance(raw, dict), f"binding #{index} is not an object")
    identifier = str(raw.get("id") or f"binding-{index:03d}")
    kind = str(raw.get("kind", "")).strip()
    _require(kind in _KINDS, f"binding {identifier!r} kind must be one of {sorted(_KINDS)}")

    anchor_raw = raw.get("anchor")
    _require(isinstance(anchor_raw, dict), f"binding {identifier!r} requires an anchor object")
    landmarks = anchor_raw.get("landmarks")
    _require(
        isinstance(landmarks, list) and bool(landmarks) and all(isinstance(name, str) for name in landmarks),
        f"binding {identifier!r} anchor.landmarks must be a non-empty list of landmark names",
    )

    text = raw.get("text")
    source = raw.get("source")
    if kind == "text":
        _require(isinstance(text, str) and text.strip() != "", f"text binding {identifier!r} requires text")
    else:
        _require(isinstance(source, str) and source.strip() != "", f"image binding {identifier!r} requires source")

    offset = raw.get("offset", {})
    _require(isinstance(offset, dict), f"binding {identifier!r} offset must be an object")
    offset_mode = str(offset.get("mode", "normalized"))
    _require(offset_mode in _OFFSET_MODES, f"binding {identifier!r} offset.mode must be one of {sorted(_OFFSET_MODES)}")

    scale = raw.get("scale", {})
    _require(isinstance(scale, dict), f"binding {identifier!r} scale must be an object")
    scale_mode = str(scale.get("mode", "bbox-diagonal"))
    _require(scale_mode in _SCALE_MODES, f"binding {identifier!r} scale.mode must be one of {sorted(_SCALE_MODES)}")
    scale_value = float(scale.get("value", 0.15))
    _require(scale_value > 0, f"binding {identifier!r} scale.value must be greater than zero")

    rotate = raw.get("rotate", {})
    _require(isinstance(rotate, dict), f"binding {identifier!r} rotate must be an object")
    rotate_mode = str(rotate.get("mode", "none"))
    _require(rotate_mode in _ROTATE_MODES, f"binding {identifier!r} rotate.mode must be one of {sorted(_ROTATE_MODES)}")
    rotate_from = rotate.get("from")
    rotate_to = rotate.get("to")
    if rotate_mode == "landmark-pair":
        _require(
            isinstance(rotate_from, str) and isinstance(rotate_to, str),
            f"binding {identifier!r} landmark-pair rotation requires 'from' and 'to' landmark names",
        )

    on_low_confidence = str(raw.get("onLowConfidence", "fade"))
    _require(
        on_low_confidence in _LOW_CONFIDENCE_POLICIES,
        f"binding {identifier!r} onLowConfidence must be one of {sorted(_LOW_CONFIDENCE_POLICIES)}",
    )

    time_range = raw.get("range", {})
    _require(isinstance(time_range, dict), f"binding {identifier!r} range must be an object")
    start_ms = float(time_range.get("startMs", 0.0))
    end_raw = time_range.get("endMs")
    end_ms = None if end_raw is None else float(end_raw)
    _require(end_ms is None or end_ms > start_ms, f"binding {identifier!r} range.endMs must be after range.startMs")

    aspect = float(raw.get("aspect", 1.0))
    _require(aspect > 0, f"binding {identifier!r} aspect must be greater than zero")

    max_speed = float(raw.get("maxSpeed", 3.0))
    _require(max_speed > 0, f"binding {identifier!r} maxSpeed must be greater than zero")

    style = raw.get("style", {})
    _require(isinstance(style, dict), f"binding {identifier!r} style must be an object")

    return Binding(
        id=identifier,
        kind=kind,
        anchor=Anchor(landmarks=tuple(str(name) for name in landmarks)),
        text=text if kind == "text" else None,
        source=source if kind == "image" else None,
        offset_mode=offset_mode,
        offset_x=float(offset.get("x", 0.0)),
        offset_y=float(offset.get("y", 0.0)),
        scale_mode=scale_mode,
        scale_value=scale_value,
        aspect=aspect,
        rotate_mode=rotate_mode,
        rotate_from=rotate_from,
        rotate_to=rotate_to,
        rotate_degrees=float(rotate.get("degrees", 0.0)),
        on_low_confidence=on_low_confidence,
        start_ms=start_ms,
        end_ms=end_ms,
        max_speed=max_speed,
        style=dict(style),
    )


def load_edit_spec(path: Path) -> list[Binding]:
    if not path.is_file():
        raise FileNotFoundError(f"EditSpec not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(document, dict), "top level value must be an object")
    schema = document.get("schemaVersion")
    _require(schema == SCHEMA_VERSION, f"unsupported EditSpec schemaVersion: {schema!r}")
    raw_bindings = document.get("bindings")
    _require(isinstance(raw_bindings, list) and bool(raw_bindings), "bindings must be a non-empty list")
    bindings = [_parse_binding(raw, index) for index, raw in enumerate(raw_bindings)]
    identifiers = [binding.id for binding in bindings]
    _require(len(set(identifiers)) == len(identifiers), "binding ids must be unique")
    return bindings


def _anchor_state(binding: Binding, keypoints: dict[str, Any]) -> tuple[float, float, float] | None:
    """Return the mean anchor position and the weakest contributing confidence."""
    xs: list[float] = []
    ys: list[float] = []
    confidences: list[float] = []
    for name in binding.anchor.landmarks:
        point = keypoints.get(name)
        if point is None:
            return None
        xs.append(float(point["x"]))
        ys.append(float(point["y"]))
        confidences.append(float(point["visibility"]))
    return sum(xs) / len(xs), sum(ys) / len(ys), min(confidences)


def _size_pixels(binding: Binding, bbox: dict[str, Any], width: int, height: int) -> float:
    bbox_width = max(0.0, float(bbox.get("width", 0.0))) * width
    bbox_height = max(0.0, float(bbox.get("height", 0.0))) * height
    if binding.scale_mode == "absolute-px":
        return binding.scale_value
    if binding.scale_mode == "bbox-width":
        reference = bbox_width
    elif binding.scale_mode == "bbox-height":
        reference = bbox_height
    else:
        reference = math.hypot(bbox_width, bbox_height)
    return binding.scale_value * reference


def _rotation_degrees(binding: Binding, keypoints: dict[str, Any], width: int, height: int) -> float:
    if binding.rotate_mode == "fixed":
        return binding.rotate_degrees
    if binding.rotate_mode != "landmark-pair":
        return 0.0
    start = keypoints.get(str(binding.rotate_from))
    end = keypoints.get(str(binding.rotate_to))
    if not start or not end:
        return 0.0
    delta_x = (float(end["x"]) - float(start["x"])) * width
    delta_y = (float(end["y"]) - float(start["y"])) * height
    if delta_x == 0.0 and delta_y == 0.0:
        return 0.0
    return math.degrees(math.atan2(delta_y, delta_x)) + binding.rotate_degrees


def _opacity_for(binding: Binding, confidence: float, has_hold_position: bool) -> tuple[float, str]:
    if confidence >= _TRACKED_CONFIDENCE:
        return 1.0, "tracked"
    if binding.on_low_confidence == "hide":
        return 0.0, "hidden-low-confidence"
    if binding.on_low_confidence == "hold":
        if confidence > _HIDDEN_CONFIDENCE or has_hold_position:
            return 1.0, "held-low-confidence"
        return 0.0, "hidden-low-confidence"
    if confidence <= _HIDDEN_CONFIDENCE:
        return 0.0, "hidden-low-confidence"
    span = _TRACKED_CONFIDENCE - _HIDDEN_CONFIDENCE
    return round((confidence - _HIDDEN_CONFIDENCE) / span, 4), "faded-low-confidence"


def resolve_bindings(
    bindings: list[Binding], render_ir: dict[str, Any], width: int, height: int
) -> dict[str, Any]:
    """Project every binding onto the render-time observation grid."""
    track = next((item for item in render_ir.get("tracks", []) if item.get("type") == "pose"), None)
    if track is None:
        raise RuntimeError("Render IR has no pose track to bind assets against.")
    observations = track.get("observations", [])
    if not observations:
        raise RuntimeError("Render IR has no observations to bind assets against.")

    resolved: list[dict[str, Any]] = []
    for binding in bindings:
        frames: list[dict[str, Any]] = []
        state_counts: dict[str, int] = {}
        last_position: tuple[float, float] | None = None
        last_time_seconds: float | None = None
        clamped_frames = 0
        missing_landmark_frames = 0

        for observation in observations:
            timestamp_ms = float(observation["sourceTimeMs"])
            time_seconds = timestamp_ms / 1000.0
            keypoints = {str(point["name"]): point for point in observation["screenSpace"]["keypoints"]}
            anchor = _anchor_state(binding, keypoints)
            in_range = timestamp_ms >= binding.start_ms and (
                binding.end_ms is None or timestamp_ms <= binding.end_ms
            )

            if anchor is None:
                missing_landmark_frames += 1
                state_counts["missing-landmark"] = state_counts.get("missing-landmark", 0) + 1
                frames.append(
                    {
                        "t": round(time_seconds, 6),
                        "x": round(last_position[0], 6) if last_position else 0.5,
                        "y": round(last_position[1], 6) if last_position else 0.5,
                        "size": 0.0,
                        "rotation": 0.0,
                        "opacity": 0.0,
                        "confidence": 0.0,
                        "state": "missing-landmark",
                    }
                )
                continue

            anchor_x, anchor_y, confidence = anchor
            size_pixels = _size_pixels(binding, observation["screenSpace"].get("bbox", {}), width, height)
            if binding.offset_mode == "size-relative":
                position_x = anchor_x + binding.offset_x * size_pixels / width
                position_y = anchor_y + binding.offset_y * size_pixels / height
            else:
                position_x = anchor_x + binding.offset_x
                position_y = anchor_y + binding.offset_y

            opacity, state = _opacity_for(binding, confidence, last_position is not None)
            if state == "held-low-confidence" and last_position is not None:
                position_x, position_y = last_position

            if last_position is not None and last_time_seconds is not None and opacity > 0:
                delta_seconds = max(1e-4, time_seconds - last_time_seconds)
                travel = math.hypot(position_x - last_position[0], position_y - last_position[1])
                allowed = binding.max_speed * delta_seconds
                if travel > allowed:
                    ratio = allowed / travel
                    position_x = last_position[0] + (position_x - last_position[0]) * ratio
                    position_y = last_position[1] + (position_y - last_position[1]) * ratio
                    clamped_frames += 1

            if not in_range:
                opacity, state = 0.0, "out-of-range"

            if opacity > 0:
                last_position = (position_x, position_y)
                last_time_seconds = time_seconds

            state_counts[state] = state_counts.get(state, 0) + 1
            frames.append(
                {
                    "t": round(time_seconds, 6),
                    "x": round(position_x, 6),
                    "y": round(position_y, 6),
                    "size": round(size_pixels, 3),
                    "rotation": round(_rotation_degrees(binding, keypoints, width, height), 4),
                    "opacity": round(opacity, 4),
                    "confidence": round(confidence, 6),
                    "state": state,
                }
            )

        visible = [frame for frame in frames if frame["opacity"] > 0]
        resolved.append(
            {
                "id": binding.id,
                "kind": binding.kind,
                "text": binding.text,
                "source": binding.source,
                "aspect": binding.aspect,
                "style": binding.style,
                "anchorLandmarks": list(binding.anchor.landmarks),
                "policy": {
                    "onLowConfidence": binding.on_low_confidence,
                    "maxSpeed": binding.max_speed,
                    "scaleMode": binding.scale_mode,
                    "scaleValue": binding.scale_value,
                    "offsetMode": binding.offset_mode,
                    "rotateMode": binding.rotate_mode,
                },
                "stats": {
                    "frames": len(frames),
                    "visibleFrames": len(visible),
                    "clampedFrames": clamped_frames,
                    "missingLandmarkFrames": missing_landmark_frames,
                    "states": state_counts,
                    "meanConfidence": round(sum(frame["confidence"] for frame in frames) / len(frames), 6)
                    if frames
                    else 0.0,
                },
                "frames": frames,
            }
        )

    return {
        "schemaVersion": SCHEMA_VERSION,
        "coordinateSpace": {
            "unit": "normalized-screen",
            "pixelWidth": width,
            "pixelHeight": height,
            "note": "size and rotation are expressed in source pixel space; x and y stay normalized.",
        },
        "bindings": resolved,
    }
