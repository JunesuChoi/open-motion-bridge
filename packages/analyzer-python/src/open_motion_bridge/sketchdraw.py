"""Sketch drawing: vectorize a photo into ordered strokes drawn like a human hand,
optionally followed by a human-like coloring pass and a close-up camera that
follows the pen.

Pipeline: edges -> polyline strokes -> coarse-to-fine ordering with
nearest-neighbour pen travel -> per-stroke timing. Coloring reveals the source
photo through deterministic brush-path masks: broad alternating wash strokes
first, then short edge-aligned detail strokes. The close-up camera follows the
precomputed pen/brush position with smoothing, speed limiting, and edge
clamping. Everything — ink strokes, brush strokes, tool positions, and the
camera table — is written to `sketch.plan.json` so the result stays
auditable and the composition stays deterministic and seek-safe. Rendering is
external; this module only generates the project.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .pipeline import SCHEMA_VERSION, _sha256, _utc_now, _write_json


def _extract_strokes(
    image: np.ndarray,
    max_dimension: int,
    canny_low: int,
    canny_high: int,
    min_stroke_px: float,
    epsilon_px: float,
) -> tuple[list[list[tuple[float, float]]], np.ndarray]:
    height, width = image.shape[0], image.shape[1]
    scale = min(1.0, max_dimension / max(width, height))
    work = (
        cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else image
    )
    work_h, work_w = work.shape[0], work.shape[1]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    strokes: list[list[tuple[float, float]]] = []
    for contour in contours:
        if cv2.arcLength(contour, False) < min_stroke_px:
            continue
        approx = cv2.approxPolyDP(contour, epsilon_px, False)
        points = [(float(p[0][0]) / work_w, float(p[0][1]) / work_h) for p in approx]
        if len(points) >= 2:
            strokes.append(points)
    return strokes, work


def _stroke_length(points: list[tuple[float, float]], width: int, height: int) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot((x1 - x0) * width, (y1 - y0) * height)
    return total


def _order_strokes(
    strokes: list[list[tuple[float, float]]], width: int, height: int
) -> list[dict[str, Any]]:
    """Coarse-to-fine ordering with nearest-neighbour pen travel inside each pass."""
    measured = [
        {"points": s, "lengthPx": _stroke_length(s, width, height)} for s in strokes
    ]
    measured.sort(key=lambda item: -item["lengthPx"])
    if not measured:
        return []
    cut = max(1, int(len(measured) * 0.3))
    passes = [measured[:cut], measured[cut:]]

    ordered: list[dict[str, Any]] = []
    pen = (0.5, 0.0)
    for group in passes:
        remaining = list(group)
        while remaining:
            best_index, best_cost, best_reversed = 0, float("inf"), False
            for index, item in enumerate(remaining):
                head, tail = item["points"][0], item["points"][-1]
                cost_head = math.hypot(head[0] - pen[0], head[1] - pen[1])
                cost_tail = math.hypot(tail[0] - pen[0], tail[1] - pen[1])
                if cost_head < best_cost:
                    best_index, best_cost, best_reversed = index, cost_head, False
                if cost_tail < best_cost:
                    best_index, best_cost, best_reversed = index, cost_tail, True
            chosen = remaining.pop(best_index)
            points = (
                list(reversed(chosen["points"])) if best_reversed else chosen["points"]
            )
            ordered.append({"points": points, "lengthPx": chosen["lengthPx"]})
            pen = points[-1]
    return ordered


def _schedule(
    ordered: list[dict[str, Any]],
    start_ms: float,
    draw_ms: float,
    gap_ms: float,
    prefix: str,
) -> list[dict[str, Any]]:
    total_length = sum(item["lengthPx"] for item in ordered) or 1.0
    budget = draw_ms - gap_ms * max(0, len(ordered) - 1)
    if budget <= 0:
        raise ValueError("Phase duration too short for the element count.")
    cursor = start_ms
    scheduled: list[dict[str, Any]] = []
    for index, item in enumerate(ordered):
        duration = max(16.0, budget * item["lengthPx"] / total_length)
        entry = {
            "id": f"{prefix}-{index:04d}",
            "startMs": round(cursor, 2),
            "durationMs": round(duration, 2),
            "lengthPx": round(item["lengthPx"], 2),
            "points": [[round(x, 5), round(y, 5)] for x, y in item["points"]],
        }
        scheduled.append(entry)
        cursor += duration + gap_ms
    return scheduled


def _noise01(*values: int) -> float:
    """Small stable integer hash for authored-looking, reproducible variation."""
    state = 2166136261
    for value in values:
        state ^= value & 0xFFFFFFFF
        state = (state * 16777619) & 0xFFFFFFFF
    return state / 0xFFFFFFFF


def _schedule_brush_pass(
    strokes: list[dict[str, Any]], start_ms: float, duration_ms: float, start_index: int
) -> None:
    if not strokes:
        return
    lengths = [math.hypot(s["x1"] - s["x0"], s["y1"] - s["y0"]) for s in strokes]
    total = sum(lengths) or 1.0
    floor_ms = min(24.0, duration_ms / len(strokes) * 0.45)
    weighted_budget = max(0.0, duration_ms - floor_ms * len(strokes))
    cursor = start_ms
    for offset, (stroke, length) in enumerate(zip(strokes, lengths, strict=True)):
        stroke_duration = floor_ms + weighted_budget * length / total
        stroke["id"] = f"paint-{start_index + offset:04d}"
        stroke["startMs"] = round(cursor, 2)
        stroke["durationMs"] = round(stroke_duration, 2)
        cursor += stroke_duration


def _paint_brush_strokes(
    work: np.ndarray, start_ms: float, paint_ms: float
) -> list[dict[str, Any]]:
    """Create a two-pass human-readable brush plan, not a field of circle stamps.

    The broad pass alternates left-to-right and right-to-left full-row sweeps.
    The detail pass selects high-gradient cells and aligns each short stroke with
    the local edge tangent. Color is sampled only for audit metadata; rendering
    reveals the source pixels through these paths so the result keeps real color
    continuity instead of becoming a mosaic.
    """
    height, width = work.shape[0], work.shape[1]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    sample_image = cv2.GaussianBlur(work, (0, 0), 2.2)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    wash: list[dict[str, Any]] = []
    spacing = max(22, width // 30)
    row_count = max(1, math.ceil(height / spacing))
    for row in range(row_count):
        cy = min(height - 1.0, (row + 0.5) * height / row_count)
        jitter_y = (_noise01(row, 11) - 0.5) * spacing * 0.32
        slope = (_noise01(row, 17) - 0.5) * spacing * 1.5
        curve = (_noise01(row, 19) - 0.5) * spacing * 1.2
        left_to_right = row % 2 == 0
        x0, x1 = (
            (-0.025 * width, 1.025 * width)
            if left_to_right
            else (1.025 * width, -0.025 * width)
        )
        y0 = min(height - 1.0, max(0.0, cy + jitter_y - slope / 2))
        y1 = min(height - 1.0, max(0.0, cy + jitter_y + slope / 2))
        sx, sy = int(min(width - 1, max(0, (x0 + x1) / 2))), int((y0 + y1) / 2)
        b, g, r = sample_image[sy, sx]
        wash.append(
            {
                "pass": "wash",
                "x0": round(x0 / width, 5),
                "y0": round(y0 / height, 5),
                "x1": round(x1 / width, 5),
                "y1": round(y1 / height, 5),
                "cx": 0.5,
                "cy": round(
                    min(height - 1.0, max(0.0, cy + jitter_y + curve)) / height, 5
                ),
                "width": round(spacing * 2.05 / width, 5),
                "opacity": 1.0,
                "sampledColor": f"#{int(r):02x}{int(g):02x}{int(b):02x}",
            }
        )

    detail_cell = max(14, width // 52)
    candidates: list[tuple[float, int, int]] = []
    for cy in range(detail_cell // 2, height, detail_cell):
        for cx in range(detail_cell // 2, width, detail_cell):
            y0, y1 = max(0, cy - detail_cell // 2), min(height, cy + detail_cell // 2)
            x0, x1 = max(0, cx - detail_cell // 2), min(width, cx + detail_cell // 2)
            score = (
                float(magnitude[y0:y1, x0:x1].mean())
                + float(gray[y0:y1, x0:x1].std()) * 1.8
            )
            candidates.append((score, cx, cy))
    max_detail_for_time = max(12, int(paint_ms * 0.32 / 35.0))
    detail_limit = min(96, max_detail_for_time, max(12, len(candidates) // 18))
    selected = sorted(candidates, key=lambda item: (-item[0], item[2], item[1]))[
        :detail_limit
    ]
    selected.sort(
        key=lambda item: (
            item[2] // detail_cell,
            item[1] if (item[2] // detail_cell) % 2 == 0 else -item[1],
        )
    )

    detail: list[dict[str, Any]] = []
    for index, (_, cx, cy) in enumerate(selected):
        gx, gy = float(grad_x[cy, cx]), float(grad_y[cy, cx])
        angle = (
            math.atan2(gy, gx) + math.pi / 2.0
            if abs(gx) + abs(gy) > 1e-6
            else (_noise01(index, 23) - 0.5) * math.pi
        )
        length = detail_cell * (1.5 + _noise01(index, 29) * 1.25)
        dx, dy = math.cos(angle) * length / 2.0, math.sin(angle) * length / 2.0
        b, g, r = sample_image[cy, cx]
        detail.append(
            {
                "pass": "detail",
                "x0": round(min(width - 1.0, max(0.0, cx - dx)) / width, 5),
                "y0": round(min(height - 1.0, max(0.0, cy - dy)) / height, 5),
                "x1": round(min(width - 1.0, max(0.0, cx + dx)) / width, 5),
                "y1": round(min(height - 1.0, max(0.0, cy + dy)) / height, 5),
                "width": round(
                    detail_cell * (0.48 + _noise01(index, 31) * 0.25) / width, 5
                ),
                "opacity": 1.0,
                "sampledColor": f"#{int(r):02x}{int(g):02x}{int(b):02x}",
            }
        )

    wash_ms = paint_ms * 0.68
    _schedule_brush_pass(wash, start_ms, wash_ms, 0)
    _schedule_brush_pass(detail, start_ms + wash_ms, paint_ms - wash_ms, len(wash))
    return wash + detail


def _position_table(
    strokes: list[dict[str, Any]],
    paint_strokes: list[dict[str, Any]],
    duration_ms: float,
    fps: float,
) -> list[dict[str, Any]]:
    """Precompute an interpolated pen/brush location for each sample."""
    events: list[dict[str, Any]] = []
    for stroke in strokes:
        events.append(
            {
                "start": float(stroke["startMs"]),
                "duration": float(stroke["durationMs"]),
                "points": stroke["points"],
                "phase": "ink",
            }
        )
    for stroke in paint_strokes:
        events.append(
            {
                "start": float(stroke["startMs"]),
                "duration": float(stroke["durationMs"]),
                "points": [[stroke["x0"], stroke["y0"]], [stroke["x1"], stroke["y1"]]],
                "control": [stroke["cx"], stroke["cy"]] if "cx" in stroke else None,
                "phase": "paint",
            }
        )
    events.sort(key=lambda e: e["start"])

    table: list[dict[str, Any]] = []
    cursor = 0
    last = (0.5, 0.5)
    phase = "ink"
    count = max(1, int(duration_ms / 1000.0 * fps) + 1)
    for i in range(count):
        t = i * 1000.0 / fps
        while (
            cursor < len(events)
            and events[cursor]["start"] + events[cursor]["duration"] < t
        ):
            last_points = events[cursor]["points"]
            last = (float(last_points[-1][0]), float(last_points[-1][1]))
            phase = str(events[cursor]["phase"])
            cursor += 1
        if cursor < len(events) and events[cursor]["start"] <= t:
            event = events[cursor]
            progress = max(
                0.0, min(1.0, (t - event["start"]) / max(1.0, event["duration"]))
            )
            points = event["points"]
            if event.get("control") is not None:
                x0, y0 = map(float, points[0])
                x1, y1 = map(float, points[-1])
                control_x, control_y = map(float, event["control"])
                inverse = 1.0 - progress
                last = (
                    inverse * inverse * x0
                    + 2.0 * inverse * progress * control_x
                    + progress * progress * x1,
                    inverse * inverse * y0
                    + 2.0 * inverse * progress * control_y
                    + progress * progress * y1,
                )
            else:
                segment_position = progress * max(1, len(points) - 1)
                segment = min(len(points) - 2, int(segment_position))
                local = segment_position - segment
                x0, y0 = points[segment]
                x1, y1 = points[segment + 1]
                last = (
                    float(x0) + (float(x1) - float(x0)) * local,
                    float(y0) + (float(y1) - float(y0)) * local,
                )
            phase = str(event["phase"])
        table.append(
            {
                "t": round(t / 1000.0, 4),
                "x": round(last[0], 5),
                "y": round(last[1], 5),
                "phase": phase,
            }
        )
    return table


def _camera_table(
    positions: list[dict[str, Any]],
    zoom: float,
    duration_ms: float,
    release_ms: float,
    mode: str,
) -> list[dict[str, float]]:
    """Smooth, speed-limited camera that eases back to the full image."""
    table: list[dict[str, float]] = []
    cx, cy = 0.5, 0.5
    alpha = 0.065 if mode == "pen-follow" else 0.04
    max_step = 0.018 if mode == "pen-follow" else 0.010
    phase_targets: dict[int, tuple[float, float]] = {}
    if mode == "phase-focus":
        buckets: dict[int, list[tuple[float, float]]] = {}
        for position in positions:
            bucket = int(float(position["t"]) / 1.15)
            buckets.setdefault(bucket, []).append(
                (float(position["x"]), float(position["y"]))
            )
        phase_targets = {
            bucket: (
                sum(x for x, _ in values) / len(values),
                sum(y for _, y in values) / len(values),
            )
            for bucket, values in buckets.items()
        }
    for position in positions:
        t_ms = float(position["t"]) * 1000.0
        if mode == "phase-focus":
            bucket_float = float(position["t"]) / 1.15
            bucket = int(bucket_float)
            blend = bucket_float - bucket
            smooth = blend * blend * (3.0 - 2.0 * blend)
            a = phase_targets.get(bucket, (float(position["x"]), float(position["y"])))
            b = phase_targets.get(bucket + 1, a)
            px, py = a[0] + (b[0] - a[0]) * smooth, a[1] + (b[1] - a[1]) * smooth
        else:
            px, py = float(position["x"]), float(position["y"])
        dx, dy = (px - cx) * alpha, (py - cy) * alpha
        distance = math.hypot(dx, dy)
        if distance > max_step:
            scale = max_step / distance
            dx, dy = dx * scale, dy * scale
        cx += dx
        cy += dy
        ramp_in = min(1.0, t_ms / 900.0)
        release = max(
            0.0, min(1.0, (t_ms - (duration_ms - release_ms)) / max(1.0, release_ms))
        )
        eased_release = release * release * (3 - 2 * release)
        s = 1.0 + (zoom - 1.0) * ramp_in * (1.0 - eased_release)
        half = 0.5 / s
        ccx = min(1.0 - half, max(half, cx))
        ccy = min(1.0 - half, max(half, cy))
        fcx = 0.5 + (ccx - 0.5) * (1.0 - eased_release)
        fcy = 0.5 + (ccy - 0.5) * (1.0 - eased_release)
        table.append(
            {
                "t": round(t_ms / 1000.0, 4),
                "s": round(s, 5),
                "cx": round(fcx, 5),
                "cy": round(fcy, 5),
            }
        )
    return table


def generate_sketch_project(
    source_image: Path,
    output_dir: Path,
    duration_ms: float = 10000.0,
    fps: float = 30.0,
    hold_ms: float = 2000.0,
    photo_fade_ms: float = 1500.0,
    max_strokes: int = 900,
    color_mode: str = "none",
    closeup_mode: str | None = None,
    closeup_zoom: float = 0.0,
    force: bool = False,
) -> dict[str, Any]:
    if not source_image.is_file():
        raise FileNotFoundError(f"Source image not found: {source_image}")
    if color_mode not in {"none", "paint"}:
        raise ValueError("color_mode must be 'none' or 'paint'")
    resolved_closeup_mode = closeup_mode or ("pen-follow" if closeup_zoom else "none")
    if resolved_closeup_mode not in {"none", "pen-follow", "phase-focus"}:
        raise ValueError("closeup_mode must be 'none', 'pen-follow', or 'phase-focus'")
    if resolved_closeup_mode != "none" and closeup_zoom == 0:
        closeup_zoom = 1.7
    if resolved_closeup_mode == "none" and closeup_zoom:
        raise ValueError(
            "closeup_zoom requires closeup_mode 'pen-follow' or 'phase-focus'"
        )
    if closeup_zoom and not 1.0 < closeup_zoom <= 3.0:
        raise ValueError("closeup_zoom must be within (1.0, 3.0] or 0 to disable")
    if duration_ms <= hold_ms + photo_fade_ms:
        raise ValueError(
            "durationMs must exceed holdMs + photoFadeMs so drawing time remains."
        )
    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode image: {source_image}")
    height, width = image.shape[0], image.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    strokes, work = _extract_strokes(
        image,
        max_dimension=1400,
        canny_low=60,
        canny_high=160,
        min_stroke_px=28.0,
        epsilon_px=1.2,
    )
    work_h, work_w = work.shape[0], work.shape[1]
    ordered = _order_strokes(strokes, work_w, work_h)[:max_strokes]
    if not ordered:
        raise RuntimeError(
            "No drawable strokes were extracted; adjust edge thresholds."
        )

    active_ms = duration_ms - hold_ms - photo_fade_ms
    draw_ms = active_ms * (0.45 if color_mode == "paint" else 1.0)
    paint_ms = active_ms - draw_ms if color_mode == "paint" else 0.0
    gap_ms = min(28.0, draw_ms * 0.15 / max(1, len(ordered)))
    scheduled = _schedule(ordered, 0.0, draw_ms, gap_ms, "stroke")
    paint_strokes = (
        _paint_brush_strokes(work, draw_ms, paint_ms) if color_mode == "paint" else []
    )

    camera: list[dict[str, float]] = []
    positions = _position_table(scheduled, paint_strokes, duration_ms, fps)
    if resolved_closeup_mode != "none":
        camera = _camera_table(
            positions,
            closeup_zoom,
            duration_ms,
            release_ms=hold_ms + photo_fade_ms,
            mode=resolved_closeup_mode,
        )

    plan = {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": _utc_now(),
        "source": {
            "sourceHash": _sha256(source_image),
            "fileName": source_image.name,
            "displayWidth": width,
            "displayHeight": height,
        },
        "timing": {
            "durationMs": duration_ms,
            "drawMs": round(draw_ms, 2),
            "paintMs": round(paint_ms, 2),
            "photoFadeMs": photo_fade_ms,
            "holdMs": hold_ms,
            "fps": fps,
            "penGapMs": round(gap_ms, 3),
        },
        "vectorization": {
            "workingSize": [work_w, work_h],
            "strokeCount": len(scheduled),
            "totalInkPx": round(sum(s["lengthPx"] for s in scheduled), 1),
            "ordering": "coarse-to-fine, nearest-neighbour pen travel",
        },
        "coloring": {
            "mode": color_mode,
            "brushStrokeCount": len(paint_strokes),
            "passes": ["alternating broad wash", "edge-aligned detail"],
            "rendering": "source image revealed through progressive brush-path mask",
        },
        "camera": {
            "mode": resolved_closeup_mode,
            "zoom": closeup_zoom,
            "samples": len(camera),
            "speedLimited": bool(camera),
        },
        "strokes": scheduled,
        "paintStrokes": paint_strokes,
        "toolTable": positions,
        "cameraTable": camera,
    }
    _write_json(output_dir / "sketch.plan.json", plan, force)

    assets = output_dir / "assets"
    assets.mkdir(exist_ok=True)
    staged = assets / ("source" + source_image.suffix.lower())
    if staged.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {staged}; use --force for generated output."
        )
    import shutil

    shutil.copy2(source_image, staged)

    html_path = output_dir / "index.html"
    if html_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {html_path}; use --force for generated output."
        )
    html_path.write_text(_render_sketch_html(plan, staged.name), encoding="utf-8")
    _write_json(
        output_dir / "index.motion.json",
        {
            "duration": round(duration_ms / 1000.0, 3),
            "assertions": [
                {"kind": "appearsBy", "selector": "#tool", "bySec": 0.2},
                {
                    "kind": "keepsMoving",
                    "withinSelector": "#scene",
                    "maxStaticSec": round((hold_ms + photo_fade_ms) / 1000.0 + 0.2, 3),
                },
            ],
        },
        force,
    )
    _write_json(
        output_dir / "open-motion-bridge.generated.json",
        {
            "schemaVersion": SCHEMA_VERSION,
            "createdAt": _utc_now(),
            "sourceHash": plan["source"]["sourceHash"],
            "mediaKind": "image",
            "target": "hyperframes-sketch",
            "compositionDurationMs": duration_ms,
            "renderFps": fps,
            "strokeCount": len(scheduled),
            "paintBrushStrokeCount": len(paint_strokes),
            "closeupMode": resolved_closeup_mode,
            "closeupZoom": closeup_zoom,
            "motionSidecar": "index.motion.json",
        },
        force,
    )
    return plan


def _render_sketch_html(plan: dict[str, Any], staged_image_name: str) -> str:
    width = int(plan["source"]["displayWidth"])
    height = int(plan["source"]["displayHeight"])
    timing = plan["timing"]
    duration = float(timing["durationMs"]) / 1000.0
    fps = float(timing["fps"])
    draw_end = float(timing["drawMs"]) + float(timing["paintMs"])
    fade_end = draw_end + float(timing["photoFadeMs"])

    path_elements: list[str] = []
    stroke_meta: list[dict[str, Any]] = []
    for stroke in plan["strokes"]:
        d = "M " + " L ".join(
            f"{x * width:.2f} {y * height:.2f}" for x, y in stroke["points"]
        )
        path_elements.append(f'<path id="{stroke["id"]}" d="{d}" />')
        stroke_meta.append(
            {"id": stroke["id"], "s": stroke["startMs"], "d": stroke["durationMs"]}
        )
    paths_markup = "".join(path_elements)
    meta_json = json.dumps(stroke_meta, separators=(",", ":"))
    paint_json = json.dumps(
        [
            {
                "x0": s["x0"],
                "y0": s["y0"],
                "x1": s["x1"],
                "y1": s["y1"],
                "cx": s.get("cx"),
                "cy": s.get("cy"),
                "w": s["width"],
                "a": s["opacity"],
                "p": s["pass"],
                "s": s["startMs"],
                "d": s["durationMs"],
            }
            for s in plan["paintStrokes"]
        ],
        separators=(",", ":"),
    )
    tool_json = json.dumps(plan["toolTable"], separators=(",", ":"))
    camera_json = json.dumps(plan["cameraTable"], separators=(",", ":"))

    return f'''<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={width}, height={height}" />
    <title>Open Motion Bridge sketch drawing</title>
    <script src="./node_modules/gsap/dist/gsap.min.js"></script>
    <style>
      * {{ box-sizing: border-box; }}
      html, body {{ margin: 0; width: {width}px; height: {height}px; overflow: hidden; background: #f7f2e9; }}
      #root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; }}
      .clip {{ position: absolute; inset: 0; }}
      #scene {{ position: absolute; inset: 0; overflow: hidden; }}
      #camera-world {{ position: absolute; left: 0; top: 0; width: {width}px; height: {height}px; transform-origin: 0 0; will-change: transform; }}
      #paper {{ position: absolute; inset: 0; background:
        radial-gradient(circle at 30% 20%, rgba(255,255,255,0.9), rgba(0,0,0,0) 60%),
        repeating-linear-gradient(0deg, rgba(0,0,0,0.012) 0 2px, rgba(0,0,0,0) 2px 4px), #f7f2e9; }}
      #paint, #paint-detail {{ position: absolute; inset: 0; }}
      #paint {{ opacity: 0.82; }}
      #paint-detail {{ opacity: 0.36; }}
      #photo {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0; }}
      #ink {{ position: absolute; inset: 0; }}
      #ink path {{ fill: none; stroke: #2b2620; stroke-width: 2.1; stroke-linecap: round; stroke-linejoin: round; opacity: 0.92; }}
      #tool {{
        position: absolute; left: 0; top: 0; width: 14px; height: 54px;
        border-radius: 7px 7px 3px 3px; transform-origin: 50% 100%; pointer-events: none;
        background: linear-gradient(90deg, #c18c52 0 28%, #e4b878 28% 72%, #a86f3f 72%);
        box-shadow: 0 4px 10px rgba(54, 38, 25, 0.24); opacity: 0;
      }}
      #tool::after {{
        content: ''; position: absolute; left: 2px; right: 2px; bottom: -12px; height: 15px;
        background: #302922; clip-path: polygon(0 0, 100% 0, 50% 100%);
      }}
      #tool[data-phase='paint'] {{ width: 20px; background: linear-gradient(90deg, #6f5844 0 22%, #b99b75 22% 78%, #634b39 78%); }}
      #tool[data-phase='paint']::after {{ bottom: -9px; height: 12px; background: #7d5a43; clip-path: polygon(0 0, 100% 0, 82% 100%, 18% 100%); }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{duration:.3f}" data-width="{width}" data-height="{height}" data-fps="{fps:g}">
      <div id="paper"></div>
      <div id="scene" class="clip" data-start="0" data-duration="{duration:.3f}" data-track-index="1">
        <div id="camera-world" data-layout-allow-overflow>
          <canvas id="paint" width="{width}" height="{height}"></canvas>
          <canvas id="paint-detail" width="{width}" height="{height}"></canvas>
          <img id="photo" src="assets/{staged_image_name}" alt="" />
          <svg id="ink" viewBox="0 0 {width} {height}">{paths_markup}</svg>
          <div id="tool" data-phase="ink"></div>
        </div>
      </div>
    </div>
    <script>
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true }});
      const W = {width}, H = {height};
      const strokes = {meta_json};
      const paintStrokes = {paint_json};
      const toolFrames = {tool_json};
      const cameraFrames = {camera_json};
      const DRAW_END = {draw_end:.2f}, FADE_END = {fade_end:.2f};
      const nodes = strokes.map((s) => {{
        const node = document.getElementById(s.id);
        const len = node.getTotalLength();
        node.style.strokeDasharray = String(len);
        node.style.strokeDashoffset = String(len);
        return {{ meta: s, node, len }};
      }});
      const photo = document.getElementById('photo');
      const paintCanvas = document.getElementById('paint');
      const ctx = paintCanvas.getContext('2d');
      const detailPaintCanvas = document.getElementById('paint-detail');
      const detailCtx = detailPaintCanvas.getContext('2d');
      const cameraNode = document.getElementById('camera-world');
      const toolNode = document.getElementById('tool');
      const drawPaintLayer = (targetCtx, passName, t) => {{
        targetCtx.clearRect(0, 0, W, H);
        let hasPaint = false;
        targetCtx.globalCompositeOperation = 'source-over';
        targetCtx.strokeStyle = '#ffffff';
        targetCtx.lineCap = 'round';
        targetCtx.lineJoin = 'round';
        for (const s of paintStrokes) {{
          if (s.p !== passName) continue;
          const p = Math.max(0, Math.min(1, (t - s.s) / s.d));
          if (p <= 0) continue;
          hasPaint = true;
          const e = p * p * (3 - 2 * p);
          const x0 = s.x0 * W, y0 = s.y0 * H;
          let x1, y1, partialCx = null, partialCy = null;
          if (s.cx !== null && s.cy !== null) {{
            const controlX = s.cx * W, controlY = s.cy * H;
            partialCx = x0 + (controlX - x0) * e;
            partialCy = y0 + (controlY - y0) * e;
            const nextCx = controlX + (s.x1 * W - controlX) * e;
            const nextCy = controlY + (s.y1 * H - controlY) * e;
            x1 = partialCx + (nextCx - partialCx) * e;
            y1 = partialCy + (nextCy - partialCy) * e;
          }} else {{
            x1 = (s.x0 + (s.x1 - s.x0) * e) * W;
            y1 = (s.y0 + (s.y1 - s.y0) * e) * H;
          }}
          const brushWidth = Math.max(2, s.w * W);
          targetCtx.globalAlpha = s.a;
          targetCtx.lineWidth = brushWidth;
          targetCtx.shadowBlur = brushWidth * (s.p === 'wash' ? 0.24 : 0.12);
          targetCtx.shadowColor = 'rgba(255,255,255,0.55)';
          targetCtx.beginPath();
          targetCtx.moveTo(x0, y0);
          if (partialCx !== null) targetCtx.quadraticCurveTo(partialCx, partialCy, x1, y1);
          else targetCtx.lineTo(x1, y1);
          targetCtx.stroke();
          const length = Math.max(1, Math.hypot(x1 - x0, y1 - y0));
          const nx = -(y1 - y0) / length, ny = (x1 - x0) / length;
          for (const side of [-1, 1]) {{
            const offset = brushWidth * 0.26 * side;
            targetCtx.globalAlpha = s.a * 0.34;
            targetCtx.lineWidth = brushWidth * 0.12;
            targetCtx.shadowBlur = 0;
            targetCtx.beginPath();
            targetCtx.moveTo(x0 + nx * offset, y0 + ny * offset);
            targetCtx.lineTo(x1 + nx * offset, y1 + ny * offset);
            targetCtx.stroke();
          }}
        }}
        targetCtx.shadowBlur = 0;
        if (hasPaint && photo.complete && photo.naturalWidth > 0) {{
          targetCtx.globalCompositeOperation = 'source-in';
          targetCtx.globalAlpha = 1;
          targetCtx.drawImage(photo, 0, 0, W, H);
        }}
        targetCtx.globalCompositeOperation = 'source-over';
        targetCtx.globalAlpha = 1;
      }};
      const drawPaint = (t) => {{
        drawPaintLayer(ctx, 'wash', t);
        drawPaintLayer(detailCtx, 'detail', t);
      }};
      const frameAt = (frames, timeSec) => {{
        if (!frames.length) return null;
        let lo = 0, hi = frames.length - 1;
        while (lo < hi) {{ const mid = Math.ceil((lo + hi) / 2); if (frames[mid].t <= timeSec) lo = mid; else hi = mid - 1; }}
        const a = frames[lo], b = frames[Math.min(frames.length - 1, lo + 1)];
        const mix = b.t > a.t ? Math.max(0, Math.min(1, (timeSec - a.t) / (b.t - a.t))) : 0;
        return {{ a, b, mix }};
      }};
      const applyCamera = (timeSec) => {{
        const sample = frameAt(cameraFrames, timeSec);
        if (!sample) {{ cameraNode.style.transform = 'translate(0px,0px) scale(1)'; return; }}
        const {{ a, b, mix }} = sample;
        const s = a.s + (b.s - a.s) * mix;
        const cx = a.cx + (b.cx - a.cx) * mix;
        const cy = a.cy + (b.cy - a.cy) * mix;
        cameraNode.style.transform = 'translate(' + ((0.5 - cx * s) * W).toFixed(3) + 'px,' + ((0.5 - cy * s) * H).toFixed(3) + 'px) scale(' + s.toFixed(6) + ')';
      }};
      const applyTool = (timeSec, timeMs) => {{
        const sample = frameAt(toolFrames, timeSec);
        if (!sample) return;
        const {{ a, b, mix }} = sample;
        const x = a.x + (b.x - a.x) * mix;
        const y = a.y + (b.y - a.y) * mix;
        const phase = mix < 0.5 ? a.phase : b.phase;
        const fade = Math.max(0, Math.min(1, (DRAW_END - timeMs) / 280));
        toolNode.dataset.phase = phase;
        toolNode.style.opacity = String(fade);
        toolNode.style.transform = 'translate(' + (x * W).toFixed(2) + 'px,' + (y * H).toFixed(2) + 'px) translate(-50%,-100%) rotate(' + (phase === 'paint' ? 28 : 34) + 'deg)';
      }};
      const draw = (timeSec) => {{
        const t = timeSec * 1000;
        for (const entry of nodes) {{
          const p = Math.max(0, Math.min(1, (t - entry.meta.s) / entry.meta.d));
          entry.node.style.strokeDashoffset = String(entry.len * (1 - p));
        }}
        drawPaint(t);
        const fade = Math.max(0, Math.min(1, (t - DRAW_END) / Math.max(1, FADE_END - DRAW_END)));
        const eased = fade * fade * (3 - 2 * fade);
        photo.style.opacity = String(eased);
        document.getElementById('ink').style.opacity = String(1 - eased * 0.85);
        applyCamera(timeSec);
        applyTool(timeSec, t);
      }};
      const state = {{ time: 0 }};
      draw(0);
      tl.to(state, {{ time: {duration:.3f}, duration: {duration:.3f}, ease: 'none', onUpdate: () => draw(state.time) }}, 0);
      window.__timelines['main'] = tl;
    </script>
  </body>
</html>
'''
