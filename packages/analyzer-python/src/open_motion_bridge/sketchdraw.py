"""Sketch drawing: vectorize a photo into ordered strokes drawn like a human hand.

The photo's edges become polyline strokes. Strokes are ordered coarse-to-fine
(long structural outlines first, short detail strokes later) and each stroke is
assigned an explicit start/duration on the timeline, proportional to its arc
length, with pen-travel gaps between strokes. The plan is written to
`sketch.plan.json` so the reveal order and timing stay auditable, and the
composition draws each stroke with stroke-dashoffset so seeking any frame is
deterministic. Rendering stays external; this module only generates the project.
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
) -> tuple[list[list[tuple[float, float]]], tuple[int, int]]:
    """Return normalized polyline strokes and the working pixel size."""
    height, width = image.shape[0], image.shape[1]
    scale = min(1.0, max_dimension / max(width, height))
    work = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else image
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
    return strokes, (work_w, work_h)


def _stroke_length(points: list[tuple[float, float]], width: int, height: int) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot((x1 - x0) * width, (y1 - y0) * height)
    return total


def _order_strokes(
    strokes: list[list[tuple[float, float]]], width: int, height: int
) -> list[dict[str, Any]]:
    """Coarse-to-fine ordering with nearest-neighbour pen travel inside each pass.

    A human sketch lays down long structural lines first and detail last. Within a
    pass the pen moves to the nearest unfinished stroke endpoint, which keeps the
    reveal spatially coherent instead of jumping randomly across the canvas.
    """
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
            points = list(reversed(chosen["points"])) if best_reversed else chosen["points"]
            ordered.append({"points": points, "lengthPx": chosen["lengthPx"]})
            pen = points[-1]
    return ordered


def _schedule(
    ordered: list[dict[str, Any]], draw_ms: float, gap_ms: float
) -> list[dict[str, Any]]:
    """Assign start/duration per stroke, proportional to arc length."""
    total_length = sum(item["lengthPx"] for item in ordered) or 1.0
    budget = draw_ms - gap_ms * max(0, len(ordered) - 1)
    if budget <= 0:
        raise ValueError("durationMs is too short for the stroke count; reduce strokes or extend duration.")
    cursor = 0.0
    scheduled: list[dict[str, Any]] = []
    for index, item in enumerate(ordered):
        duration = max(16.0, budget * item["lengthPx"] / total_length)
        scheduled.append(
            {
                "id": f"stroke-{index:04d}",
                "startMs": round(cursor, 2),
                "durationMs": round(duration, 2),
                "lengthPx": round(item["lengthPx"], 2),
                "points": [[round(x, 5), round(y, 5)] for x, y in item["points"]],
            }
        )
        cursor += duration + gap_ms
    return scheduled


def generate_sketch_project(
    source_image: Path,
    output_dir: Path,
    duration_ms: float = 10000.0,
    fps: float = 30.0,
    hold_ms: float = 2000.0,
    photo_fade_ms: float = 1500.0,
    max_strokes: int = 900,
    force: bool = False,
) -> dict[str, Any]:
    """Vectorize, schedule, and emit a self-drawing HyperFrames project."""
    if not source_image.is_file():
        raise FileNotFoundError(f"Source image not found: {source_image}")
    if duration_ms <= hold_ms + photo_fade_ms:
        raise ValueError("durationMs must exceed holdMs + photoFadeMs so drawing time remains.")
    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode image: {source_image}")
    height, width = image.shape[0], image.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    strokes, (work_w, work_h) = _extract_strokes(
        image, max_dimension=1400, canny_low=60, canny_high=160, min_stroke_px=28.0, epsilon_px=1.2
    )
    ordered = _order_strokes(strokes, work_w, work_h)
    if len(ordered) > max_strokes:
        ordered = ordered[:max_strokes]
    if not ordered:
        raise RuntimeError("No drawable strokes were extracted; adjust edge thresholds.")

    draw_ms = duration_ms - hold_ms - photo_fade_ms
    gap_ms = min(28.0, draw_ms * 0.15 / max(1, len(ordered)))
    scheduled = _schedule(ordered, draw_ms, gap_ms)

    plan = {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": _utc_now(),
        "source": {"sourceHash": _sha256(source_image), "fileName": source_image.name, "displayWidth": width, "displayHeight": height},
        "timing": {"durationMs": duration_ms, "drawMs": round(draw_ms, 2), "photoFadeMs": photo_fade_ms, "holdMs": hold_ms, "fps": fps, "penGapMs": round(gap_ms, 3)},
        "vectorization": {"workingSize": [work_w, work_h], "strokeCount": len(scheduled), "totalInkPx": round(sum(s["lengthPx"] for s in scheduled), 1), "ordering": "coarse-to-fine, nearest-neighbour pen travel"},
        "strokes": scheduled,
    }
    _write_json(output_dir / "sketch.plan.json", plan, force)

    assets = output_dir / "assets"
    assets.mkdir(exist_ok=True)
    staged = assets / ("source" + source_image.suffix.lower())
    if staged.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {staged}; use --force for generated output.")
    import shutil

    shutil.copy2(source_image, staged)

    html_path = output_dir / "index.html"
    if html_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {html_path}; use --force for generated output.")
    html_path.write_text(_render_sketch_html(plan, staged.name), encoding="utf-8")
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
    draw_end = float(timing["drawMs"])
    fade_end = draw_end + float(timing["photoFadeMs"])

    path_elements: list[str] = []
    stroke_meta: list[dict[str, Any]] = []
    for stroke in plan["strokes"]:
        d = "M " + " L ".join(f"{x * width:.2f} {y * height:.2f}" for x, y in stroke["points"])
        path_elements.append(f'<path id="{stroke["id"]}" d="{d}" />')
        stroke_meta.append({"id": stroke["id"], "s": stroke["startMs"], "d": stroke["durationMs"]})
    paths_markup = "".join(path_elements)
    meta_json = json.dumps(stroke_meta, separators=(",", ":"))

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
      #paper {{ position: absolute; inset: 0; background:
        radial-gradient(circle at 30% 20%, rgba(255,255,255,0.9), rgba(0,0,0,0) 60%),
        repeating-linear-gradient(0deg, rgba(0,0,0,0.012) 0 2px, rgba(0,0,0,0) 2px 4px), #f7f2e9; }}
      #photo {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0; }}
      #ink {{ position: absolute; inset: 0; }}
      #ink path {{ fill: none; stroke: #2b2620; stroke-width: 2.1; stroke-linecap: round; stroke-linejoin: round; opacity: 0.92; }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{duration:.3f}" data-width="{width}" data-height="{height}" data-fps="{fps:g}">
      <div id="paper" class="clip" data-start="0" data-duration="{duration:.3f}" data-track-index="1"></div>
      <img id="photo" class="clip" src="assets/{staged_image_name}" alt="" data-start="0" data-duration="{duration:.3f}" data-track-index="2" />
      <svg id="ink" class="clip" viewBox="0 0 {width} {height}" data-start="0" data-duration="{duration:.3f}" data-track-index="3">{paths_markup}</svg>
    </div>
    <script>
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true }});
      const strokes = {meta_json};
      const DRAW_END = {draw_end:.2f}, FADE_END = {fade_end:.2f};
      const nodes = strokes.map((s) => {{
        const node = document.getElementById(s.id);
        const len = node.getTotalLength();
        node.style.strokeDasharray = String(len);
        node.style.strokeDashoffset = String(len);
        return {{ meta: s, node, len }};
      }});
      const photo = document.getElementById('photo');
      const draw = (timeSec) => {{
        const t = timeSec * 1000;
        for (const entry of nodes) {{
          const p = Math.max(0, Math.min(1, (t - entry.meta.s) / entry.meta.d));
          entry.node.style.strokeDashoffset = String(entry.len * (1 - p));
        }}
        const fade = Math.max(0, Math.min(1, (t - DRAW_END) / Math.max(1, FADE_END - DRAW_END)));
        const eased = fade * fade * (3 - 2 * fade);
        photo.style.opacity = String(eased);
        document.getElementById('ink').style.opacity = String(1 - eased * 0.85);
      }};
      const state = {{ time: 0 }};
      draw(0);
      tl.to(state, {{ time: {duration:.3f}, duration: {duration:.3f}, ease: 'none', onUpdate: () => draw(state.time) }}, 0);
      window.__timelines['main'] = tl;
    </script>
  </body>
</html>
'''
