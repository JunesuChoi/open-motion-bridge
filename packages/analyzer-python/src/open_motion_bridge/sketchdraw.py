"""Sketch drawing: vectorize a photo into ordered strokes drawn like a human hand,
optionally followed by a human-like coloring pass and a close-up camera that
follows the pen.

Pipeline: edges -> polyline strokes -> coarse-to-fine ordering with
nearest-neighbour pen travel -> per-stroke timing. Coloring paints soft brush
stamps (big background brush first, then a smaller detail brush) using colors
sampled from the photo. The close-up camera follows the precomputed pen/brush
position with smoothing and edge clamping. Everything — strokes, stamps, and
the camera table — is written to `sketch.plan.json` so the result stays
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
            points = list(reversed(chosen["points"])) if best_reversed else chosen["points"]
            ordered.append({"points": points, "lengthPx": chosen["lengthPx"]})
            pen = points[-1]
    return ordered


def _schedule(
    ordered: list[dict[str, Any]], start_ms: float, draw_ms: float, gap_ms: float, prefix: str
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


def _paint_stamps(work: np.ndarray, start_ms: float, paint_ms: float) -> list[dict[str, Any]]:
    """Two-pass brush stamps: big background brush first, then a detail brush.

    Colors come from a lightly blurred copy of the photo so stamps look like mixed
    paint instead of pixel noise. Order is serpentine per pass — the way a person
    sweeps a wash across the paper — and every stamp carries explicit timing.
    """
    height, width = work.shape[0], work.shape[1]
    blurred = cv2.GaussianBlur(work, (0, 0), 4)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    stamps: list[dict[str, Any]] = []

    def add_pass(cell: int, radius_ratio: float, detail_only: bool) -> None:
        rows = max(1, height // cell)
        cols = max(1, width // cell)
        for row in range(rows):
            col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
            for col in col_range:
                y0, y1 = row * cell, min(height, (row + 1) * cell)
                x0, x1 = col * cell, min(width, (col + 1) * cell)
                if detail_only and float(gray[y0:y1, x0:x1].std()) < 14.0:
                    continue
                cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
                jitter = ((hash((row, col, cell)) % 1000) / 1000.0 - 0.5) * cell * 0.3
                b, g, r = blurred[int(min(height - 1, cy)), int(min(width - 1, cx))]
                stamps.append(
                    {
                        "x": round((cx + jitter) / width, 5),
                        "y": round((cy - jitter) / height, 5),
                        "r": round(cell * radius_ratio / width, 5),
                        "color": f"#{int(r):02x}{int(g):02x}{int(b):02x}",
                    }
                )

    add_pass(cell=max(24, width // 26), radius_ratio=0.85, detail_only=False)
    add_pass(cell=max(12, width // 60), radius_ratio=0.75, detail_only=True)

    if not stamps:
        return []
    per = max(8.0, paint_ms / len(stamps))
    duration = min(260.0, per * 6)
    for index, stamp in enumerate(stamps):
        stamp["id"] = f"paint-{index:04d}"
        stamp["startMs"] = round(start_ms + index * per, 2)
        stamp["durationMs"] = round(duration, 2)
    return stamps


def _position_table(
    strokes: list[dict[str, Any]], stamps: list[dict[str, Any]], duration_ms: float, fps: float
) -> list[tuple[float, float, float]]:
    """Precompute the pen/brush position for every camera sample."""
    events: list[tuple[float, float, float, float]] = []
    for stroke in strokes:
        points = stroke["points"]
        seg = max(1, len(points) - 1)
        for i in range(seg):
            t0 = stroke["startMs"] + stroke["durationMs"] * i / seg
            events.append((t0, stroke["durationMs"] / seg, points[i][0], points[i][1]))
    for stamp in stamps:
        events.append((stamp["startMs"], stamp["durationMs"], stamp["x"], stamp["y"]))
    events.sort(key=lambda e: e[0])

    table: list[tuple[float, float, float]] = []
    cursor = 0
    last = (0.5, 0.5)
    count = max(1, int(duration_ms / 1000.0 * fps))
    for i in range(count):
        t = i * 1000.0 / fps
        while cursor < len(events) and events[cursor][0] + events[cursor][1] < t:
            cursor += 1
        if cursor < len(events) and events[cursor][0] <= t:
            last = (events[cursor][2], events[cursor][3])
        table.append((round(t, 2), last[0], last[1]))
    return table


def _camera_table(
    positions: list[tuple[float, float, float]],
    zoom: float,
    duration_ms: float,
    release_ms: float,
) -> list[dict[str, float]]:
    """Smooth follow camera: EMA toward the pen, clamped to the frame, easing back
    to full view during the final release window."""
    table: list[dict[str, float]] = []
    cx, cy = 0.5, 0.5
    alpha = 0.10
    for t, px, py in positions:
        cx += (px - cx) * alpha
        cy += (py - cy) * alpha
        ramp_in = min(1.0, t / 900.0)
        release = max(0.0, min(1.0, (t - (duration_ms - release_ms)) / max(1.0, release_ms)))
        eased_release = release * release * (3 - 2 * release)
        s = 1.0 + (zoom - 1.0) * ramp_in * (1.0 - eased_release)
        half = 0.5 / s
        ccx = min(1.0 - half, max(half, cx))
        ccy = min(1.0 - half, max(half, cy))
        fcx = 0.5 + (ccx - 0.5) * (1.0 - eased_release)
        fcy = 0.5 + (ccy - 0.5) * (1.0 - eased_release)
        table.append({"t": round(t / 1000.0, 4), "s": round(s, 5), "cx": round(fcx, 5), "cy": round(fcy, 5)})
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
    closeup_zoom: float = 0.0,
    force: bool = False,
) -> dict[str, Any]:
    if not source_image.is_file():
        raise FileNotFoundError(f"Source image not found: {source_image}")
    if color_mode not in {"none", "paint"}:
        raise ValueError("color_mode must be 'none' or 'paint'")
    if closeup_zoom and not 1.0 < closeup_zoom <= 3.0:
        raise ValueError("closeup_zoom must be within (1.0, 3.0] or 0 to disable")
    if duration_ms <= hold_ms + photo_fade_ms:
        raise ValueError("durationMs must exceed holdMs + photoFadeMs so drawing time remains.")
    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode image: {source_image}")
    height, width = image.shape[0], image.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    strokes, work = _extract_strokes(
        image, max_dimension=1400, canny_low=60, canny_high=160, min_stroke_px=28.0, epsilon_px=1.2
    )
    work_h, work_w = work.shape[0], work.shape[1]
    ordered = _order_strokes(strokes, work_w, work_h)[:max_strokes]
    if not ordered:
        raise RuntimeError("No drawable strokes were extracted; adjust edge thresholds.")

    active_ms = duration_ms - hold_ms - photo_fade_ms
    draw_ms = active_ms * (0.45 if color_mode == "paint" else 1.0)
    paint_ms = active_ms - draw_ms if color_mode == "paint" else 0.0
    gap_ms = min(28.0, draw_ms * 0.15 / max(1, len(ordered)))
    scheduled = _schedule(ordered, 0.0, draw_ms, gap_ms, "stroke")
    stamps = _paint_stamps(work, draw_ms, paint_ms) if color_mode == "paint" else []

    camera: list[dict[str, float]] = []
    if closeup_zoom:
        positions = _position_table(scheduled, stamps, duration_ms, fps)
        camera = _camera_table(positions, closeup_zoom, duration_ms, release_ms=hold_ms + photo_fade_ms)

    plan = {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": _utc_now(),
        "source": {"sourceHash": _sha256(source_image), "fileName": source_image.name, "displayWidth": width, "displayHeight": height},
        "timing": {"durationMs": duration_ms, "drawMs": round(draw_ms, 2), "paintMs": round(paint_ms, 2), "photoFadeMs": photo_fade_ms, "holdMs": hold_ms, "fps": fps, "penGapMs": round(gap_ms, 3)},
        "vectorization": {"workingSize": [work_w, work_h], "strokeCount": len(scheduled), "totalInkPx": round(sum(s["lengthPx"] for s in scheduled), 1), "ordering": "coarse-to-fine, nearest-neighbour pen travel"},
        "coloring": {"mode": color_mode, "stampCount": len(stamps), "passes": "background brush then detail brush, serpentine order"},
        "camera": {"mode": "pen-follow" if closeup_zoom else "static", "zoom": closeup_zoom, "samples": len(camera)},
        "strokes": scheduled,
        "paintStamps": stamps,
        "cameraTable": camera,
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
            "paintStampCount": len(stamps),
            "closeupZoom": closeup_zoom,
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
        d = "M " + " L ".join(f"{x * width:.2f} {y * height:.2f}" for x, y in stroke["points"])
        path_elements.append(f'<path id="{stroke["id"]}" d="{d}" />')
        stroke_meta.append({"id": stroke["id"], "s": stroke["startMs"], "d": stroke["durationMs"]})
    paths_markup = "".join(path_elements)
    meta_json = json.dumps(stroke_meta, separators=(",", ":"))
    stamps_json = json.dumps(
        [
            {"x": s["x"], "y": s["y"], "r": s["r"], "c": s["color"], "s": s["startMs"], "d": s["durationMs"]}
            for s in plan["paintStamps"]
        ],
        separators=(",", ":"),
    )
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
      #camera {{ position: absolute; left: 0; top: 0; width: {width}px; height: {height}px; transform-origin: 0 0; }}
      #paper {{ position: absolute; inset: 0; background:
        radial-gradient(circle at 30% 20%, rgba(255,255,255,0.9), rgba(0,0,0,0) 60%),
        repeating-linear-gradient(0deg, rgba(0,0,0,0.012) 0 2px, rgba(0,0,0,0) 2px 4px), #f7f2e9; }}
      #paint {{ position: absolute; inset: 0; }}
      #photo {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0; }}
      #ink {{ position: absolute; inset: 0; }}
      #ink path {{ fill: none; stroke: #2b2620; stroke-width: 2.1; stroke-linecap: round; stroke-linejoin: round; opacity: 0.92; }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{duration:.3f}" data-width="{width}" data-height="{height}" data-fps="{fps:g}">
      <div id="camera" class="clip" data-start="0" data-duration="{duration:.3f}" data-track-index="1">
        <div id="paper"></div>
        <canvas id="paint" width="{width}" height="{height}"></canvas>
        <img id="photo" src="assets/{staged_image_name}" alt="" />
        <svg id="ink" viewBox="0 0 {width} {height}">{paths_markup}</svg>
      </div>
    </div>
    <script>
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true }});
      const W = {width}, H = {height};
      const strokes = {meta_json};
      const stamps = {stamps_json};
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
      const cameraNode = document.getElementById('camera');
      const drawPaint = (t) => {{
        ctx.clearRect(0, 0, W, H);
        for (const s of stamps) {{
          const p = Math.max(0, Math.min(1, (t - s.s) / s.d));
          if (p <= 0) continue;
          const e = p * p * (3 - 2 * p);
          const r = s.r * W * (0.6 + 0.4 * e);
          const g = ctx.createRadialGradient(s.x * W, s.y * H, 0, s.x * W, s.y * H, r);
          g.addColorStop(0, s.c);
          g.addColorStop(0.75, s.c);
          g.addColorStop(1, s.c + '00');
          ctx.globalAlpha = 0.85 * e;
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.arc(s.x * W, s.y * H, r, 0, Math.PI * 2);
          ctx.fill();
        }}
        ctx.globalAlpha = 1;
      }};
      const applyCamera = (timeSec) => {{
        if (!cameraFrames.length) return;
        let lo = 0, hi = cameraFrames.length - 1;
        while (lo < hi) {{ const mid = Math.ceil((lo + hi) / 2); if (cameraFrames[mid].t <= timeSec) lo = mid; else hi = mid - 1; }}
        const f = cameraFrames[lo];
        cameraNode.style.transform = 'translate(' + ((0.5 - f.cx * f.s) * W).toFixed(3) + 'px,' + ((0.5 - f.cy * f.s) * H).toFixed(3) + 'px) scale(' + f.s.toFixed(6) + ')';
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
      }};
      const state = {{ time: 0 }};
      draw(0);
      tl.to(state, {{ time: {duration:.3f}, duration: {duration:.3f}, ease: 'none', onUpdate: () => draw(state.time) }}, 0);
      window.__timelines['main'] = tl;
    </script>
  </body>
</html>
'''
