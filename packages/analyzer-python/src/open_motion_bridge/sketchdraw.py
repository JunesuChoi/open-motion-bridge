"""Sketch drawing: vectorize a photo into ordered strokes drawn like a human hand,
optionally followed by a human-like coloring pass and a close-up camera that
follows the pen.

Pipeline: edges -> polyline strokes -> coarse-to-fine ordering with
nearest-neighbour pen travel -> per-stroke timing. Coloring can either preserve
the legacy source-reveal masks or paint deterministic, locally sampled RGB
strokes ordered by Lab-aware image regions. The close-up camera follows the
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
import mediapipe as mp
import numpy as np

from .pipeline import SCHEMA_VERSION, _sha256, _utc_now, _write_json

_SELFIE_SEGMENTER: Any | None = None


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


def _lab_distance(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b, strict=True)))


def _sampled_color_strokes(
    work: np.ndarray,
    start_ms: float,
    paint_ms: float,
    start_point: tuple[float, float],
) -> list[dict[str, Any]]:
    """Paint the image with real RGB strokes ordered by local Lab regions.

    A deterministic grid avoids an optional segmentation dependency. Adjacent
    cells with similar Lab medians become regions; regions and cells are then
    traversed from the final ink coordinate with nearest-neighbour travel. Each
    stroke stores the RGB color it actually renders, so the plan can be audited
    without sampling the source again in the browser.
    """
    height, width = work.shape[:2]
    blurred = cv2.GaussianBlur(work, (0, 0), 1.35)
    lab_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    cell = max(14, int(math.ceil(max(width, height) / 38.0)))
    cells: list[dict[str, Any]] = []
    by_grid: dict[tuple[int, int], int] = {}
    for gy, y0 in enumerate(range(0, height, cell)):
        for gx, x0 in enumerate(range(0, width, cell)):
            x1, y1 = min(width, x0 + cell), min(height, y0 + cell)
            if x1 <= x0 or y1 <= y0:
                continue
            color_patch = blurred[y0:y1, x0:x1]
            lab_patch = lab_image[y0:y1, x0:x1]
            b, g, r = np.median(color_patch.reshape(-1, 3), axis=0)
            lab = tuple(
                float(value) for value in np.median(lab_patch.reshape(-1, 3), axis=0)
            )
            cx, cy = (x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0
            px, py = int(round(cx)), int(round(cy))
            tangent = (
                math.atan2(float(grad_y[py, px]), float(grad_x[py, px])) + math.pi / 2.0
            )
            index = len(cells)
            by_grid[(gx, gy)] = index
            cells.append(
                {
                    "index": index,
                    "grid": (gx, gy),
                    "center": (cx / width, cy / height),
                    "lab": lab,
                    "color": f"#{int(r):02x}{int(g):02x}{int(b):02x}",
                    "angle": tangent,
                    "cellPx": min(x1 - x0, y1 - y0),
                }
            )

    # Flood-fill locally coherent color regions. The threshold is deliberately
    # broad enough for photographic gradients while keeping skin, hair, clothes,
    # and background from collapsing into one top-to-bottom sweep.
    unassigned = set(range(len(cells)))
    regions: list[list[int]] = []
    cell_regions: dict[int, int] = {}
    while unassigned:
        seed = min(unassigned)
        unassigned.remove(seed)
        queue = [seed]
        region = [seed]
        queue_cursor = 0
        while queue_cursor < len(queue):
            current = queue[queue_cursor]
            queue_cursor += 1
            gx, gy = cells[current]["grid"]
            for neighbour_grid in (
                (gx - 1, gy),
                (gx + 1, gy),
                (gx, gy - 1),
                (gx, gy + 1),
            ):
                neighbour = by_grid.get(neighbour_grid)
                if neighbour not in unassigned:
                    continue
                if (
                    _lab_distance(cells[current]["lab"], cells[neighbour]["lab"])
                    <= 24.0
                ):
                    unassigned.remove(neighbour)
                    queue.append(neighbour)
                    region.append(neighbour)
        region_id = len(regions)
        regions.append(region)
        for cell_index in region:
            cell_regions[cell_index] = region_id

    pen = start_point
    ordered_cells: list[int] = []
    remaining_regions = list(regions)
    while remaining_regions:
        region_index = min(
            range(len(remaining_regions)),
            key=lambda ri: min(
                math.hypot(
                    cells[ci]["center"][0] - pen[0], cells[ci]["center"][1] - pen[1]
                )
                for ci in remaining_regions[ri]
            ),
        )
        region = remaining_regions.pop(region_index)
        remaining_cells = list(region)
        while remaining_cells:
            chosen_index = min(
                range(len(remaining_cells)),
                key=lambda ci: math.hypot(
                    cells[remaining_cells[ci]]["center"][0] - pen[0],
                    cells[remaining_cells[ci]]["center"][1] - pen[1],
                ),
            )
            chosen = remaining_cells.pop(chosen_index)
            ordered_cells.append(chosen)
            pen = cells[chosen]["center"]

    strokes: list[dict[str, Any]] = []
    pen = start_point
    for order, cell_index in enumerate(ordered_cells):
        item = cells[cell_index]
        cx, cy = item["center"]
        angle = float(item["angle"])
        if not math.isfinite(angle):
            angle = (_noise01(order, 41) - 0.5) * math.pi
        half_length = float(item["cellPx"]) * (0.70 + _noise01(order, 43) * 0.15)
        dx = math.cos(angle) * half_length / width
        dy = math.sin(angle) * half_length / height
        a, b = (cx - dx, cy - dy), (cx + dx, cy + dy)
        if math.hypot(b[0] - pen[0], b[1] - pen[1]) < math.hypot(
            a[0] - pen[0], a[1] - pen[1]
        ):
            a, b = b, a
        x0, y0 = min(1.0, max(0.0, a[0])), min(1.0, max(0.0, a[1]))
        x1, y1 = min(1.0, max(0.0, b[0])), min(1.0, max(0.0, b[1]))
        strokes.append(
            {
                "pass": "sampled",
                "region": cell_regions[cell_index],
                "x0": round(x0, 5),
                "y0": round(y0, 5),
                "x1": round(x1, 5),
                "y1": round(y1, 5),
                "width": round(float(item["cellPx"]) * 0.82 / width, 5),
                "opacity": 0.9,
                "sampledColor": item["color"],
                "color": item["color"],
                "lab": [round(value, 2) for value in item["lab"]],
            }
        )
        pen = (x1, y1)

    _schedule_brush_pass(strokes, start_ms, paint_ms, 0)
    return strokes


def _cielab_triplet(opencv_lab: np.ndarray) -> tuple[float, float, float]:
    """Convert OpenCV's uint8 Lab encoding into conventional CIE Lab values."""
    return (
        float(opencv_lab[0]) * 100.0 / 255.0,
        float(opencv_lab[1]) - 128.0,
        float(opencv_lab[2]) - 128.0,
    )


def _delta_e76_map(current_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
    target_cie = np.empty_like(target_lab, dtype=np.float32)
    current_cie = np.empty_like(current_lab, dtype=np.float32)
    target_cie[..., 0] = target_lab[..., 0] * (100.0 / 255.0)
    target_cie[..., 1:] = target_lab[..., 1:] - 128.0
    current_cie[..., 0] = current_lab[..., 0] * (100.0 / 255.0)
    current_cie[..., 1:] = current_lab[..., 1:] - 128.0
    return np.linalg.norm(target_cie - current_cie, axis=2)


def _subject_foreground_mask(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a local semantic portrait matte with a deterministic fallback.

    The drawing needs to distinguish a person from stage lettering and other
    high-contrast background objects. This matte is a routing hint for ink and
    pigment detail, never a claim that the result is pixel-perfect segmentation.
    """
    height, width = image.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    fallback = ((xx - width * 0.5) / max(1.0, width * 0.47)) ** 2 + (
        (yy - height * 0.52) / max(1.0, height * 0.60)
    ) ** 2 <= 1.0
    metadata: dict[str, Any] = {
        "source": "central-portrait-prior",
        "threshold": 0.35,
        "foregroundRatio": round(float(fallback.mean()), 5),
    }
    try:
        global _SELFIE_SEGMENTER
        if _SELFIE_SEGMENTER is None:
            _SELFIE_SEGMENTER = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
        result = _SELFIE_SEGMENTER.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        confidence = result.segmentation_mask
        semantic = confidence >= 0.35
        ratio = float(semantic.mean())
        if 0.08 <= ratio <= 0.88:
            matte = cv2.morphologyEx(
                semantic.astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((5, 5), np.uint8),
                iterations=1,
            ).astype(bool)
            metadata = {
                "source": "mediapipe-selfie-segmentation",
                "threshold": 0.35,
                "foregroundRatio": round(float(matte.mean()), 5),
            }
            return matte, metadata
    except (AttributeError, RuntimeError, cv2.error):
        pass
    return fallback, metadata


def _face_candidate_box(image: np.ndarray) -> tuple[int, int, int, int, str]:
    """Find a local face box, retaining a conservative centre prior as fallback."""
    height, width = image.shape[:2]
    detected_faces: list[tuple[int, int, int, int]] = []
    try:
        cascade_path = str(
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty():
            detected_faces = [
                tuple(map(int, face))
                for face in cascade.detectMultiScale(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(24, 24),
                )
            ]
    except (AttributeError, cv2.error):
        detected_faces = []
    if detected_faces:
        fx, fy, fw, fh = max(
            detected_faces,
            key=lambda box: box[2]
            * box[3]
            * (1.2 - abs((box[0] + box[2] / 2) / width - 0.5)),
        )
        return fx, fy, fw, fh, "opencv-haar"
    fw, fh = int(width * 0.34), int(height * 0.30)
    fx, fy = int(width * 0.5 - fw / 2), int(height * 0.36 - fh / 2)
    return fx, fy, fw, fh, "central-portrait-prior"


def _face_candidate_mask(image: np.ndarray) -> tuple[np.ndarray, str]:
    """Build an ellipse slightly inside the face box for delicate ink treatment."""
    height, width = image.shape[:2]
    fx, fy, fw, fh, source = _face_candidate_box(image)
    yy, xx = np.ogrid[:height, :width]
    mask = ((xx - (fx + fw / 2.0)) / max(1.0, fw * 0.52)) ** 2 + (
        (yy - (fy + fh / 2.0)) / max(1.0, fh * 0.52)
    ) ** 2 <= 1.0
    return mask, source


def _stroke_foreground_fraction(
    points: list[tuple[float, float]], foreground: np.ndarray
) -> float:
    """Estimate how much of an extracted SVG contour belongs to the subject."""
    height, width = foreground.shape
    samples: list[tuple[int, int]] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        segment = max(
            1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) * max(width, height)))
        )
        for step in range(segment + 1):
            mix = step / segment
            samples.append(
                (
                    min(width - 1, max(0, int(round((x0 + (x1 - x0) * mix) * width)))),
                    min(
                        height - 1, max(0, int(round((y0 + (y1 - y0) * mix) * height)))
                    ),
                )
            )
    if not samples:
        return 0.0
    return float(np.mean([foreground[y, x] for x, y in samples]))


def _masked_completion(
    mask: np.ndarray, coverage: np.ndarray, delta: np.ndarray
) -> dict[str, float]:
    if not np.any(mask):
        return {"coverage": 0.0, "meanDeltaE76": 999.0, "largestHoleRatio": 1.0}
    touched = coverage >= 0.5
    region_size = int(mask.sum())
    holes = np.logical_and(mask, np.logical_not(touched)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    largest_hole = (
        float(stats[1:, cv2.CC_STAT_AREA].max()) / region_size
        if len(stats) > 1
        else 0.0
    )
    return {
        "coverage": round(float(touched[mask].mean()), 5),
        "meanDeltaE76": round(float(delta[mask].mean()), 3),
        "largestHoleRatio": round(largest_hole, 5),
    }


def _coverage_metrics(
    coverage: np.ndarray,
    current_lab: np.ndarray,
    target_lab: np.ndarray,
    importance: np.ndarray,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure completion from the simulated pigment field, not stroke count."""
    touched = coverage >= 0.5
    domain = (
        evaluation_mask.astype(bool)
        if evaluation_mask is not None
        else np.ones_like(touched, dtype=bool)
    )
    important = np.logical_and(importance >= 0.72, domain)
    overall = float(touched[domain].mean()) if np.any(domain) else 0.0
    important_coverage = (
        float(touched[important].mean()) if np.any(important) else overall
    )
    holes = np.logical_and(domain, np.logical_not(touched)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    largest_hole = (
        float(stats[1:, cv2.CC_STAT_AREA].max()) / int(domain.sum())
        if len(stats) > 1 and np.any(domain)
        else 0.0
    )
    delta = _delta_e76_map(current_lab, target_lab)
    important_delta = (
        float(delta[important].mean()) if np.any(important) else float(delta.mean())
    )
    residual = 0.72 * (1.0 - coverage) + 0.28 * np.minimum(delta / 100.0, 1.0)
    return {
        "overallCoverage": round(overall, 5),
        "importantCoverage": round(important_coverage, 5),
        "largestHoleRatio": round(largest_hole, 5),
        "importantMeanDeltaE76": round(important_delta, 3),
        "meanResidual": round(float(residual[domain].mean()), 5)
        if np.any(domain)
        else 0.0,
    }


def _portrait_importance_map(
    image: np.ndarray, gradient: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]], list[np.ndarray]]:
    """Build a deterministic saliency field with optional local face evidence.

    A local Haar cascade is used when OpenCV ships one and detects a face. The
    central portrait prior is retained as a fallback, while likely skin blobs
    outside the face become hand candidates. These are only prioritization
    hints; they are recorded as heuristics rather than claimed detections.
    """
    height, width = image.shape[:2]
    percentile = float(np.percentile(gradient, 96.0)) or 1.0
    normalized_gradient = np.clip(gradient / percentile, 0.0, 1.0)
    importance = 0.25 + normalized_gradient * 0.32
    regions: list[dict[str, Any]] = []
    region_masks: list[np.ndarray] = []

    fx, fy, fw, fh, face_source = _face_candidate_box(image)

    yy, xx = np.ogrid[:height, :width]
    face_cx, face_cy = fx + fw / 2.0, fy + fh / 2.0
    face_mask = ((xx - face_cx) / max(1.0, fw * 0.62)) ** 2 + (
        (yy - face_cy) / max(1.0, fh * 0.62)
    ) ** 2 <= 1.0
    importance[face_mask] += 0.43
    regions.append(
        {
            "id": "face-001",
            "kind": "face-candidate",
            "source": face_source,
            "box": [
                round(fx / width, 5),
                round(fy / height, 5),
                round(fw / width, 5),
                round(fh / height, 5),
            ],
        }
    )
    region_masks.append(face_mask.copy())

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array((0, 133, 77)), np.array((255, 180, 135)))
    skin[face_mask] = 0
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    label_count, label_map, stats, centroids = cv2.connectedComponentsWithStats(skin, 8)
    hand_candidates: list[tuple[float, int]] = []
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(18, int(width * height * 0.003)):
            continue
        cx, cy = map(float, centroids[label])
        if cy > height * 0.82:
            continue
        distance_from_face = math.hypot(
            (cx - face_cx) / max(1.0, width), (cy - face_cy) / max(1.0, height)
        )
        hand_candidates.append((area * (0.6 + distance_from_face), label))
    for hand_index, (_, label) in enumerate(
        sorted(hand_candidates, reverse=True)[:2], 1
    ):
        x, y, w, h, _ = map(int, stats[label])
        component = label_map == label
        importance[component] += 0.31
        regions.append(
            {
                "id": f"hand-{hand_index:03d}",
                "kind": "hand-candidate",
                "source": "skin-component-heuristic",
                "box": [
                    round(x / width, 5),
                    round(y / height, 5),
                    round(w / width, 5),
                    round(h / height, 5),
                ],
            }
        )
        region_masks.append(component.copy())
    # Quantize the saliency field so OpenCV's parallel floating-point kernels
    # cannot move boundary pixels across the completion threshold between runs.
    deterministic_importance = np.round(np.clip(importance, 0.0, 1.0), 4).astype(
        np.float32
    )
    return deterministic_importance, regions, region_masks


def _residual_pigment_strokes(
    work: np.ndarray,
    start_ms: float,
    paint_ms: float,
    start_point: tuple[float, float],
    fps: float = 30.0,
    foreground_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Precompute subject-aware, boundary-bounded pigment strokes."""
    source_h, source_w = work.shape[:2]
    scale = min(1.0, 288.0 / max(source_w, source_h))
    sim_w = max(48, int(round(source_w * scale)))
    sim_h = max(48, int(round(source_h * scale)))
    source_image = cv2.resize(work, (sim_w, sim_h), interpolation=cv2.INTER_AREA)
    if foreground_mask is None:
        full_foreground, foreground_metadata = _subject_foreground_mask(work)
    else:
        full_foreground = foreground_mask.astype(bool)
        foreground_metadata = {
            "source": "provided-subject-matte",
            "threshold": None,
            "foregroundRatio": round(float(full_foreground.mean()), 5),
        }
    foreground = (
        cv2.resize(
            full_foreground.astype(np.uint8),
            (sim_w, sim_h),
            interpolation=cv2.INTER_AREA,
        )
        >= 0.5
    )
    # Background is deliberately a broad color field. It keeps the scene's
    # light and palette without tracing stage text as though it were anatomy.
    blurred_background = cv2.GaussianBlur(source_image, (0, 0), 8.0)
    image = np.where(foreground[..., None], source_image, blurred_background)
    image = cv2.GaussianBlur(image, (0, 0), 0.85)
    target_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    jxx = cv2.GaussianBlur(grad_x * grad_x, (0, 0), 2.1)
    jyy = cv2.GaussianBlur(grad_y * grad_y, (0, 0), 2.1)
    jxy = cv2.GaussianBlur(grad_x * grad_y, (0, 0), 2.1)
    tangent = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy) + math.pi / 2.0
    source_grad_x = cv2.Sobel(source_gray, cv2.CV_32F, 1, 0, ksize=3)
    source_grad_y = cv2.Sobel(source_gray, cv2.CV_32F, 0, 1, ksize=3)
    source_gradient = cv2.magnitude(source_grad_x, source_grad_y)
    importance, importance_regions, importance_masks = _portrait_importance_map(
        source_image, source_gradient
    )
    importance = np.where(foreground, importance, np.minimum(importance, 0.16)).astype(
        np.float32
    )
    skin_focus = np.zeros((sim_h, sim_w), dtype=bool)
    for mask in importance_masks:
        skin_focus |= mask
    hair_focus = np.logical_and(
        foreground,
        np.logical_and(
            np.logical_not(skin_focus),
            gray <= np.percentile(gray[foreground], 34.0)
            if np.any(foreground)
            else False,
        ),
    )
    strong_edge_threshold = max(8.0, float(np.percentile(gradient, 88.0)))

    paper = np.full((1, 1, 3), (233, 242, 247), dtype=np.uint8)
    paper_lab = cv2.cvtColor(paper, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    current_lab = np.empty_like(target_lab, dtype=np.float32)
    current_lab[:] = paper_lab
    coverage = np.zeros((sim_h, sim_w), dtype=np.float32)
    strokes: list[dict[str, Any]] = []
    settle_strokes: list[dict[str, Any]] = []

    def metrics() -> dict[str, float]:
        return _coverage_metrics(
            coverage, current_lab, target_lab, importance, foreground
        )

    def region_metrics() -> list[dict[str, Any]]:
        delta = _delta_e76_map(current_lab, target_lab)
        results: list[dict[str, Any]] = []
        for region, mask in zip(importance_regions, importance_masks, strict=True):
            results.append({**region, **_masked_completion(mask, coverage, delta)})
        return results

    def quick_values() -> tuple[float, float]:
        delta = _delta_e76_map(current_lab, target_lab)
        residual = 0.58 * (1.0 - coverage) + 0.42 * np.minimum(delta / 60.0, 1.0)
        return (
            float((coverage[foreground] >= 0.5).mean()),
            float(residual[foreground].mean()),
        )

    def residual_field(
        importance_weight: float, color_weight: float, sigma: float
    ) -> np.ndarray:
        delta = _delta_e76_map(current_lab, target_lab)
        field = (
            (1.0 - color_weight) * (1.0 - coverage)
            + color_weight * np.minimum(delta / 70.0, 1.0)
        ) * (0.7 + importance * importance_weight)
        field *= np.where(foreground, 1.0, 0.22)
        return cv2.GaussianBlur(field, (0, 0), sigma)

    def cv_lab_distance(left: np.ndarray, right: np.ndarray) -> float:
        diff = left.astype(np.float32) - right.astype(np.float32)
        return math.sqrt(
            (float(diff[0]) * 100.0 / 255.0) ** 2
            + float(diff[1]) ** 2
            + float(diff[2]) ** 2
        )

    def bounded_path(
        phase: str,
        center: tuple[int, int],
        max_length: float,
        radius: float,
        forced_start: tuple[float, float] | None,
    ) -> tuple[list[tuple[float, float]], dict[str, Any]]:
        px, py = center
        center_lab = target_lab[py, px]
        allowed_delta = {
            "mass": 10.0,
            "form": 8.0,
            "accent": 6.0,
            "settle-correction": 6.0,
        }[phase]
        initial_angle = float(tangent[py, px])
        if not math.isfinite(initial_angle) or float(gradient[py, px]) < 0.5:
            initial_angle = (_noise01(px, py, len(strokes), 97) - 0.5) * math.pi
        max_steps = max(2, int(round(max_length / 2.0)))
        stopped_by_color = False
        stopped_by_edge = False
        max_observed_delta = 0.0

        def ray(sign: float) -> list[tuple[float, float]]:
            nonlocal stopped_by_color, stopped_by_edge, max_observed_delta
            points: list[tuple[float, float]] = []
            x, y = float(px), float(py)
            vx, vy = math.cos(initial_angle) * sign, math.sin(initial_angle) * sign
            for step in range(max_steps):
                ix = min(sim_w - 1, max(0, int(round(x))))
                iy = min(sim_h - 1, max(0, int(round(y))))
                local_angle = float(tangent[iy, ix])
                if math.isfinite(local_angle):
                    candidates = (
                        (math.cos(local_angle), math.sin(local_angle)),
                        (-math.cos(local_angle), -math.sin(local_angle)),
                    )
                    local_vx, local_vy = max(
                        candidates, key=lambda vector: vector[0] * vx + vector[1] * vy
                    )
                    vx, vy = vx * 0.7 + local_vx * 0.3, vy * 0.7 + local_vy * 0.3
                    norm = math.hypot(vx, vy) or 1.0
                    vx, vy = vx / norm, vy / norm
                nx, ny = x + vx * 2.0, y + vy * 2.0
                if nx < 0 or nx >= sim_w or ny < 0 or ny >= sim_h:
                    break
                sample_x = min(sim_w - 1, max(0, int(round(nx))))
                sample_y = min(sim_h - 1, max(0, int(round(ny))))
                delta = cv_lab_distance(center_lab, target_lab[sample_y, sample_x])
                max_observed_delta = max(max_observed_delta, delta)
                if delta > allowed_delta:
                    stopped_by_color = True
                    break
                if (
                    step >= max(1, int(radius * 0.4))
                    and gradient[sample_y, sample_x] >= strong_edge_threshold
                ):
                    stopped_by_edge = True
                    break
                x, y = nx, ny
                points.append((x, y))
            return points

        positive = ray(1.0)
        if forced_start is None:
            negative = ray(-1.0)
            pixel_path = list(reversed(negative)) + [(float(px), float(py))] + positive
        else:
            pixel_path = [(forced_start[0] * sim_w, forced_start[1] * sim_h)] + positive
        if len(pixel_path) < 2:
            fallback_x = min(sim_w - 1.0, max(0.0, px + math.cos(initial_angle) * 1.5))
            fallback_y = min(sim_h - 1.0, max(0.0, py + math.sin(initial_angle) * 1.5))
            pixel_path.append((fallback_x, fallback_y))
        normalized = [(x / sim_w, y / sim_h) for x, y in pixel_path]
        return normalized, {
            "allowedDeltaE76": allowed_delta,
            "strongEdgeThreshold": round(strong_edge_threshold, 3),
            "maxObservedDeltaE76": round(max_observed_delta, 3),
            "stoppedByColorBoundary": stopped_by_color,
            "stoppedByStrongEdge": stopped_by_edge,
        }

    def add_stroke(
        phase: str,
        center: tuple[int, int],
        radius: float,
        base_length: float,
        opacity: float,
        forced_start: tuple[float, float] | None = None,
        target: list[dict[str, Any]] | None = None,
        correction_reason: str | None = None,
    ) -> None:
        px, py = center
        region = (
            "skin"
            if skin_focus[py, px]
            else "hair"
            if hair_focus[py, px]
            else "garment-or-prop"
            if foreground[py, px]
            else "background"
        )
        if region == "skin":
            radius *= {
                "mass": 0.70,
                "form": 0.80,
                "accent": 0.90,
                "settle-correction": 0.82,
            }[phase]
            base_length *= 0.78
            opacity *= 0.88
        elif region == "hair":
            radius *= {
                "mass": 0.84,
                "form": 0.90,
                "accent": 0.96,
                "settle-correction": 0.90,
            }[phase]
            base_length *= 0.86
        patch_scale = {
            "mass": 0.35,
            "form": 0.24,
            "accent": 0.08,
            "settle-correction": 0.05,
        }[phase]
        patch_radius = max(0, int(round(radius * patch_scale)))
        xlo, xhi = max(0, px - patch_radius), min(sim_w, px + patch_radius + 1)
        ylo, yhi = max(0, py - patch_radius), min(sim_h, py + patch_radius + 1)
        patch = image[ylo:yhi, xlo:xhi]
        lab_patch = target_lab[ylo:yhi, xlo:xhi]
        bgr = np.median(patch.reshape(-1, 3), axis=0)
        sampled_lab_cv = np.median(lab_patch.reshape(-1, 3), axis=0)
        color_lab = _cielab_triplet(sampled_lab_cv)
        color = f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"

        local_std = (
            float(np.mean(np.std(lab_patch.reshape(-1, 3), axis=0)))
            if patch.size > 1
            else 0.0
        )
        length = base_length * max(0.5, 1.0 - local_std / 110.0)
        points, boundary = bounded_path(phase, center, length, radius, forced_start)
        before_coverage, before_residual = quick_values()
        mask = np.zeros((sim_h, sim_w), dtype=np.uint8)
        pixel_points = np.array(
            [[int(round(x * sim_w)), int(round(y * sim_h))] for x, y in points],
            dtype=np.int32,
        )
        cv2.polylines(
            mask,
            [pixel_points],
            False,
            255,
            max(1, int(round(radius * 2.0))),
            lineType=cv2.LINE_AA,
        )
        candidate_y, candidate_x = np.nonzero(mask)
        if len(candidate_y):
            candidate_lab = target_lab[candidate_y, candidate_x]
            delta_lab = candidate_lab - sampled_lab_cv
            delta_lab[:, 0] *= 100.0 / 255.0
            candidate_delta = np.linalg.norm(delta_lab, axis=1)
            outside_boundary = candidate_delta > float(boundary["allowedDeltaE76"])
            mask[candidate_y[outside_boundary], candidate_x[outside_boundary]] = 0
            boundary["clippedPixelRatio"] = round(float(outside_boundary.mean()), 4)
        alpha = mask.astype(np.float32) / 255.0 * opacity
        alpha_3 = alpha[..., None]
        coverage[:] = coverage + alpha * (1.0 - coverage)
        current_lab[:] = current_lab * (1.0 - alpha_3) + sampled_lab_cv * alpha_3
        after_coverage, after_residual = quick_values()
        mask_pixels = mask > 0
        stroke_importance = (
            float(importance[mask_pixels].mean()) if np.any(mask_pixels) else 0.0
        )
        destination = strokes if target is None else target
        destination.append(
            {
                "pass": phase,
                "phase": phase,
                "points": [[round(x, 5), round(y, 5)] for x, y in points],
                "x0": round(points[0][0], 5),
                "y0": round(points[0][1], 5),
                "x1": round(points[-1][0], 5),
                "y1": round(points[-1][1], 5),
                "width": round(radius * 2.0 / sim_w, 5),
                "opacity": opacity,
                "sampledColor": color,
                "color": color,
                "lab": [round(value, 2) for value in color_lab],
                "importance": round(stroke_importance, 4),
                "region": region,
                "residualBefore": round(before_residual, 5),
                "residualAfter": round(after_residual, 5),
                "coverageBefore": round(before_coverage, 5),
                "coverageAfter": round(after_coverage, 5),
                "boundary": boundary,
                **(
                    {"correctionReason": correction_reason} if correction_reason else {}
                ),
            }
        )

    max_dim = float(max(sim_w, sim_h))
    phase_configs = (
        ("mass", 0.91, max(5.0, max_dim / 27.0), 6.0, 0.94, 0.24, 0.18, 260),
        ("form", 0.985, max(2.8, max_dim / 68.0), 5.0, 0.92, 0.58, 0.42, 420),
        ("accent", 0.997, max(1.6, max_dim / 120.0), 4.0, 0.95, 1.0, 0.78, 480),
    )
    for (
        phase,
        target_coverage,
        radius,
        length_factor,
        opacity,
        weight,
        color_weight,
        limit,
    ) in phase_configs:
        if phase == "mass" and not strokes:
            sx = min(sim_w - 1, max(0, int(round(start_point[0] * sim_w))))
            sy = min(sim_h - 1, max(0, int(round(start_point[1] * sim_h))))
            add_stroke(
                phase,
                (sx, sy),
                radius,
                radius * 2.4,
                opacity,
                forced_start=start_point,
            )
        for _ in range(limit):
            current_coverage, _ = quick_values()
            phase_done = current_coverage >= target_coverage
            if phase == "accent":
                if phase_done and _ % 12 == 0:
                    current = metrics()
                    current_regions = region_metrics()
                    phase_done = (
                        current["importantCoverage"] >= 0.997
                        and current["largestHoleRatio"] <= 0.001
                        and current["importantMeanDeltaE76"] <= 11.2
                        and all(
                            region["coverage"] >= 0.998
                            and region["meanDeltaE76"] <= 11.2
                            for region in current_regions
                        )
                    )
                else:
                    phase_done = False
            if phase_done:
                break
            field = residual_field(weight, color_weight, max(0.65, radius * 0.32))
            py, px = np.unravel_index(int(np.argmax(field)), field.shape)
            add_stroke(
                phase,
                (int(px), int(py)),
                radius,
                radius * length_factor,
                opacity,
            )

    edges = cv2.Canny(gray, 55, 145)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    eligible = [
        contour
        for contour in contours
        if cv2.arcLength(contour, True) >= max_dim * 0.22
        and float(foreground[contour[:, 0, 1], contour[:, 0, 0]].mean()) >= 0.72
    ]
    if eligible:

        def contour_score(contour: np.ndarray) -> float:
            points = contour[:, 0, :]
            values = importance[points[:, 1], points[:, 0]]
            return float(cv2.arcLength(contour, True)) * (0.55 + float(values.mean()))

        outer_contour = max(eligible, key=contour_score)
        epsilon = max(0.8, cv2.arcLength(outer_contour, True) * 0.004)
        simplified = cv2.approxPolyDP(outer_contour, epsilon, True)[:, 0, :]
        outer_points = [(float(x) / sim_w, float(y) / sim_h) for x, y in simplified]
    else:
        outer_points = []
    face = importance_regions[0]["box"]
    fx, fy, fw, fh = face
    face_feature_contours: list[np.ndarray] = []
    face_x0, face_y0 = int(fx * sim_w), int(fy * sim_h)
    face_x1, face_y1 = int((fx + fw) * sim_w), int((fy + fh) * sim_h)
    for contour in contours:
        pts = contour[:, 0, :]
        center_x, center_y = pts[:, 0].mean(), pts[:, 1].mean()
        if not (face_x0 <= center_x <= face_x1 and face_y0 <= center_y <= face_y1):
            continue
        if not (
            face_y0 + (face_y1 - face_y0) * 0.24
            <= center_y
            <= face_y0 + (face_y1 - face_y0) * 0.84
        ):
            continue
        if (
            cv2.arcLength(contour, False) >= max_dim * 0.025
            and float(foreground[pts[:, 1], pts[:, 0]].mean()) >= 0.9
        ):
            face_feature_contours.append(contour)
    if face_feature_contours:
        feature = max(
            face_feature_contours, key=lambda contour: cv2.arcLength(contour, False)
        )
        feature = cv2.approxPolyDP(feature, 0.9, False)[:, 0, :]
        feature_points = [(float(x) / sim_w, float(y) / sim_h) for x, y in feature]
    else:
        feature_points = []

    def add_lock(
        role: str, points: list[tuple[float, float]], width_px: float, opacity: float
    ) -> None:
        if len(points) < 2:
            return
        if len(points) > 80:
            step = max(1, len(points) // 72)
            points = points[::step]
        pixel_points = np.array(
            [
                [
                    min(sim_w - 1, max(0, int(round(x * sim_w)))),
                    min(sim_h - 1, max(0, int(round(y * sim_h)))),
                ]
                for x, y in points
            ],
            dtype=np.int32,
        )
        samples = image[pixel_points[:, 1], pixel_points[:, 0]]
        sample_labs = target_lab[pixel_points[:, 1], pixel_points[:, 0]]
        darkness = sample_labs[:, 0]
        dark_indices = np.argsort(darkness)[: max(1, len(darkness) // 3)]
        lock_bgr = np.median(samples[dark_indices], axis=0)
        lock_lab_cv = np.median(sample_labs[dark_indices], axis=0)
        lock_color = (
            f"#{int(lock_bgr[2]):02x}{int(lock_bgr[1]):02x}{int(lock_bgr[0]):02x}"
        )
        before_coverage, before_residual = quick_values()
        lock_mask = np.zeros((sim_h, sim_w), dtype=np.uint8)
        cv2.polylines(
            lock_mask,
            [pixel_points],
            False,
            255,
            max(1, int(round(width_px))),
            lineType=cv2.LINE_AA,
        )
        lock_alpha = lock_mask.astype(np.float32) / 255.0 * opacity
        coverage[:] = coverage + lock_alpha * (1.0 - coverage)
        current_lab[:] = (
            current_lab * (1.0 - lock_alpha[..., None])
            + lock_lab_cv * lock_alpha[..., None]
        )
        after_coverage, after_residual = quick_values()
        strokes.append(
            {
                "pass": "final-lock",
                "phase": "final-lock",
                "role": role,
                "points": [[round(x, 5), round(y, 5)] for x, y in points],
                "x0": round(points[0][0], 5),
                "y0": round(points[0][1], 5),
                "x1": round(points[-1][0], 5),
                "y1": round(points[-1][1], 5),
                "width": round(width_px / sim_w, 5),
                "opacity": opacity,
                "sampledColor": lock_color,
                "color": lock_color,
                "lab": [round(value, 2) for value in _cielab_triplet(lock_lab_cv)],
                "importance": round(float(importance[lock_mask > 0].mean()), 4),
                "residualBefore": round(before_residual, 5),
                "residualAfter": round(after_residual, 5),
                "coverageBefore": round(before_coverage, 5),
                "coverageAfter": round(after_coverage, 5),
                "boundary": {"source": "edge-contour", "narrowLock": True},
            }
        )

    add_lock("subject-contour", outer_points, 1.0, 0.30)
    add_lock("face-feature-contour", feature_points, 0.55, 0.20)

    before_settle = {**metrics(), "regions": region_metrics()}
    hole_map = np.logical_and(foreground, coverage < 0.5).astype(np.uint8)
    labels, _, stats, centroids = cv2.connectedComponentsWithStats(hole_map, 8)
    hole_candidates = sorted(
        (
            (int(stats[label, cv2.CC_STAT_AREA]), label)
            for label in range(1, labels)
            if int(stats[label, cv2.CC_STAT_AREA])
            <= max(12, int(sim_w * sim_h * 0.0015))
        ),
        reverse=True,
    )[:16]
    for area, label in hole_candidates:
        cx, cy = centroids[label]
        radius = min(2.2, max(0.8, math.sqrt(area / math.pi) * 0.72))
        add_stroke(
            "settle-correction",
            (int(round(cx)), int(round(cy))),
            radius,
            radius * 2.4,
            0.88,
            target=settle_strokes,
            correction_reason="small-hole-gap-close",
        )
    for _ in range(520):
        current = metrics()
        regions_now = region_metrics()
        if (
            current["overallCoverage"] >= 0.993
            and current["importantCoverage"] >= 0.995
            and current["largestHoleRatio"] <= 0.0015
            and current["importantMeanDeltaE76"] <= 12.0
            and all(
                region["coverage"] >= 0.995 and region["meanDeltaE76"] <= 12.0
                for region in regions_now
            )
        ):
            break
        if current["largestHoleRatio"] > 0.0015:
            holes_now = np.logical_and(foreground, coverage < 0.5).astype(np.uint8)
            label_count, label_map, stats, _ = cv2.connectedComponentsWithStats(
                holes_now, 8
            )
            largest_label = 1 + int(np.argmax(stats[1:label_count, cv2.CC_STAT_AREA]))
            largest_component = (label_map == largest_label).astype(np.uint8)
            distance = cv2.distanceTransform(
                largest_component, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            py, px = np.unravel_index(int(np.argmax(distance)), distance.shape)
            reason = "residual-gap-close"
        else:
            delta = _delta_e76_map(current_lab, target_lab)
            correction_field = delta * (0.55 + importance) + (1.0 - coverage) * 95.0
            for region, mask in zip(regions_now, importance_masks, strict=True):
                if region["coverage"] < 0.995 or region["meanDeltaE76"] > 12.0:
                    correction_field[mask] *= 1.8
            py, px = np.unravel_index(
                int(np.argmax(correction_field)), correction_field.shape
            )
            reason = (
                "residual-gap-close"
                if coverage[int(py), int(px)] < 0.5
                else "boundary-local-color-correction"
            )
        add_stroke(
            "settle-correction",
            (int(px), int(py)),
            max(2.2, max_dim / 100.0),
            max(5.0, max_dim / 36.0),
            0.98,
            target=settle_strokes,
            correction_reason=reason,
        )
    after_settle = {**metrics(), "regions": region_metrics()}

    phase_ratios = {"mass": 0.42, "form": 0.28, "accent": 0.22, "final-lock": 0.08}
    cursor = start_ms
    paint_index = 0
    phase_timing: dict[str, dict[str, float | int]] = {}
    for phase in ("mass", "form", "accent", "final-lock"):
        selected = [stroke for stroke in strokes if stroke["phase"] == phase]
        phase_duration = paint_ms * phase_ratios[phase]
        phase_end = cursor + phase_duration
        minimum_window = max(1.0, 3000.0 / fps)
        visible_events = (
            min(len(selected), max(1, int(phase_duration / minimum_window)))
            if selected
            else 0
        )
        if selected:
            for batch in range(visible_events):
                begin = round(batch * len(selected) / visible_events)
                end = round((batch + 1) * len(selected) / visible_events)
                group = selected[begin:end]
                window_start = cursor + phase_duration * batch / visible_events
                window_end = cursor + phase_duration * (batch + 1) / visible_events
                travel_ms = min(
                    window_end - window_start - 1.0,
                    max(1000.0 / fps, (window_end - window_start) * 0.28),
                )
                contact_start = window_start + travel_ms
                contact_duration = max(1.0, window_end - contact_start)
                for group_index, stroke in enumerate(group):
                    stroke["id"] = f"paint-{paint_index:04d}"
                    stroke["startMs"] = round(contact_start, 2)
                    stroke["durationMs"] = round(contact_duration, 2)
                    stroke["visibleEvent"] = f"{phase}-{batch:03d}"
                    stroke["toolContact"] = group_index == 0
                    paint_index += 1
        phase_timing[phase] = {
            "startMs": round(cursor, 2),
            "endMs": round(phase_end, 2),
            "strokeCount": len(selected),
            "visibleEventCount": visible_events,
            "minimumContactMs": round(2000.0 / fps, 2),
        }
        cursor = phase_end

    targets = {
        "overallCoverage": 0.992,
        "importantCoverage": 0.995,
        "largestHoleRatioMax": 0.0015,
        "importantMeanDeltaE76Max": 12.0,
        "regionCoverage": 0.995,
        "regionMeanDeltaE76Max": 12.0,
    }
    measured = after_settle
    region_pass = all(
        region["coverage"] >= targets["regionCoverage"]
        and region["meanDeltaE76"] <= targets["regionMeanDeltaE76Max"]
        for region in measured["regions"]
    )
    completion = {
        "coverageThreshold": 0.5,
        "targets": targets,
        "measured": measured,
        "passed": bool(
            measured["overallCoverage"] >= targets["overallCoverage"]
            and measured["importantCoverage"] >= targets["importantCoverage"]
            and measured["largestHoleRatio"] <= targets["largestHoleRatioMax"]
            and measured["importantMeanDeltaE76"] <= targets["importantMeanDeltaE76Max"]
            and region_pass
        ),
        "selection": "subject-matte residual, color/edge-bounded streamlines, and region-gated local correction",
        "simulationSize": [sim_w, sim_h],
        "importanceRegions": importance_regions,
    }
    settle_metrics = {
        "strokeCount": len(settle_strokes),
        "smallHoleCorrections": sum(
            stroke.get("correctionReason") == "small-hole-gap-close"
            for stroke in settle_strokes
        ),
        "localColorCorrections": sum(
            stroke.get("correctionReason") == "boundary-local-color-correction"
            for stroke in settle_strokes
        ),
        "residualGapCorrections": sum(
            stroke.get("correctionReason") == "residual-gap-close"
            for stroke in settle_strokes
        ),
        "before": before_settle,
        "after": after_settle,
    }
    return strokes, {
        "phaseTiming": phase_timing,
        "completion": completion,
        "settleStrokes": settle_strokes,
        "settleMetrics": settle_metrics,
        "subjectMatte": {
            **foreground_metadata,
            "simulationForegroundRatio": round(float(foreground.mean()), 5),
            "backgroundTreatment": "blurred low-frequency color field; no contour lock",
            "regionCounts": {
                "skin": int(skin_focus.sum()),
                "hair": int(hair_focus.sum()),
                "foreground": int(foreground.sum()),
            },
        },
    }


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
        if stroke.get("toolContact") is False:
            continue
        events.append(
            {
                "start": float(stroke["startMs"]),
                "duration": float(stroke["durationMs"]),
                "points": stroke.get(
                    "points",
                    [[stroke["x0"], stroke["y0"]], [stroke["x1"], stroke["y1"]]],
                ),
                "control": [stroke["cx"], stroke["cy"]] if "cx" in stroke else None,
                "phase": "contact"
                if "toolContact" in stroke
                else stroke.get("phase", "paint"),
            }
        )
    events.sort(key=lambda e: e["start"])
    with_travel: list[dict[str, Any]] = []
    for event in events:
        if with_travel and event["phase"] == "contact":
            previous = with_travel[-1]
            previous_end = float(previous["start"]) + float(previous["duration"])
            gap = float(event["start"]) - previous_end
            if gap >= 4.0:
                previous_end_point = previous["points"][-1]
                next_start_point = event["points"][0]
                lift_duration = gap * 0.32
                with_travel.append(
                    {
                        "start": previous_end,
                        "duration": lift_duration,
                        "points": [previous_end_point, previous_end_point],
                        "phase": "lift",
                    }
                )
                with_travel.append(
                    {
                        "start": previous_end + lift_duration,
                        "duration": gap - lift_duration,
                        "points": [previous_end_point, next_start_point],
                        "phase": "travel",
                    }
                )
        with_travel.append(event)
    events = sorted(with_travel, key=lambda event: event["start"])

    table: list[dict[str, Any]] = []
    cursor = 0
    last = (0.5, 0.5)
    phase = "ink"
    count = max(1, int(duration_ms / 1000.0 * fps) + 1)
    sample_times = {round(i * 1000.0 / fps, 4) for i in range(count)}
    for event in events:
        sample_times.add(round(float(event["start"]), 4))
        sample_times.add(
            round(float(event["start"]) + float(event["duration"]) * 0.5, 4)
        )
        sample_times.add(round(float(event["start"]) + float(event["duration"]), 4))
    for t in sorted(time for time in sample_times if time <= duration_ms):
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
    release_start_ms: float | None = None,
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
        release_start = (
            duration_ms - release_ms if release_start_ms is None else release_start_ms
        )
        release = max(0.0, min(1.0, (t_ms - release_start) / max(1.0, release_ms)))
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
    color_modes = {
        "none",
        "paint",
        "reveal",
        "sampled-strokes",
        "hybrid-paint",
        "residual-pigment",
    }
    if color_mode not in color_modes:
        raise ValueError(
            "color_mode must be 'none', 'paint', 'reveal', "
            "'sampled-strokes', 'hybrid-paint', or 'residual-pigment'"
        )
    resolved_color_mode = "reveal" if color_mode == "paint" else color_mode
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
    subject_foreground, subject_matte = _subject_foreground_mask(work)
    subject_strokes = [
        stroke
        for stroke in strokes
        if _stroke_foreground_fraction(stroke, subject_foreground) >= 0.68
    ]
    if subject_strokes:
        strokes = subject_strokes
    ordered = _order_strokes(strokes, work_w, work_h)[:max_strokes]
    if not ordered:
        raise RuntimeError(
            "No drawable strokes were extracted; adjust edge thresholds."
        )

    active_ms = duration_ms - hold_ms - photo_fade_ms
    has_color_phase = resolved_color_mode != "none"
    draw_ms = active_ms * (
        0.42
        if resolved_color_mode == "residual-pigment"
        else (0.45 if has_color_phase else 1.0)
    )
    paint_ms = active_ms - draw_ms if has_color_phase else 0.0
    gap_ms = min(28.0, draw_ms * 0.15 / max(1, len(ordered)))
    scheduled = _schedule(ordered, 0.0, draw_ms, gap_ms, "stroke")
    face_detail_mask, face_detail_source = _face_candidate_mask(work)
    face_detail_count = 0
    for stroke in scheduled:
        face_fraction = _stroke_foreground_fraction(
            [(float(x), float(y)) for x, y in stroke["points"]], face_detail_mask
        )
        is_face_detail = face_fraction >= 0.56
        if is_face_detail:
            face_detail_count += 1
        stroke["inkRole"] = "face-detail" if is_face_detail else "subject-outline"
        stroke["inkOpacity"] = 0.44 if is_face_detail else 0.84
        stroke["inkWidth"] = 1.35 if is_face_detail else 2.05
    residual_metadata: dict[str, Any] = {}
    settle_strokes: list[dict[str, Any]] = []
    if resolved_color_mode == "reveal":
        paint_strokes = _paint_brush_strokes(work, draw_ms, paint_ms)
    elif resolved_color_mode in {"sampled-strokes", "hybrid-paint"}:
        final_ink_point = tuple(map(float, scheduled[-1]["points"][-1]))
        paint_strokes = _sampled_color_strokes(work, draw_ms, paint_ms, final_ink_point)
    elif resolved_color_mode == "residual-pigment":
        final_ink_point = tuple(map(float, scheduled[-1]["points"][-1]))
        paint_strokes, residual_metadata = _residual_pigment_strokes(
            work,
            draw_ms,
            paint_ms,
            final_ink_point,
            fps,
            subject_foreground,
        )
        settle_strokes = residual_metadata.pop("settleStrokes")
    else:
        paint_strokes = []

    pigment_end_ms = draw_ms + paint_ms
    settle_start_ms = pigment_end_ms
    settle_end_ms = (
        pigment_end_ms + photo_fade_ms * 0.55
        if resolved_color_mode == "residual-pigment"
        else pigment_end_ms
    )
    tool_lift_start_ms = pigment_end_ms
    tool_lift_end_ms = (
        pigment_end_ms + min(360.0, photo_fade_ms * 0.38)
        if resolved_color_mode == "residual-pigment"
        else pigment_end_ms
    )
    camera_release_start_ms = (
        settle_end_ms
        if resolved_color_mode == "residual-pigment"
        else duration_ms - hold_ms - photo_fade_ms
    )
    camera_release_end_ms = (
        duration_ms - hold_ms
        if resolved_color_mode == "residual-pigment"
        else duration_ms
    )
    if residual_metadata:
        residual_metadata["phaseTiming"].update(
            {
                "settle": {
                    "startMs": round(settle_start_ms, 2),
                    "endMs": round(settle_end_ms, 2),
                },
                "toolLift": {
                    "startMs": round(tool_lift_start_ms, 2),
                    "endMs": round(tool_lift_end_ms, 2),
                },
                "cameraRelease": {
                    "startMs": round(camera_release_start_ms, 2),
                    "endMs": round(camera_release_end_ms, 2),
                },
                "finalHold": {
                    "startMs": round(duration_ms - hold_ms, 2),
                    "endMs": round(duration_ms, 2),
                },
            }
        )
        if settle_strokes:
            settle_duration = max(1.0, settle_end_ms - settle_start_ms)
            visible_ms = max(2000.0 / fps, settle_duration * 0.18)
            for index, stroke in enumerate(settle_strokes):
                start = settle_start_ms + settle_duration * 0.72 * index / max(
                    1, len(settle_strokes) - 1
                )
                stroke["id"] = f"settle-{index:04d}"
                stroke["startMs"] = round(start, 2)
                stroke["durationMs"] = round(min(visible_ms, settle_end_ms - start), 2)

    camera: list[dict[str, float]] = []
    positions = _position_table(scheduled, paint_strokes, duration_ms, fps)
    if resolved_color_mode == "residual-pigment":
        for position in positions:
            t_ms = float(position["t"]) * 1000.0
            if tool_lift_start_ms <= t_ms <= tool_lift_end_ms:
                position["phase"] = "lift"
    if resolved_closeup_mode != "none":
        if resolved_color_mode == "residual-pigment":
            camera = _camera_table(
                positions,
                closeup_zoom,
                duration_ms,
                release_ms=max(1.0, camera_release_end_ms - camera_release_start_ms),
                mode=resolved_closeup_mode,
                release_start_ms=camera_release_start_ms,
            )
        else:
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
            "pigmentEndMs": round(pigment_end_ms, 2),
            "settleStartMs": round(settle_start_ms, 2),
            "settleEndMs": round(settle_end_ms, 2),
            "toolLiftStartMs": round(tool_lift_start_ms, 2),
            "toolLiftEndMs": round(tool_lift_end_ms, 2),
            "cameraReleaseStartMs": round(camera_release_start_ms, 2),
            "cameraReleaseEndMs": round(camera_release_end_ms, 2),
        },
        "vectorization": {
            "workingSize": [work_w, work_h],
            "strokeCount": len(scheduled),
            "totalInkPx": round(sum(s["lengthPx"] for s in scheduled), 1),
            "ordering": "coarse-to-fine, nearest-neighbour pen travel",
            "foregroundDetailFilter": {
                **subject_matte,
                "keptStrokeCount": len(ordered),
                "foregroundFractionMin": 0.68,
            },
            "faceInkTreatment": {
                "source": face_detail_source,
                "faceDetailStrokeCount": face_detail_count,
                "faceDetailOpacity": 0.44,
                "faceDetailWidth": 1.35,
            },
        },
        "coloring": {
            "mode": color_mode,
            "resolvedMode": resolved_color_mode,
            "brushStrokeCount": len(paint_strokes),
            "passes": (
                ["alternating broad wash", "edge-aligned detail"]
                if resolved_color_mode == "reveal"
                else (
                    [
                        "residual-selected mass fields",
                        "structure-aligned form strokes",
                        "importance-weighted accents",
                        "contour completion lock",
                        "deterministic pigment settle",
                    ]
                    if resolved_color_mode == "residual-pigment"
                    else ["Lab-region ordered RGB sampled strokes"]
                    if has_color_phase
                    else []
                )
            ),
            "rendering": (
                "source image revealed through progressive brush-path mask"
                if resolved_color_mode == "reveal"
                else (
                    "actual RGB residual-pigment strokes with deterministic settle; source pixels never enter the rendered canvas"
                    if resolved_color_mode == "residual-pigment"
                    else "actual RGB canvas strokes with a subtle source texture finish"
                    if resolved_color_mode == "hybrid-paint"
                    else "actual RGB canvas strokes; source pixels are not used as a paint mask"
                )
            ),
            "textureMix": (
                0.14
                if resolved_color_mode == "hybrid-paint"
                else (1.0 if resolved_color_mode in {"none", "reveal"} else 0.0)
            ),
            "phaseTiming": residual_metadata.get("phaseTiming", {}),
            "completion": residual_metadata.get("completion"),
            "settleMetrics": residual_metadata.get("settleMetrics"),
            "subjectMatte": residual_metadata.get("subjectMatte", subject_matte),
        },
        "camera": {
            "mode": resolved_closeup_mode,
            "zoom": closeup_zoom,
            "samples": len(camera),
            "speedLimited": bool(camera),
        },
        "strokes": scheduled,
        "paintStrokes": paint_strokes,
        "settleStrokes": settle_strokes,
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
        output_dir / "package.json",
        {
            "name": "open-motion-bridge-sketch",
            "private": True,
            "version": "0.0.0",
            "dependencies": {"gsap": "3.13.0"},
        },
        force,
    )
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
            "runtimeInstall": "npm install",
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
            {
                "id": stroke["id"],
                "s": stroke["startMs"],
                "d": stroke["durationMs"],
                "a": stroke.get("inkOpacity", 0.92),
                "w": stroke.get("inkWidth", 2.1),
            }
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
                "pts": s.get("points"),
                "w": s["width"],
                "a": s["opacity"],
                "p": s["pass"],
                "phase": s.get("phase", "paint"),
                "c": s.get("color", s.get("sampledColor", "#ffffff")),
                "s": s["startMs"],
                "d": s["durationMs"],
            }
            for s in plan["paintStrokes"]
        ],
        separators=(",", ":"),
    )
    settle_json = json.dumps(
        [
            {
                "x0": stroke["x0"],
                "y0": stroke["y0"],
                "x1": stroke["x1"],
                "y1": stroke["y1"],
                "pts": stroke.get("points"),
                "w": stroke["width"],
                "a": stroke["opacity"],
                "p": stroke["pass"],
                "phase": stroke.get("phase", "settle-correction"),
                "c": stroke["color"],
                "s": stroke["startMs"],
                "d": stroke["durationMs"],
            }
            for stroke in plan.get("settleStrokes", [])
        ],
        separators=(",", ":"),
    )
    tool_json = json.dumps(plan["toolTable"], separators=(",", ":"))
    camera_json = json.dumps(plan["cameraTable"], separators=(",", ":"))
    color_mode = str(plan["coloring"].get("resolvedMode", plan["coloring"]["mode"]))
    texture_mix = float(plan["coloring"].get("textureMix", 0.0))
    if color_mode == "reveal":
        paint_renderer_js = """
      const drawPaintLayer = (targetCtx, passName, t) => {
        targetCtx.clearRect(0, 0, W, H);
        let hasPaint = false;
        targetCtx.globalCompositeOperation = 'source-over';
        targetCtx.strokeStyle = '#ffffff';
        targetCtx.lineCap = 'round';
        targetCtx.lineJoin = 'round';
        for (const s of paintStrokes) {
          if (s.p !== passName) continue;
          const p = Math.max(0, Math.min(1, (t - s.s) / s.d));
          if (p <= 0) continue;
          hasPaint = true;
          const e = p * p * (3 - 2 * p);
          const x0 = s.x0 * W, y0 = s.y0 * H;
          let x1, y1, partialCx = null, partialCy = null;
          if (s.cx !== null && s.cy !== null) {
            const controlX = s.cx * W, controlY = s.cy * H;
            partialCx = x0 + (controlX - x0) * e;
            partialCy = y0 + (controlY - y0) * e;
            const nextCx = controlX + (s.x1 * W - controlX) * e;
            const nextCy = controlY + (s.y1 * H - controlY) * e;
            x1 = partialCx + (nextCx - partialCx) * e;
            y1 = partialCy + (nextCy - partialCy) * e;
          } else {
            x1 = (s.x0 + (s.x1 - s.x0) * e) * W;
            y1 = (s.y0 + (s.y1 - s.y0) * e) * H;
          }
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
        }
        targetCtx.shadowBlur = 0;
        if (hasPaint && photo.complete && photo.naturalWidth > 0) {
          targetCtx.globalCompositeOperation = 'source-in';
          targetCtx.globalAlpha = 1;
          targetCtx.drawImage(photo, 0, 0, W, H);
        }
        targetCtx.globalCompositeOperation = 'source-over';
        targetCtx.globalAlpha = 1;
      };
      const drawPaint = (t) => {
        drawPaintLayer(ctx, 'wash', t);
        drawPaintLayer(detailCtx, 'detail', t);
      };
"""
    elif color_mode == "residual-pigment":
        paint_renderer_js = """
      const strokePath = (targetCtx, s, progress, widthScale = 1, alphaScale = 1) => {
        const points = s.pts || [[s.x0, s.y0], [s.x1, s.y1]];
        if (points.length < 2 || progress <= 0) return;
        const lengths = [];
        let total = 0;
        for (let index = 1; index < points.length; index += 1) {
          const dx = (points[index][0] - points[index - 1][0]) * W;
          const dy = (points[index][1] - points[index - 1][1]) * H;
          const length = Math.hypot(dx, dy);
          lengths.push(length);
          total += length;
        }
        let remaining = total * Math.max(0, Math.min(1, progress));
        targetCtx.globalAlpha = s.a * alphaScale;
        targetCtx.strokeStyle = s.c;
        targetCtx.lineWidth = Math.max(1.25, s.w * W * widthScale);
        targetCtx.beginPath();
        targetCtx.moveTo(points[0][0] * W, points[0][1] * H);
        for (let index = 1; index < points.length && remaining > 0; index += 1) {
          const segment = lengths[index - 1];
          const amount = segment > 0 ? Math.min(1, remaining / segment) : 1;
          const x = (points[index - 1][0] + (points[index][0] - points[index - 1][0]) * amount) * W;
          const y = (points[index - 1][1] + (points[index][1] - points[index - 1][1]) * amount) * H;
          targetCtx.lineTo(x, y);
          remaining -= segment;
        }
        targetCtx.stroke();
      };
      const drawPaintLayer = (targetCtx, t) => {
        targetCtx.clearRect(0, 0, W, H);
        targetCtx.globalCompositeOperation = 'source-over';
        targetCtx.lineCap = 'round';
        targetCtx.lineJoin = 'round';
        for (const s of paintStrokes) {
          const progress = Math.max(0, Math.min(1, (t - s.s) / s.d));
          if (progress <= 0) continue;
          const eased = progress * progress * (3 - 2 * progress);
          strokePath(targetCtx, s, eased, 1, 1);
          if (s.phase === 'form' || s.phase === 'accent') {
            strokePath(targetCtx, s, eased, 0.24, 0.22);
          }
        }
        for (const s of settleStrokes) {
          const progress = Math.max(0, Math.min(1, (t - s.s) / s.d));
          if (progress <= 0) continue;
          const eased = progress * progress * (3 - 2 * progress);
          strokePath(targetCtx, s, eased, 1, 1);
        }
        targetCtx.globalAlpha = 1;
      };
      const drawPaint = (t) => {
        drawPaintLayer(ctx, t);
        detailCtx.clearRect(0, 0, W, H);
      };
"""
    else:
        paint_renderer_js = """
      const drawPaintLayer = (targetCtx, t) => {
        targetCtx.clearRect(0, 0, W, H);
        targetCtx.globalCompositeOperation = 'source-over';
        targetCtx.lineCap = 'round';
        targetCtx.lineJoin = 'round';
        for (const s of paintStrokes) {
          const p = Math.max(0, Math.min(1, (t - s.s) / s.d));
          if (p <= 0) continue;
          const e = p * p * (3 - 2 * p);
          const x0 = s.x0 * W, y0 = s.y0 * H;
          const x1 = (s.x0 + (s.x1 - s.x0) * e) * W;
          const y1 = (s.y0 + (s.y1 - s.y0) * e) * H;
          const brushWidth = Math.max(2, s.w * W);
          targetCtx.globalAlpha = s.a;
          targetCtx.strokeStyle = s.c;
          targetCtx.lineWidth = brushWidth;
          targetCtx.shadowBlur = brushWidth * 0.08;
          targetCtx.shadowColor = s.c;
          targetCtx.beginPath();
          targetCtx.moveTo(x0, y0);
          targetCtx.lineTo(x1, y1);
          targetCtx.stroke();
          targetCtx.globalAlpha = s.a * 0.28;
          targetCtx.lineWidth = Math.max(1, brushWidth * 0.13);
          targetCtx.shadowBlur = 0;
          targetCtx.beginPath();
          targetCtx.moveTo(x0, y0 - brushWidth * 0.22);
          targetCtx.lineTo(x1, y1 - brushWidth * 0.22);
          targetCtx.stroke();
        }
        targetCtx.shadowBlur = 0;
        targetCtx.globalAlpha = 1;
      };
      const drawPaint = (t) => {
        drawPaintLayer(ctx, t);
        detailCtx.clearRect(0, 0, W, H);
      };
"""

    if color_mode == "residual-pigment":
        photo_markup = ""
        photo_setup_js = ""
        visual_finish_js = """
        const settle = Math.max(0, Math.min(1, (t - SETTLE_START) / Math.max(1, SETTLE_END - SETTLE_START)));
        const settleEase = settle * settle * (3 - 2 * settle);
        document.getElementById('ink').style.opacity = String(1 - settleEase * 0.48);
"""
        tool_opacity_js = """
        const lift = Math.max(0, Math.min(1, (timeMs - TOOL_LIFT_START) / Math.max(1, TOOL_LIFT_END - TOOL_LIFT_START)));
        const liftEase = lift * lift * (3 - 2 * lift);
        const phaseLift = phase === 'lift' ? 0.42 : (phase === 'travel' ? 0.2 : 0);
        toolNode.style.opacity = String((1 - liftEase) * (phase === 'travel' ? 0.82 : 1));
        const liftY = -(liftEase * 0.055 + phaseLift * 0.018) * H;
        const rotation = (phase === 'ink' ? 34 : 28) + liftEase * 26 + phaseLift * 18;
"""
    else:
        photo_markup = f'<img id="photo" src="assets/{staged_image_name}" alt="" />'
        photo_setup_js = "const photo = document.getElementById('photo');"
        visual_finish_js = f"""
        const fade = Math.max(0, Math.min(1, (t - DRAW_END) / Math.max(1, FADE_END - DRAW_END)));
        const eased = fade * fade * (3 - 2 * fade);
        photo.style.opacity = String(eased * {texture_mix:.3f});
        document.getElementById('ink').style.opacity = String(1 - eased * {0.85 if color_mode == "reveal" else 0.68});
"""
        tool_opacity_js = """
        const liftEase = 0;
        const liftY = 0;
        const rotation = phase === 'paint' ? 28 : 34;
        toolNode.style.opacity = String(Math.max(0, Math.min(1, (DRAW_END - timeMs) / 280)));
"""

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
      #paint {{ opacity: {0.82 if color_mode == "reveal" else 0.96}; }}
      #paint-detail {{ opacity: {0.36 if color_mode == "reveal" else 0}; }}
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
      #tool:not([data-phase='ink']) {{ width: 20px; background: linear-gradient(90deg, #6f5844 0 22%, #b99b75 22% 78%, #634b39 78%); }}
      #tool:not([data-phase='ink'])::after {{ bottom: -9px; height: 12px; background: #7d5a43; clip-path: polygon(0 0, 100% 0, 82% 100%, 18% 100%); }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{duration:.3f}" data-width="{width}" data-height="{height}" data-fps="{fps:g}">
      <div id="paper"></div>
      <div id="scene" class="clip" data-start="0" data-duration="{duration:.3f}" data-track-index="1">
        <div id="camera-world" data-layout-allow-overflow>
          <canvas id="paint" width="{width}" height="{height}"></canvas>
          <canvas id="paint-detail" width="{width}" height="{height}"></canvas>
          {photo_markup}
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
      const settleStrokes = {settle_json};
      const toolFrames = {tool_json};
      const cameraFrames = {camera_json};
      const DRAW_END = {draw_end:.2f}, FADE_END = {fade_end:.2f};
      const SETTLE_START = {float(timing.get("settleStartMs", draw_end)):.2f};
      const SETTLE_END = {float(timing.get("settleEndMs", draw_end)):.2f};
      const TOOL_LIFT_START = {float(timing.get("toolLiftStartMs", draw_end)):.2f};
      const TOOL_LIFT_END = {float(timing.get("toolLiftEndMs", draw_end)):.2f};
      const nodes = strokes.map((s) => {{
        const node = document.getElementById(s.id);
        const len = node.getTotalLength();
        node.style.strokeDasharray = String(len);
        node.style.strokeDashoffset = String(len);
        return {{ meta: s, node, len }};
      }});
      {photo_setup_js}
      const paintCanvas = document.getElementById('paint');
      const ctx = paintCanvas.getContext('2d');
      const detailPaintCanvas = document.getElementById('paint-detail');
      const detailCtx = detailPaintCanvas.getContext('2d');
      const cameraNode = document.getElementById('camera-world');
      const toolNode = document.getElementById('tool');
{paint_renderer_js}
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
        toolNode.dataset.phase = phase;
        {tool_opacity_js}
        toolNode.style.transform = 'translate(' + (x * W).toFixed(2) + 'px,' + (y * H + liftY).toFixed(2) + 'px) translate(-50%,-100%) rotate(' + rotation + 'deg)';
      }};
      const draw = (timeSec) => {{
        const t = timeSec * 1000;
        for (const entry of nodes) {{
          const p = Math.max(0, Math.min(1, (t - entry.meta.s) / entry.meta.d));
          entry.node.style.strokeDashoffset = String(entry.len * (1 - p));
          entry.node.style.opacity = String(entry.meta.a);
          entry.node.style.strokeWidth = String(entry.meta.w);
        }}
        drawPaint(t);
        {visual_finish_js}
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
