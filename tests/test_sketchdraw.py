import json

import cv2
import numpy as np
import pytest

from open_motion_bridge.sketchdraw import (
    _camera_table,
    _paint_brush_strokes,
    _position_table,
    generate_sketch_project,
)


def _synthetic_portrait(width=320, height=400):
    image = np.full((height, width, 3), (226, 220, 210), dtype=np.uint8)
    cv2.ellipse(
        image, (width // 2, height // 2), (85, 125), 0, 0, 360, (170, 185, 210), -1
    )
    cv2.circle(image, (width // 2 - 30, height // 2 - 20), 9, (35, 35, 35), -1)
    cv2.circle(image, (width // 2 + 30, height // 2 - 20), 9, (35, 35, 35), -1)
    cv2.line(
        image,
        (width // 2 - 30, height // 2 + 45),
        (width // 2 + 30, height // 2 + 45),
        (60, 70, 140),
        6,
    )
    return image


def test_paint_plan_uses_long_brush_paths_and_is_deterministic():
    image = _synthetic_portrait()
    first = _paint_brush_strokes(image, 1000.0, 4000.0)
    second = _paint_brush_strokes(image, 1000.0, 4000.0)

    assert first == second
    assert {stroke["pass"] for stroke in first} == {"wash", "detail"}
    wash = [stroke for stroke in first if stroke["pass"] == "wash"]
    assert wash
    assert all(abs(stroke["x1"] - stroke["x0"]) > 0.9 for stroke in wash)
    assert all(
        first[index]["startMs"] <= first[index + 1]["startMs"]
        for index in range(len(first) - 1)
    )
    assert first[-1]["startMs"] + first[-1]["durationMs"] == pytest.approx(
        5000.0, abs=0.1
    )


def test_tool_position_interpolates_and_camera_releases_to_full_frame():
    ink = [
        {
            "startMs": 0.0,
            "durationMs": 1000.0,
            "points": [[0.1, 0.2], [0.9, 0.2]],
        }
    ]
    paint = [
        {
            "startMs": 1000.0,
            "durationMs": 1000.0,
            "x0": 0.9,
            "y0": 0.2,
            "x1": 0.2,
            "y1": 0.8,
        }
    ]
    positions = _position_table(ink, paint, duration_ms=3000.0, fps=10.0)
    midpoint = next(
        position for position in positions if position["t"] == pytest.approx(0.5)
    )
    assert midpoint["x"] == pytest.approx(0.5)
    assert midpoint["phase"] == "ink"

    camera = _camera_table(
        positions, 1.8, 3000.0, release_ms=1000.0, mode="phase-focus"
    )
    assert camera[-1] == {"t": 3.0, "s": 1.0, "cx": 0.5, "cy": 0.5}
    for frame in camera:
        half = 0.5 / frame["s"]
        assert half - 1e-5 <= frame["cx"] <= 1.0 - half + 1e-5
        assert half - 1e-5 <= frame["cy"] <= 1.0 - half + 1e-5


def test_generated_project_uses_brush_mask_and_explicit_closeup_mode(tmp_path):
    source = tmp_path / "synthetic.png"
    assert cv2.imwrite(str(source), _synthetic_portrait())
    output = tmp_path / "project"

    plan = generate_sketch_project(
        source,
        output,
        duration_ms=5000.0,
        fps=12.0,
        hold_ms=700.0,
        photo_fade_ms=600.0,
        max_strokes=120,
        color_mode="paint",
        closeup_mode="phase-focus",
        closeup_zoom=1.6,
    )
    html = (output / "index.html").read_text(encoding="utf-8")
    saved_plan = json.loads((output / "sketch.plan.json").read_text(encoding="utf-8"))
    motion = json.loads((output / "index.motion.json").read_text(encoding="utf-8"))

    assert plan["camera"]["mode"] == "phase-focus"
    assert plan["coloring"]["brushStrokeCount"] > 0
    assert saved_plan["paintStrokes"] == plan["paintStrokes"]
    assert "globalCompositeOperation = 'source-in'" in html
    assert "targetCtx.lineTo" in html
    assert "ctx.arc" not in html
    assert 'id="camera-world"' in html
    assert 'id="tool"' in html
    assert 'id="paint-detail"' in html
    assert motion["assertions"][0] == {
        "kind": "appearsBy",
        "selector": "#tool",
        "bySec": 0.2,
    }
    assert motion["assertions"][1]["kind"] == "keepsMoving"


def test_closeup_mode_and_zoom_conflict_fails_loudly(tmp_path):
    source = tmp_path / "synthetic.png"
    assert cv2.imwrite(str(source), _synthetic_portrait())
    with pytest.raises(ValueError, match="closeup_zoom requires"):
        generate_sketch_project(
            source,
            tmp_path / "project",
            color_mode="none",
            closeup_mode="none",
            closeup_zoom=1.5,
        )
