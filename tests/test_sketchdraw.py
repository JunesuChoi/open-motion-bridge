import json

import cv2
import numpy as np
import pytest

from open_motion_bridge.sketchdraw import (
    _camera_table,
    _paint_brush_strokes,
    _position_table,
    _residual_pigment_strokes,
    _sampled_color_strokes,
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


def test_sampled_color_plan_is_rgb_driven_deterministic_and_starts_near_pen():
    image = _synthetic_portrait()
    start = (0.82, 0.18)
    first = _sampled_color_strokes(image, 1000.0, 4000.0, start)
    second = _sampled_color_strokes(image, 1000.0, 4000.0, start)

    assert first == second
    assert first
    assert {stroke["pass"] for stroke in first} == {"sampled"}
    assert all(stroke["color"].startswith("#") for stroke in first)
    assert all(stroke["sampledColor"] == stroke["color"] for stroke in first)
    assert all(len(stroke["lab"]) == 3 for stroke in first)
    distance = np.hypot(first[0]["x0"] - start[0], first[0]["y0"] - start[1])
    assert distance < 0.08
    assert first[-1]["startMs"] + first[-1]["durationMs"] == pytest.approx(
        5000.0, abs=0.1
    )


def test_residual_pigment_plan_is_deterministic_multiphase_and_completion_driven():
    image = _synthetic_portrait()
    start = (0.82, 0.18)
    first, first_meta = _residual_pigment_strokes(image, 1000.0, 4000.0, start)
    second, second_meta = _residual_pigment_strokes(image, 1000.0, 4000.0, start)

    assert first == second
    assert first_meta == second_meta
    assert first
    assert first[0]["x0"] == pytest.approx(start[0], abs=1e-5)
    assert first[0]["y0"] == pytest.approx(start[1], abs=1e-5)
    phases = [stroke["phase"] for stroke in first]
    assert list(dict.fromkeys(phases)) == ["mass", "form", "accent", "final-lock"]
    assert all(stroke["color"].startswith("#") for stroke in first)
    assert all(len(stroke["lab"]) == 3 for stroke in first)
    assert all("importance" in stroke for stroke in first)
    assert all("residualBefore" in stroke for stroke in first)
    assert all("coverageAfter" in stroke for stroke in first)
    final_locks = [stroke for stroke in first if stroke["phase"] == "final-lock"]
    assert {stroke["role"] for stroke in final_locks} == {
        "subject-contour",
        "face-oval",
        "face-feature-contour",
    }
    assert all(len(stroke["points"]) >= 2 for stroke in final_locks)
    assert all(stroke["boundary"]["narrowLock"] for stroke in final_locks)
    assert final_locks[-1]["startMs"] + final_locks[-1]["durationMs"] == pytest.approx(
        5000.0, abs=0.02
    )
    assert first_meta["completion"]["passed"]
    measured = first_meta["completion"]["measured"]
    targets = first_meta["completion"]["targets"]
    assert measured["overallCoverage"] >= targets["overallCoverage"]
    assert measured["importantCoverage"] >= targets["importantCoverage"]
    assert measured["largestHoleRatio"] <= targets["largestHoleRatioMax"]
    assert measured["importantMeanDeltaE76"] <= targets["importantMeanDeltaE76Max"]
    assert all(
        region["coverage"] >= targets["regionCoverage"]
        and region["meanDeltaE76"] <= targets["regionMeanDeltaE76Max"]
        for region in measured["regions"]
    )
    assert first_meta["settleStrokes"]
    assert first_meta["settleMetrics"]["strokeCount"] == len(
        first_meta["settleStrokes"]
    )
    assert first_meta["settleMetrics"]["after"] == measured
    assert all(
        stroke["pass"] == "settle-correction"
        and stroke["boundary"]["allowedDeltaE76"] <= 6.0
        and "clippedPixelRatio" in stroke["boundary"]
        for stroke in first_meta["settleStrokes"]
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
    package = json.loads((output / "package.json").read_text(encoding="utf-8"))

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
    assert package["dependencies"] == {"gsap": "3.13.0"}


@pytest.mark.parametrize(
    ("color_mode", "expected_texture_mix"),
    [("sampled-strokes", 0.0), ("hybrid-paint", 0.14)],
)
def test_generated_sampled_project_paints_colors_without_source_mask(
    tmp_path, color_mode, expected_texture_mix
):
    source = tmp_path / "synthetic.png"
    assert cv2.imwrite(str(source), _synthetic_portrait())
    output = tmp_path / color_mode

    plan = generate_sketch_project(
        source,
        output,
        duration_ms=5000.0,
        fps=12.0,
        hold_ms=700.0,
        photo_fade_ms=600.0,
        max_strokes=120,
        color_mode=color_mode,
        closeup_mode="phase-focus",
        closeup_zoom=1.6,
    )
    html = (output / "index.html").read_text(encoding="utf-8")

    assert plan["coloring"]["resolvedMode"] == color_mode
    assert plan["coloring"]["textureMix"] == expected_texture_mix
    assert plan["paintStrokes"]
    assert all("color" in stroke for stroke in plan["paintStrokes"])
    assert "globalCompositeOperation = 'source-in'" not in html
    assert "targetCtx.strokeStyle = s.c" in html
    assert '"c":"#' in html


def test_generated_residual_project_has_lock_settle_lift_and_delayed_camera_release(
    tmp_path,
):
    source = tmp_path / "synthetic.png"
    assert cv2.imwrite(str(source), _synthetic_portrait())
    output = tmp_path / "residual"

    plan = generate_sketch_project(
        source,
        output,
        duration_ms=6000.0,
        fps=20.0,
        hold_ms=900.0,
        photo_fade_ms=800.0,
        max_strokes=120,
        color_mode="residual-pigment",
        closeup_mode="phase-focus",
        closeup_zoom=1.6,
    )
    html = (output / "index.html").read_text(encoding="utf-8")
    coloring = plan["coloring"]
    phases = coloring["phaseTiming"]

    assert coloring["resolvedMode"] == "residual-pigment"
    assert coloring["textureMix"] == 0.0
    assert coloring["completion"]["passed"]
    assert phases["final-lock"]["endMs"] == pytest.approx(
        plan["timing"]["pigmentEndMs"], abs=0.02
    )
    assert phases["settle"]["startMs"] == phases["final-lock"]["endMs"]
    assert phases["cameraRelease"]["startMs"] >= phases["settle"]["endMs"]
    assert phases["finalHold"]["startMs"] == phases["cameraRelease"]["endMs"]
    before_release = [
        frame
        for frame in plan["cameraTable"]
        if frame["t"] * 1000 <= phases["cameraRelease"]["startMs"]
    ][-1]
    assert before_release["s"] > 1.0
    assert plan["cameraTable"][-1] == {
        "t": 6.0,
        "s": 1.0,
        "cx": 0.5,
        "cy": 0.5,
    }
    assert "globalCompositeOperation = 'source-in'" not in html
    assert "drawImage(" not in html
    assert 'id="photo"' not in html
    assert "assets/source" not in html
    assert "strokePath(targetCtx" in html
    assert "SETTLE_START" in html
    assert "TOOL_LIFT_START" in html


def test_legacy_paint_alias_resolves_to_reveal(tmp_path):
    source = tmp_path / "synthetic.png"
    assert cv2.imwrite(str(source), _synthetic_portrait())
    plan = generate_sketch_project(
        source,
        tmp_path / "legacy",
        duration_ms=5000.0,
        hold_ms=700.0,
        photo_fade_ms=600.0,
        color_mode="paint",
    )
    assert plan["coloring"]["mode"] == "paint"
    assert plan["coloring"]["resolvedMode"] == "reveal"
    assert plan["coloring"]["textureMix"] == 1.0


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
