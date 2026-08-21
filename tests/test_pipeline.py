import json

from open_motion_bridge.pipeline import (
    _interpolate_render_observations,
    _render_html,
    _render_svg,
    _sample_step,
    _smooth_pose_observations,
    _temporal_smoothing_config,
)


def _ir(render_duration_ms=1000):
    point = {"name": "nose", "x": 0.5, "y": 0.25, "z": 0.0, "visibility": 0.9}
    return {
        "source": {
            "displayWidth": 1080,
            "displayHeight": 1920,
            "durationMs": 1000,
            "renderDurationMs": render_duration_ms,
            "sourceHash": "sha256:test",
        },
        "tracks": [{"observations": [{"sourceTimeMs": 0, "screenSpace": {"keypoints": [point]}}]}],
    }


def test_hyperframes_output_has_deterministic_contract():
    content = _render_html(_ir(), [{"time": 0, "points": [[0.5, 0.25, 0.9]]}], "source")
    assert 'data-composition-id="main"' in content
    assert 'data-start="0"' in content
    assert 'data-duration="1.000"' in content
    assert "window.__timelines['main']" in content
    assert 'id="pose-canvas"' in content
    assert "const interpolatePose" in content
    assert "Math.random" not in content


def test_hyperframes_uses_safe_render_duration_not_longer_pts_duration():
    content = _render_html(_ir(render_duration_ms=900), [{"time": 0, "points": [[0.5, 0.25, 0.9]]}], "source")
    assert 'data-duration="0.900"' in content
    assert "time: 0.900, duration: 0.900" in content


def test_sketch_svg_carries_timing_data():
    content = _render_svg(_ir(), [(0.0, 1.0, "<circle />")])
    assert 'data-start="0.000"' in content
    assert "Open Motion Bridge exact pose trace" in content
    assert json.loads(json.dumps({"ok": True})) == {"ok": True}


def _observation(timestamp_ms, x, visibility=0.95):
    point = {"name": "nose", "x": x, "y": 0.25, "z": 0.0, "visibility": visibility}
    return {
        "frameIndex": round(timestamp_ms / 10),
        "sourceTimeMs": timestamp_ms,
        "confidence": visibility,
        "occlusion": "none",
        "screenSpace": {"bbox": {"x": x, "y": 0.25, "width": 0, "height": 0}, "keypoints": [point]},
        "quality": {"manualCorrectionRequired": False},
    }


def test_native_sampling_keeps_every_decodable_source_frame():
    assert _sample_step(59.94, 0) == 1
    assert _sample_step(59.94, 30) == 2


def test_temporal_processor_smooths_then_interpolates_to_render_grid():
    config = _temporal_smoothing_config("stable", render_fps=20, visibility_threshold=0.2, max_gap_ms=250)
    raw = [_observation(0, 0.1), _observation(50, 0.9)]
    smoothed = _smooth_pose_observations(raw, config)
    assert 0.1 < smoothed[1]["screenSpace"]["keypoints"][0]["x"] < 0.9

    render = _interpolate_render_observations(smoothed, duration_ms=100, config=config)
    assert [item["sourceTimeMs"] for item in render] == [0.0, 50.0]
    assert render[1]["quality"]["temporalSmoothingApplied"] is True


def test_temporal_processor_hides_long_missing_intervals_instead_of_inventing_confidence():
    config = _temporal_smoothing_config("balanced", render_fps=10, visibility_threshold=0.2, max_gap_ms=100)
    raw = [_observation(0, 0.1), _observation(500, 0.9)]
    render = _interpolate_render_observations(_smooth_pose_observations(raw, config), duration_ms=500, config=config)
    assert render[2]["sourceTimeMs"] == 200.0
    assert render[2]["screenSpace"]["keypoints"][0]["visibility"] == 0.0
    assert render[2]["quality"]["manualCorrectionRequired"] is True


def test_temporal_interpolation_matches_keypoints_by_name_not_array_position():
    config = _temporal_smoothing_config("balanced", render_fps=20, visibility_threshold=0.2, max_gap_ms=250)
    left = _observation(0, 0.1)
    right = _observation(100, 0.9)
    left["screenSpace"]["keypoints"].append({"name": "left_eye", "x": 0.2, "y": 0.3, "z": 0.0, "visibility": 0.9})
    right["screenSpace"]["keypoints"].append({"name": "left_eye", "x": 0.8, "y": 0.3, "z": 0.0, "visibility": 0.9})
    right["screenSpace"]["keypoints"].reverse()

    render = _interpolate_render_observations([left, right], duration_ms=100, config=config)
    midpoint = {point["name"]: point for point in render[1]["screenSpace"]["keypoints"]}
    assert midpoint["nose"]["x"] == 0.5
    assert midpoint["left_eye"]["x"] == 0.5
