import json

from open_motion_bridge.pipeline import _render_html, _render_svg


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
