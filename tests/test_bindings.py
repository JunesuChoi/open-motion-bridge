import json

import pytest

from open_motion_bridge.bindings import load_edit_spec, resolve_bindings


def _write_spec(tmp_path, bindings):
    spec = tmp_path / "edits.spec.json"
    spec.write_text(
        json.dumps({"schemaVersion": "0.1.0", "bindings": bindings}), encoding="utf-8"
    )
    return spec


def _render_ir(observations):
    return {"tracks": [{"type": "pose", "observations": observations}]}


def _observation(t_ms, x, y, visibility=0.9):
    return {
        "sourceTimeMs": t_ms,
        "screenSpace": {
            "bbox": {"x": x - 0.1, "y": y - 0.1, "width": 0.2, "height": 0.4},
            "keypoints": [
                {"name": "nose", "x": x, "y": y, "z": 0.0, "visibility": visibility},
                {"name": "left_wrist", "x": x + 0.1, "y": y + 0.2, "z": 0.0, "visibility": visibility},
            ],
        },
    }


def test_edit_spec_rejects_unknown_policy(tmp_path):
    spec = _write_spec(
        tmp_path,
        [{"id": "a", "kind": "text", "text": "HI", "anchor": {"landmarks": ["nose"]}, "onLowConfidence": "explode"}],
    )
    with pytest.raises(ValueError, match="onLowConfidence"):
        load_edit_spec(spec)


def test_edit_spec_rejects_duplicate_ids(tmp_path):
    binding = {"id": "a", "kind": "text", "text": "HI", "anchor": {"landmarks": ["nose"]}}
    spec = _write_spec(tmp_path, [binding, dict(binding)])
    with pytest.raises(ValueError, match="unique"):
        load_edit_spec(spec)


def test_resolved_binding_follows_anchor_and_scales_with_bbox(tmp_path):
    spec = _write_spec(
        tmp_path,
        [
            {
                "id": "face",
                "kind": "text",
                "text": "MARK",
                "anchor": {"landmarks": ["nose"]},
                "scale": {"mode": "bbox-height", "value": 0.5},
            }
        ],
    )
    bindings = load_edit_spec(spec)
    resolved = resolve_bindings(
        bindings, _render_ir([_observation(0, 0.4, 0.3), _observation(100, 0.6, 0.35)]), 1000, 2000
    )
    frames = resolved["bindings"][0]["frames"]
    assert frames[0]["x"] == pytest.approx(0.4)
    assert frames[1]["x"] == pytest.approx(0.6, abs=0.11)
    assert frames[0]["opacity"] == 1.0
    # bbox height 0.4 of a 2000px frame -> 800px reference, scaled by 0.5.
    assert frames[0]["size"] == pytest.approx(400.0)


def test_low_confidence_fade_hides_untrustworthy_anchor(tmp_path):
    spec = _write_spec(
        tmp_path,
        [{"id": "a", "kind": "text", "text": "HI", "anchor": {"landmarks": ["nose"]}, "onLowConfidence": "fade"}],
    )
    bindings = load_edit_spec(spec)
    resolved = resolve_bindings(
        bindings,
        _render_ir([_observation(0, 0.5, 0.5, visibility=0.9), _observation(100, 0.5, 0.5, visibility=0.1)]),
        1000,
        1000,
    )
    frames = resolved["bindings"][0]["frames"]
    assert frames[0]["opacity"] == 1.0
    assert frames[1]["opacity"] == 0.0
    assert frames[1]["state"] == "hidden-low-confidence"


def test_max_speed_clamps_single_frame_jumps(tmp_path):
    spec = _write_spec(
        tmp_path,
        [{"id": "a", "kind": "text", "text": "HI", "anchor": {"landmarks": ["nose"]}, "maxSpeed": 1.0}],
    )
    bindings = load_edit_spec(spec)
    resolved = resolve_bindings(
        bindings,
        _render_ir([_observation(0, 0.1, 0.5), _observation(100, 0.9, 0.5)]),
        1000,
        1000,
    )
    frames = resolved["bindings"][0]["frames"]
    # 0.8 normalized in 0.1s exceeds maxSpeed 1.0/s; movement is clamped to 0.1.
    assert frames[1]["x"] == pytest.approx(0.2)
    assert resolved["bindings"][0]["stats"]["clampedFrames"] == 1


def test_missing_landmark_is_reported_not_guessed(tmp_path):
    spec = _write_spec(
        tmp_path,
        [{"id": "a", "kind": "text", "text": "HI", "anchor": {"landmarks": ["left_hand_00"]}}],
    )
    bindings = load_edit_spec(spec)
    resolved = resolve_bindings(bindings, _render_ir([_observation(0, 0.5, 0.5)]), 1000, 1000)
    frame = resolved["bindings"][0]["frames"][0]
    assert frame["state"] == "missing-landmark"
    assert frame["opacity"] == 0.0
