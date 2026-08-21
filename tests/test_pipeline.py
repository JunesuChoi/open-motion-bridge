import json

import pytest

from open_motion_bridge.cli import main
from open_motion_bridge import pipeline
from open_motion_bridge.pipeline import (
    COCO_WHOLEBODY_KEYPOINT_NAMES,
    MMPoseOptions,
    _interpolate_render_observations,
    _mmpose_candidates,
    _mmpose_landmarks,
    _render_html,
    _render_svg,
    _sample_step,
    _select_mmpose_subject,
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
    content = _render_html(_ir(), [{"time": 0, "points": {"nose": [0.5, 0.25, 0.9]}}], "source")
    assert 'data-composition-id="main"' in content
    assert 'data-start="0"' in content
    assert 'data-duration="1.000"' in content
    assert "window.__timelines['main']" in content
    assert 'id="pose-canvas"' in content
    assert "const interpolatePose" in content
    assert "Math.random" not in content


def test_hyperframes_uses_safe_render_duration_not_longer_pts_duration():
    content = _render_html(
        _ir(render_duration_ms=900),
        [{"time": 0, "points": {"nose": [0.5, 0.25, 0.9]}}],
        "source",
    )
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


def _mmpose_keypoints():
    return [[index * 10.0, index * 5.0] for index in range(133)]


def test_mmpose_wholebody_normalizer_exposes_all_canonical_named_keypoints():
    landmarks = _mmpose_landmarks([_mmpose_keypoints()], [[0.9] * 133], width=1000, height=500)

    assert len(COCO_WHOLEBODY_KEYPOINT_NAMES) == 133
    assert len(landmarks) == 133
    assert landmarks[0] == {"name": "nose", "x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.9}
    assert landmarks[9]["name"] == "left_wrist"
    assert landmarks[10]["name"] == "right_wrist"
    assert landmarks[23]["name"] == "face_00"
    assert landmarks[91]["name"] == "left_hand_00"
    assert landmarks[112]["name"] == "right_hand_00"


def test_mmpose_candidate_selection_prefers_iou_after_initial_subject_choice():
    candidates = [
        {"confidence": 0.9, "bbox": {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.5}},
        {"confidence": 0.8, "bbox": {"x": 0.6, "y": 0.1, "width": 0.3, "height": 0.5}},
    ]
    initial, initial_iou = _select_mmpose_subject(candidates, None)
    continued, continued_iou = _select_mmpose_subject(candidates, {"x": 0.58, "y": 0.1, "width": 0.3, "height": 0.5})

    assert initial is candidates[0]
    assert initial_iou is None
    assert continued is candidates[1]
    assert continued_iou and continued_iou > 0.8


def test_mmpose_prediction_payload_requires_the_expected_wholebody_shape():
    result = {"predictions": [[{"keypoints": _mmpose_keypoints(), "keypoint_scores": [0.8] * 133}]]}
    candidates = _mmpose_candidates(result, width=1000, height=500)
    assert len(candidates) == 1
    assert candidates[0]["landmarks"][10]["name"] == "right_wrist"

    with pytest.raises(RuntimeError, match="133 COCO-WholeBody"):
        _mmpose_landmarks([[0.0, 0.0]], [0.8], width=1000, height=500)


def test_cli_rejects_mmpose_without_explicit_local_assets_before_video_ingest():
    with pytest.raises(ValueError, match="requires explicit local assets"):
        main(
            [
                "analyze",
                "placeholder.mp4",
                "--output",
                "placeholder-output",
                "--pose-provider",
                "mmpose-rtmpose-l-wholebody",
            ]
        )


def test_mmpose_runtime_failure_happens_before_ingest_artifact_creation(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"not-decoded")
    options = MMPoseOptions(
        pose_config=tmp_path / "pose.py",
        pose_weights=tmp_path / "pose.pth",
        detector_config=tmp_path / "detector.py",
        detector_weights=tmp_path / "detector.pth",
        device="cpu",
    )
    for asset in (options.pose_config, options.pose_weights, options.detector_config, options.detector_weights):
        asset.write_text("placeholder", encoding="utf-8")
    def unavailable_runtime(_: MMPoseOptions):
        raise RuntimeError("MMPose runtime unavailable")

    monkeypatch.setattr(pipeline, "_create_mmpose_inferencer", unavailable_runtime)

    output = tmp_path / "analysis"
    with pytest.raises(RuntimeError, match="runtime unavailable"):
        pipeline.analyze_video(
            video,
            output,
            sample_fps=1,
            force=False,
            pose_provider="mmpose-rtmpose-l-wholebody",
            mmpose_options=options,
        )
    assert not output.exists()
