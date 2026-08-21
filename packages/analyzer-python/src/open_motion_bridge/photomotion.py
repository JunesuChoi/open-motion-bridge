"""Photo-to-motion-graphics: single-image analysis plus a declarative MotionSpec.

A photo is treated as a one-observation Tracking IR. Motion is synthesized, not
observed: a MotionSpec declares a duration and motion primitives, the resolver
samples them onto a render-FPS grid, and the output is the same auditable
per-frame transform table that video bindings use, so the external renderer and
the reprojection verifier work unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from .bindings import _parse_binding, resolve_bindings
from .pipeline import (
    SCHEMA_VERSION,
    _binding_elements_markup,
    _binding_render_payload,
    _mediapipe_landmarks,
    _bbox,
    _sha256,
    _stage_binding_assets,
    _utc_now,
    _write_json,
)

_EASES = {"linear", "inout"}
_MOTION_KINDS = {"camera-path", "attach", "tile-reveal"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid MotionSpec: {message}")


def _ease(name: str, ratio: float) -> float:
    ratio = max(0.0, min(1.0, ratio))
    if name == "inout":
        return ratio * ratio * (3.0 - 2.0 * ratio)
    return ratio


def analyze_image(image_path: Path, output_dir: Path, force: bool = False) -> None:
    """Create a single-observation Tracking IR from a still photo."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"OpenCV could not decode image: {image_path}")
    height, width = frame.shape[0], frame.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        landmarks = _mediapipe_landmarks(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    observations: list[dict[str, Any]] = []
    if landmarks:
        mean_visibility = sum(item["visibility"] for item in landmarks) / len(landmarks)
        observations.append(
            {
                "frameIndex": 0,
                "sourceTimeMs": 0.0,
                "confidence": round(mean_visibility, 6),
                "occlusion": "none" if mean_visibility >= 0.5 else "partial",
                "screenSpace": {"bbox": _bbox(landmarks), "keypoints": landmarks},
                "quality": {
                    "interpolated": False,
                    "driftWarning": False,
                    "manualCorrectionRequired": mean_visibility < 0.5,
                },
            }
        )

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": _utc_now(),
        "source": {
            "sourceHash": _sha256(image_path),
            "fileName": image_path.name,
            "mediaKind": "image",
            "displayWidth": width,
            "displayHeight": height,
            "durationMs": 0.0,
        },
    }
    ir = {
        "schemaVersion": SCHEMA_VERSION,
        "source": manifest["source"],
        "coordinateSystem": {
            "unit": "normalized",
            "screenSpace": "x-right-y-down, [0,1] relative to display orientation",
            "stabilizedSpace": "not-applicable-for-still-image",
        },
        "frames": [{"frameIndex": 0, "sourceTimeMs": 0.0, "decodeStatus": "decoded"}],
        "tracks": [
            {
                "id": "person-001",
                "type": "pose",
                "provider": {
                    "name": "mediapipe-pose",
                    "version": getattr(mp, "__version__", "unknown"),
                    "modelIdentifier": "Pose(static_image_mode=True, model_complexity=2)",
                },
                "lifecycle": {"firstFrame": 0 if observations else None, "lastFrame": 0 if observations else None, "reidentifiedFrom": [], "idChanges": []},
                "observations": observations,
            }
        ],
        "cameraMotion": [],
        "analysis": {
            "status": "completed" if observations else "completed-with-no-pose-detections",
            "sampledFrames": 1,
            "poseDetections": len(observations),
            "limitations": ["Single still image; all motion must come from a MotionSpec, none was observed."],
        },
        "provenance": [
            {"kind": "analysis", "createdAt": _utc_now(), "tool": "open-motion-bridge", "toolVersion": SCHEMA_VERSION, "sourceHash": manifest["source"]["sourceHash"]}
        ],
    }
    _write_json(output_dir / "source-manifest.json", manifest, force)
    _write_json(output_dir / "tracking.ir.json", ir, force)


def load_motion_spec(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"MotionSpec not found: {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(spec, dict), "top level value must be an object")
    _require(spec.get("schemaVersion") == SCHEMA_VERSION, f"unsupported schemaVersion: {spec.get('schemaVersion')!r}")
    duration_ms = float(spec.get("durationMs", 0))
    _require(duration_ms > 0, "durationMs must be greater than zero")
    fps = float(spec.get("fps", 30.0))
    _require(fps > 0, "fps must be greater than zero")
    motions = spec.get("motions")
    _require(isinstance(motions, list) and bool(motions), "motions must be a non-empty list")
    camera_count = 0
    for index, motion in enumerate(motions):
        _require(isinstance(motion, dict), f"motion #{index} is not an object")
        kind = motion.get("kind")
        _require(kind in _MOTION_KINDS, f"motion #{index} kind must be one of {sorted(_MOTION_KINDS)}")
        if kind == "camera-path":
            camera_count += 1
            _require(camera_count == 1, "only one camera-path motion is allowed")
            _require(str(motion.get("ease", "inout")) in _EASES, "camera-path ease must be linear or inout")
        if kind == "tile-reveal":
            _require(int(motion.get("rows", 1)) >= 1 and int(motion.get("cols", 1)) >= 1, "tile-reveal rows/cols must be >= 1")
    return spec


def _camera_table(spec: dict[str, Any], frame_times_ms: list[float]) -> list[dict[str, float]]:
    camera = next((m for m in spec["motions"] if m["kind"] == "camera-path"), None)
    duration_ms = float(spec["durationMs"])
    table: list[dict[str, float]] = []
    for t in frame_times_ms:
        if camera is None:
            table.append({"t": round(t / 1000.0, 6), "s": 1.0, "cx": 0.5, "cy": 0.5})
            continue
        start = float(camera.get("startMs", 0.0))
        end = float(camera.get("endMs", duration_ms))
        ratio = _ease(str(camera.get("ease", "inout")), (t - start) / max(1e-6, end - start))
        f, to = camera.get("from", {}), camera.get("to", {})
        s = float(f.get("zoom", 1.0)) + (float(to.get("zoom", 1.0)) - float(f.get("zoom", 1.0))) * ratio
        cx = float(f.get("cx", 0.5)) + (float(to.get("cx", 0.5)) - float(f.get("cx", 0.5))) * ratio
        cy = float(f.get("cy", 0.5)) + (float(to.get("cy", 0.5)) - float(f.get("cy", 0.5))) * ratio
        table.append({"t": round(t / 1000.0, 6), "s": round(s, 6), "cx": round(cx, 6), "cy": round(cy, 6)})
    return table


def _compose_camera(bindings_payload: dict[str, Any], camera: list[dict[str, float]]) -> None:
    """Fold the camera transform into every binding frame so the resolved table is final screen space."""
    for binding in bindings_payload["bindings"]:
        for frame, cam in zip(binding["frames"], camera, strict=True):
            s, cx, cy = cam["s"], cam["cx"], cam["cy"]
            frame["x"] = round(0.5 + (frame["x"] - cx) * s, 6)
            frame["y"] = round(0.5 + (frame["y"] - cy) * s, 6)
            frame["size"] = round(frame["size"] * s, 3)


def generate_photo_project(
    ir_path: Path,
    source_image: Path,
    motion_spec_path: Path,
    output_dir: Path,
    force: bool = False,
) -> None:
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    if ir.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported IR schema: {ir.get('schemaVersion')}")
    if ir.get("source", {}).get("mediaKind") != "image":
        raise ValueError("generate-photo requires an IR produced by analyze-image (mediaKind: image).")
    if not source_image.is_file():
        raise FileNotFoundError(f"Source image not found: {source_image}")
    spec = load_motion_spec(motion_spec_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration_ms = float(spec["durationMs"])
    fps = float(spec.get("fps", 30.0))
    frame_count = max(1, int(duration_ms / 1000.0 * fps + 0.999999))
    frame_times = [i * 1000.0 / fps for i in range(frame_count)]

    track = next((t for t in ir.get("tracks", []) if t.get("type") == "pose"), None)
    base = track["observations"][0] if track and track.get("observations") else None
    attach_raw = [m for m in spec["motions"] if m["kind"] == "attach"]
    if attach_raw and base is None:
        raise RuntimeError("MotionSpec has attach motions but the image analysis found no pose.")

    width = int(ir["source"]["displayWidth"])
    height = int(ir["source"]["displayHeight"])
    camera = _camera_table(spec, frame_times)

    bindings_payload: dict[str, Any] | None = None
    if attach_raw:
        # A photo is a video whose only observation repeats: replicate it on the render
        # grid and reuse the video binding resolver unchanged.
        synthetic_ir = {
            "tracks": [
                {
                    "type": "pose",
                    "observations": [dict(base, sourceTimeMs=t) for t in frame_times],
                }
            ]
        }
        parsed = [_parse_binding(dict(m, kind=m.get("assetKind", "text")), i) for i, m in enumerate(attach_raw)]
        bindings_payload = resolve_bindings(parsed, synthetic_ir, width, height)
        bindings_payload["sourceMotionSpec"] = motion_spec_path.name
        bindings_payload["sourceHash"] = ir["source"]["sourceHash"]
        _stage_binding_assets(bindings_payload, motion_spec_path, output_dir / "assets")
        _compose_camera(bindings_payload, camera)

    assets = output_dir / "assets"
    assets.mkdir(exist_ok=True)
    staged = assets / ("source" + source_image.suffix.lower())
    if staged.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {staged}; use --force for generated output.")
    import shutil

    shutil.copy2(source_image, staged)

    tile = next((m for m in spec["motions"] if m["kind"] == "tile-reveal"), None)
    html = _render_photo_html(ir, spec, camera, bindings_payload, tile, staged.name)
    html_path = output_dir / "index.html"
    if html_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {html_path}; use --force for generated output.")
    html_path.write_text(html, encoding="utf-8")
    if bindings_payload is not None:
        _write_json(output_dir / "bindings.resolved.json", bindings_payload, force)
    _write_json(
        output_dir / "open-motion-bridge.generated.json",
        {
            "schemaVersion": SCHEMA_VERSION,
            "createdAt": _utc_now(),
            "sourceIr": ir_path.name,
            "sourceHash": ir["source"]["sourceHash"],
            "mediaKind": "image",
            "target": "hyperframes-photo",
            "compositionDurationMs": duration_ms,
            "renderFps": fps,
            "motions": [m["kind"] for m in spec["motions"]],
            "bindings": [
                {"id": b["id"], "kind": b["kind"], "stats": b["stats"]}
                for b in (bindings_payload or {}).get("bindings", [])
            ],
        },
        force,
    )


def _render_photo_html(
    ir: dict[str, Any],
    spec: dict[str, Any],
    camera: list[dict[str, float]],
    bindings_payload: dict[str, Any] | None,
    tile: dict[str, Any] | None,
    staged_image_name: str,
) -> str:
    width = int(ir["source"]["displayWidth"])
    height = int(ir["source"]["displayHeight"])
    duration = float(spec["durationMs"]) / 1000.0
    fps = float(spec.get("fps", 30.0))
    camera_json = json.dumps(camera, separators=(",", ":"))
    binding_markup = _binding_elements_markup(bindings_payload)
    bindings_json = json.dumps(_binding_render_payload(bindings_payload), separators=(",", ":"))

    tile_markup = ""
    tile_json = "null"
    if tile is not None:
        rows, cols = int(tile.get("rows", 4)), int(tile.get("cols", 4))
        pieces = []
        for r in range(rows):
            for c in range(cols):
                pieces.append(
                    f'<div class="omb-tile" data-r="{r}" data-c="{c}" style="left:{c * 100 / cols:.4f}%;top:{r * 100 / rows:.4f}%;width:{100 / cols:.4f}%;height:{100 / rows:.4f}%;'
                    f'background-image:url(assets/{staged_image_name});background-size:{cols * 100}% {rows * 100}%;'
                    f'background-position:{(c * 100 / max(1, cols - 1)) if cols > 1 else 0:.4f}% {(r * 100 / max(1, rows - 1)) if rows > 1 else 0:.4f}%;"></div>'
                )
        tile_markup = "".join(pieces)
        tile_json = json.dumps(
            {"rows": rows, "cols": cols, "startMs": float(tile.get("startMs", 0.0)), "durationMs": float(tile.get("durationMs", 1400.0)), "staggerMs": float(tile.get("staggerMs", 45.0))},
            separators=(",", ":"),
        )

    return f'''<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={width}, height={height}" />
    <title>Open Motion Bridge photo motion</title>
    <script src="./node_modules/gsap/dist/gsap.min.js"></script>
    <style>
      * {{ box-sizing: border-box; }}
      html, body {{ margin: 0; width: {width}px; height: {height}px; overflow: hidden; background: #17130f; }}
      #root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; font-family: Arial, sans-serif; }}
      .clip {{ position: absolute; inset: 0; }}
      #camera {{ position: absolute; left: 0; top: 0; width: {width}px; height: {height}px; transform-origin: 0 0; }}
      #photo {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }}
      #tile-layer {{ position: absolute; inset: 0; }}
      .omb-tile {{ position: absolute; opacity: 0; }}
      #binding-layer {{ position: absolute; inset: 0; pointer-events: none; }}
      .omb-binding {{ position: absolute; left: 0; top: 0; opacity: 0; transform-origin: 50% 50%; will-change: transform, opacity; white-space: nowrap; }}
      .omb-binding-text {{ line-height: 1; paint-order: stroke fill; }}
      .omb-binding-image {{ object-fit: contain; }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{duration:.3f}" data-width="{width}" data-height="{height}" data-fps="{fps:g}">
      <div id="camera" class="clip" data-start="0" data-duration="{duration:.3f}" data-track-index="1">
        <img id="photo" src="assets/{staged_image_name}" alt="" />
        <div id="tile-layer">{tile_markup}</div>
      </div>
      <div id="binding-layer" class="clip" data-start="0" data-duration="{duration:.3f}" data-track-index="2">{binding_markup}</div>
    </div>
    <script>
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true }});
      const W = {width}, H = {height};
      const cameraFrames = {camera_json};
      const tileSpec = {tile_json};
      const bindingTracks = {bindings_json};
      const frameAt = (frames, time, key) => {{
        let lo = 0, hi = frames.length - 1;
        while (lo < hi) {{ const mid = Math.ceil((lo + hi) / 2); if (frames[mid][key] <= time) lo = mid; else hi = mid - 1; }}
        const a = frames[lo], b = frames[Math.min(lo + 1, frames.length - 1)];
        if (!a || !b || a === b || b[key] <= a[key]) return a;
        const r = Math.max(0, Math.min(1, (time - a[key]) / (b[key] - a[key])));
        const out = {{}};
        for (const k of Object.keys(a)) out[k] = typeof a[k] === 'number' ? a[k] + (b[k] - a[k]) * r : a[k];
        return out;
      }};
      const cameraNode = document.getElementById('camera');
      const applyCamera = (time) => {{
        const f = frameAt(cameraFrames, time, 't');
        if (!f) return;
        cameraNode.style.transform = 'translate(' + ((0.5 - f.cx * f.s) * W).toFixed(3) + 'px,' + ((0.5 - f.cy * f.s) * H).toFixed(3) + 'px) scale(' + f.s.toFixed(6) + ')';
      }};
      const tiles = Array.from(document.querySelectorAll('.omb-tile'));
      const photoNode = document.getElementById('photo');
      const applyTiles = (time) => {{
        if (!tileSpec) return;
        const tMs = time * 1000;
        let allDone = true;
        for (const node of tiles) {{
          const r = +node.dataset.r, c = +node.dataset.c;
          const order = r * tileSpec.cols + c;
          const start = tileSpec.startMs + order * tileSpec.staggerMs;
          const p = Math.max(0, Math.min(1, (tMs - start) / tileSpec.durationMs));
          const e = p * p * (3 - 2 * p);
          node.style.opacity = String(e);
          node.style.transform = 'translateY(' + ((1 - e) * 40).toFixed(2) + 'px)';
          if (e < 1) allDone = false;
        }}
        photoNode.style.opacity = allDone ? '1' : '0';
      }};
      const bindingNodes = bindingTracks.map((track) => {{
        const node = document.querySelector('[data-binding-id="' + track.id + '"]');
        if (node && track.kind === 'text') node.style.fontSize = '100px';
        return {{ track, node, measured: 0 }};
      }});
      const applyBindings = (time) => {{
        for (const entry of bindingNodes) {{
          if (!entry.node) continue;
          const f = frameAt(entry.track.frames, time, 't');
          if (!f) continue;
          if (f.opacity <= 0) {{ entry.node.style.opacity = '0'; continue; }}
          const sizePx = f.size;
          let w = sizePx, h = sizePx / (entry.track.aspect || 1);
          if (entry.track.kind === 'text') {{
            if (!entry.measured) entry.measured = entry.node.getBoundingClientRect().width / 100 || 1;
            entry.node.style.fontSize = sizePx.toFixed(3) + 'px';
            w = sizePx * entry.measured; h = sizePx;
          }} else {{
            entry.node.style.width = w.toFixed(3) + 'px';
            entry.node.style.height = h.toFixed(3) + 'px';
          }}
          entry.node.style.opacity = String(f.opacity);
          entry.node.style.transform = 'translate(' + (f.x * W - w / 2).toFixed(3) + 'px,' + (f.y * H - h / 2).toFixed(3) + 'px) rotate(' + (f.rotation || 0).toFixed(3) + 'deg)';
        }}
      }};
      const state = {{ time: 0 }};
      const drawAll = (time) => {{ applyCamera(time); applyTiles(time); applyBindings(time); }};
      drawAll(0);
      tl.to(state, {{ time: {duration:.3f}, duration: {duration:.3f}, ease: 'none', onUpdate: () => drawAll(state.time) }}, 0);
      window.__timelines['main'] = tl;
    </script>
  </body>
</html>
'''
