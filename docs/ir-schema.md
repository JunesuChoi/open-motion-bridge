# Tracking IR schema

Tracking IR is a renderer-independent, immutable observation record. It stores what analysis observed, not what a user later wants to stylize.

## Required top-level shape

```json
{
  "schemaVersion": "0.1.0",
  "source": {
    "sourceHash": "sha256:...",
    "displayWidth": 1920,
    "displayHeight": 1080,
    "nominalFps": 30,
    "variableFrameRate": false,
    "durationMs": 10000,
    "timebase": "1/90000"
  },
  "coordinateSystem": {
    "unit": "normalized",
    "screenSpace": "x-right-y-down, [0,1] relative to display orientation",
    "stabilizedSpace": "camera-compensated normalized coordinates"
  },
  "frames": [],
  "tracks": [],
  "cameraMotion": [],
  "analysis": {},
  "provenance": []
}
```

## Frame record

Each sampled frame identifies both source time and decode order.

```json
{
  "frameIndex": 42,
  "sourceTimeMs": 1400,
  "presentationTimestamp": 126000,
  "isKeyframe": false,
  "decodeStatus": "decoded"
}
```

`sourceTimeMs` is canonical. An implementation must not infer timing as `frameIndex / nominalFps` for variable-frame-rate media.

## Track record

```json
{
  "id": "person-001",
  "type": "pose",
  "provider": {
    "name": "mediapipe-pose",
    "version": "recorded-at-runtime",
    "modelIdentifier": "provider-defined",
    "licenseHint": "provider-defined"
  },
  "lifecycle": {
    "firstFrame": 0,
    "lastFrame": 299,
    "reidentifiedFrom": [],
    "idChanges": []
  },
  "observations": [
    {
      "frameIndex": 42,
      "sourceTimeMs": 1400,
      "confidence": 0.95,
      "occlusion": "none",
      "screenSpace": {
        "bbox": { "x": 0.32, "y": 0.14, "width": 0.21, "height": 0.61 },
        "keypoints": [
          { "name": "left_wrist", "x": 0.38, "y": 0.51, "z": -0.02, "visibility": 0.91 }
        ]
      },
      "stabilizedSpace": {
        "bbox": { "x": 0.30, "y": 0.15, "width": 0.21, "height": 0.61 },
        "keypoints": [
          { "name": "left_wrist", "x": 0.36, "y": 0.52, "z": -0.02, "visibility": 0.91 }
        ]
      },
      "quality": {
        "interpolated": false,
        "driftWarning": false,
        "manualCorrectionRequired": false
      }
    }
  ]
}
```

`type` is one of `pose`, `object`, or `roi-object`. A pose track uses named landmarks; object tracks may omit `keypoints` and use a bbox, mask reference, contour, or path geometry according to the published minor schema.

## Camera motion record

```json
{
  "frameIndex": 42,
  "sourceTimeMs": 1400,
  "method": "feature-homography",
  "sourceToStabilized": [1, 0, -0.02, 0, 1, 0.01, 0, 0, 1],
  "confidence": 0.84,
  "inlierCount": 128,
  "warnings": []
}
```

The exact matrix convention must be declared in schema metadata and never inferred from array length alone.

## Quality summary

The resolver computes, but does not overwrite, source observations with quality summaries:

```json
{
  "trackId": "person-001",
  "averageConfidence": 0.91,
  "continuity": 0.98,
  "hasDriftWarning": false,
  "hasTrackIdChange": false,
  "hasOcclusion": false,
  "reviewRecommendation": "auto-approve"
}
```

## Render Tracking IR

`tracking.ir.json` is immutable raw provider evidence. Generation creates a sibling `render.tracking.ir.json`; it is a derived, renderer-facing artifact and never overwrites the raw observations.

```json
{
  "tracks": [{ "id": "person-001-render", "type": "pose" }],
  "temporalProcessing": {
    "algorithm": "one-euro-filter + confidence-aware linear interpolation",
    "profile": "balanced",
    "renderFps": 30,
    "visibilityThreshold": 0.2,
    "maxGapMs": 250,
    "rawIrMutable": false
  }
}
```

Short per-landmark confidence losses may hold the last filtered position and are explicitly marked as interpolated. Long gaps are emitted with zero visibility and `manualCorrectionRequired: true`; the resolver must not invent confident coordinates.

## Schema evolution

- Additive compatible fields increment the minor version.
- Breaking changes increment the major version and require a migration tool.
- Consumers must reject unknown major versions and retain unrecognized additive fields when round-tripping.
- The published JSON Schema is the executable source of truth; examples in this document are explanatory.
