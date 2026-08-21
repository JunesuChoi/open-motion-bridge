# EditSpec contract

An EditSpec is the validated, declarative interpretation of a user or agent instruction. It is neither generated source code nor an opportunity to execute arbitrary commands.

## Resolution flow

```text
natural-language request
  -> agent interpretation
  -> candidate EditSpec
  -> schema + semantic validation
  -> visible summary / approval where required
  -> edits.patch.json
  -> resolved.ir.json
```

The agent must report ambiguous target selection, unsupported effect names, overlapping conflicting edits, unavailable analysis data, or risky automatic-review requests. It must not invent a target or silently drop an instruction.

## Core shape

```json
{
  "schemaVersion": "0.1.0",
  "sourceIrHash": "sha256:...",
  "reviewPolicy": "manual",
  "profile": "youtube-shorts-9x16",
  "operations": [
    {
      "id": "op-001",
      "timeRange": { "startMs": 2000, "endMs": 5000 },
      "selector": { "trackId": "person-001", "landmark": "left_wrist" },
      "action": "emphasize-trajectory",
      "parameters": { "coordinateSpace": "screenSpace", "strokeWidth": 8 },
      "sourceInstruction": "2초부터 5초까지 왼손 궤적을 강조해줘"
    }
  ]
}
```

## Selectors

- `trackId`: exact IR ID, such as `person-001`.
- `landmark`: provider-normalized name such as `left_wrist`; it is invalid for a bbox-only object track.
- `roiId`: user-drawn ROI identity recorded by the verifier.
- `semanticLabel`: a candidate label only. It must resolve to one and only one approved track before execution.
- `allApprovedTracks`: explicit opt-in for a multi-track operation.

## Supported action families

| Family | Examples |
| --- | --- |
| Visibility | include, exclude, fade, reveal, hide-range |
| Motion emphasis | emphasize-trajectory, trail, freeze, delay, slow-motion-representation |
| Coordinate behavior | use-screen-space, use-stabilized-space, follow-camera, pin-to-world |
| Style | motion-overlay, exact-svg-trace, rough-sketch, freehand-sketch, minimal-ui |
| Layout | reframe-subject, apply-safe-area, crop-follow-target |
| Review | manual-review, auto-review, review-risk-only |

An action can be added in a future version without interpreting it as JavaScript, CSS, HTML, or shell code.

## Conflict handling

1. Operations are ordered and have stable IDs.
2. Incompatible operations on the same target/time range require an explicit priority or user confirmation.
3. `exclude` is blocking until a later explicit `include` references the excluded operation.
4. A request for `auto-review` cannot override a blocking quality condition; it can only propose auto-review for eligible tracks.
5. A profile change recomputes layout operations and must be shown in the resolved preview.

## Example requests

| Natural-language request | Required EditSpec interpretation |
| --- | --- |
| “2초부터 5초까지만 스케치로” | Range selection plus `exact-svg-trace` and `rough-sketch` style operations |
| “왼손 움직임을 강조해줘” | Resolve an approved pose track, select `left_wrist`, add `emphasize-trajectory` |
| “빨간 공만 따라가” | Require a uniquely resolved object label or user ROI; otherwise request selection |
| “카메라 흔들림을 없애” | `use-stabilized-space`; warn if camera confidence is insufficient |
| “화면에 붙어서 같이 움직이게” | `use-screen-space` or `follow-camera` depending on the resolved target |
| “인스타 릴스로 만들고 인물을 중앙에” | `instagram-reel-9x16` profile plus `reframe-subject` |
| “유튜브 가로 버전과 숏츠 버전을 둘 다” | Produce two named profile jobs from the same resolved IR |
| “지금 자동으로 승인해” | Request `auto-review`; retain failures for manual review |
| “사물은 빼고 사람만 남겨” | `exclude` all approved object tracks with visible target list |
| “더 자유롭게, 만화처럼 보이게” | Produce a candidate style plan; require explicit supported style parameters before generator execution |

## Patch record

`edits.patch.json` records validated operations, actor (`user`, `agent`, or `system`), timestamp, schema version, source IR hash, approval state, and any override reason. The system resolves patches into a new derived IR; it never edits raw analysis observations in place.
