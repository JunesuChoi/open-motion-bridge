# Codex integration plan

Codex reads the root `SKILL.md`, routes video work to a candidate EditSpec and photo-motion work to a MotionSpec, validates the chosen contract, and invokes the same local `omb` CLI as every other integration. A still-photo drawing request may invoke `sketch-image` directly after resolving color and close-up choices; recommend `residual-pigment` for residual-driven, source-free RGB mass/form/accent painting with an explicit completion lock and measured finish, retain `sampled-strokes` as the earlier Lab-region traversal, offer `hybrid-paint` for a subtle texture finish, and describe `paint`/`reveal` as legacy source-mask behavior. Generated projects are handed to an external renderer rather than rendered by this Skill.

Codex-specific instructions must not change Tracking IR, patch semantics, auto-review thresholds, the external-render boundary, or the permission/approval gate. They may add UX-oriented prompt examples after the CLI and schema exist.
