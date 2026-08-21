# Claude integration plan

Claude reads the root `SKILL.md`, turns a user's request into a candidate EditSpec, validates it against the published schema, and invokes the same local `omb` CLI as every other integration.

Claude-specific instructions must not change Tracking IR, patch semantics, auto-review thresholds, or the permission/approval gate. They may add UX-oriented prompt examples after the CLI and schema exist.
