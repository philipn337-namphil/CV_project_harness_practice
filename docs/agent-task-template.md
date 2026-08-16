# Agent Task Template

Use this template when giving a future agent a scoped task in this repository.

## Goal

Describe the single outcome the agent should produce.

## Context

Summarize the relevant repository state, prior decisions, and contract documents to read first.

Required context documents when relevant:

- `AGENTS.md`
- `docs/architecture.md`
- `docs/data-contract.md`
- `docs/checkpoint-contract.md`
- `docs/execution-baseline.md`

## Target

List the exact files, modules, tests, or docs the agent should inspect or modify.

## Constraints

State the boundaries clearly:

- PROTECTED areas
- REFACTOR areas
- NEW areas
- Whether production code may be edited
- Whether checkpoints, large files, or data paths may be changed

## Expected Behavior

Describe the desired runtime behavior, shapes, labels, inputs, outputs, and user-visible effects.

Include canonical model details when relevant:

- `GarbageDumpingVideoMAE`
- input `(B, 16, 3, 224, 224)`
- output dict with `clip_logits`, `frame_logits`, `last_hidden_state`
- labels `0 = legal`, `1 = illegal`

## Verification

List the exact checks the agent should run.

Examples:

```powershell
git status
git diff --stat
python -m pytest
```

If a check is expected to fail because of environment setup, ask the agent to record the blocker and continue with available verification.

## Scope

Define what is in scope and what is out of scope.

## Do Not

List forbidden actions.

Examples:

- Do not rewrite unrelated production code.
- Do not change label semantics.
- Do not overwrite `best_model.pt`.
- Do not commit large videos, checkpoints, or local cache files.
- Do not collapse 48-frame source windows into the 16-frame model input contract.

## Definition of Done

The task is complete when:

- The requested files are updated.
- The canonical contract is preserved.
- Verification was run or documented as blocked.
- The diff contains only expected changes.
- The final response summarizes changes, checks, and remaining blockers.

## Final Report

Ask the agent to report:

- What changed
- What was verified
- What was intentionally not changed
- Any known blockers or follow-up tasks
