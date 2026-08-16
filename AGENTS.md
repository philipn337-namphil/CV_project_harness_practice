# AGENTS.md

This repository is a harness practice project for the KHUDA_173 garbage dumping detection pipeline. The first responsibility of any agent is to preserve the verified baseline and contracts before editing production code.

## Canonical Contract

- Canonical model: `GarbageDumpingVideoMAE`
- Legacy training model: `VideoMAEMultiHead`
- Model input: `pixel_values` with shape `(B, 16, 3, 224, 224)`
- Normalization: ImageNet mean/std
- Model output: dict with:
  - `clip_logits`: `(B, 2)`
  - `frame_logits`: `(B, 16)`
  - `last_hidden_state`
- Labels:
  - `0 = legal`
  - `1 = illegal`
- Canonical dataset/batch interface: dict with `pixel_values`, `clip_label`, `frame_labels`, `clip_id`
- Source video windows and model input frames are different concepts. A source clip/window may cover 48 frames, but the VideoMAE input must be sampled down to 16 frames.

## Change Boundaries

### PROTECTED

Do not change these casually:

- `legacy/final_package/model.py`
- `legacy/final_package/pipeline.py`
- `legacy/final_package/requirements.txt`
- Existing checkpoint semantics, especially `best_model.pt`
- Existing dataset and manifest label semantics
- Existing production training/evaluation files under `seraph/` unless the task explicitly targets them

Protected does not mean untouchable forever. It means changes need a clear reason, a small diff, and verification against the contracts in `docs/`.

### REFACTOR

These areas are allowed refactor targets after the baseline is protected:

- Unifying train/test model construction around `GarbageDumpingVideoMAE`
- Moving hard-coded Linux paths into config, CLI args, or environment variables
- Standardizing all dataloaders on the dict batch interface
- Separating source video window length from model input frame count
- Aligning frame labels with the canonical 16-frame model output
- Adding dependency files and smoke checks

### NEW

Prefer new harness work in isolated areas:

- `tests/`
- `harness/`
- Small fixture manifests or generated tensors
- Documentation under `docs/`

New harness code should prove contracts before refactoring production code.

## Before Editing

1. Run `git status` and inspect existing changes.
2. Run `git diff` for tracked changes.
3. Identify whether the requested work touches PROTECTED, REFACTOR, or NEW areas.
4. Read the relevant contract document in `docs/`.
5. Keep user-created files, copied packages, checkpoints, and artifacts intact unless the user explicitly asks otherwise.

## While Editing

- Keep edits narrow and contract-driven.
- Do not rewrite production code while doing documentation or harness-only tasks.
- Do not change label meanings.
- Do not change model tensor shapes without updating tests and docs in the same commit.
- Do not silently convert dict batches back to tuple batches.
- Do not merge the 48-frame source window concept with the 16-frame model input concept.

## After Editing

1. Check the file list or tree.
2. Run `git status`.
3. Run `git diff --stat`.
4. Inspect key diffs.
5. Verify no unrelated production code changed.
6. Stage only files that belong to the task.
7. Commit with a message that describes the user-visible outcome.

## Checkpoint Protection

`best_model.pt` is treated as a compatibility anchor. The verified checkpoint contains `model_state_dict` with:

- 182 `backbone.*` keys
- `clip_head.0.*`
- `clip_head.3.*`
- `frame_head.0.*`
- `frame_head.3.*`

This structure is compatible with `GarbageDumpingVideoMAE`. Do not rename these modules, change head layer structure, or change output class count without a migration plan and a checkpoint compatibility check.

## Large Artifact Policy

Do not commit large generated files or private/local data:

- checkpoints: `*.pt`, `*.pth`, `*.ckpt`, `checkpoints/`
- model runs: `yolo_runs/`, experiment logs
- videos and archives: `*.mp4`, `*.avi`, `*.mov`, `*.mkv`, `*.zip`, `*.tar`, `*.tar.gz`
- Python caches: `__pycache__/`, `*.pyc`

If a large artifact is needed to reproduce behavior, document its path, checksum, or expected location instead of committing it.

## Definition of Done

A task is done only when:

- The canonical contract is preserved or intentionally updated with documentation.
- Production code changes, if any, are limited to the requested scope.
- Checkpoint compatibility is not broken.
- The source window vs 16-frame model input distinction remains clear.
- Verification commands were run or skipped with a documented reason.
- `git status` shows only expected changes before commit, and clean status after commit/push when requested.
