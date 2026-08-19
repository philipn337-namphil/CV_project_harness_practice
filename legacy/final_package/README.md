# Final Package Reference

This directory preserves the final packaged implementation used to recover the canonical model contract.

It is kept as a historical reference and should not be used as the primary development path.

## Canonical Development Path

The active canonical model implementation is:

```text
src/khuda_cv/model.py
```

## Historical Reference Files

```text
model.py
pipeline.py
requirements.txt
```

These files represent the final packaged inference/deployment version that was used together with the trained model artifact.

## Important

Do not modify these files during normal refactoring.

When behavior needs to change:

1. Modify the canonical implementation under `src/`.
2. Run the verification harness.
3. Confirm checkpoint compatibility.
4. Keep this directory unchanged unless intentionally updating the historical reference.

## Legacy Relationship

```text
seraph/
-> earlier training / experiment implementation

legacy/final_package/
-> final packaged historical reference

src/khuda_cv/
-> canonical development implementation
```
