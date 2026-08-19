# Day 4 - Canonicalization and First Safe Refactor

Day 4 focused on using the verification harness from Day 3 as a safety net for the first behavior-preserving refactor. The goal was to stop treating legacy folders as the active development surface and establish a clear canonical model path under `src/`.

## 1. Day 3 Baseline

Before changing the project structure, the repository baseline was checked.

```text
main == origin/main
working tree clean
python -m pytest -> 9/9 passed
```

This confirmed that the Day 3 harness was green before the canonicalization work began.

## 2. Legacy Model Comparison

The repository contained two different model implementations:

```text
seraph/models/model.py
-> VideoMAEMultiHead

legacy/final_package/model.py
-> GarbageDumpingVideoMAE
-> FocalLoss
-> build_model
```

`seraph/models/model.py` represents an earlier training or experiment implementation. `legacy/final_package/model.py` represents the final packaged implementation that matches the trained checkpoint contract.

The important conclusion was that `legacy/final_package/model.py` should define the recovered canonical behavior, but the legacy folder itself should remain a historical reference rather than the place where future development happens.

## 3. Canonical Package Structure

A new canonical development package was introduced under `src/`:

```text
src/
`-- khuda_cv/
    |-- __init__.py
    `-- model.py
```

The intended ownership boundary is now:

```text
src/khuda_cv/
-> active canonical implementation

legacy/final_package/
-> final packaged historical reference

seraph/
-> earlier training and experiment code
```

## 4. Behavior-Preserving Model Copy

The final packaged model was copied into the new canonical path:

```text
legacy/final_package/model.py
src/khuda_cv/model.py
```

The copy was verified by SHA256 hash:

```text
A853D323D5F0777CE39A0D6DC34E86825F47AD7880F4EF06526F2D13F958A69E
```

Both files produced the same hash, confirming that this step changed the development location without changing model behavior.

## 5. Harness Import Path Migration

The verification harness was then updated to import the canonical model from:

```text
src/khuda_cv/model.py
```

The affected contract tests were:

```text
tests/test_imports.py
tests/test_model_contract.py
tests/test_checkpoint_contract.py
```

Each test now points at `src/khuda_cv/model.py` instead of `legacy/final_package/model.py`.

## 6. Contract Reverification

After the import path migration, the focused harness checks were rerun:

```text
tests/test_imports.py -> passed
tests/test_model_contract.py -> passed
tests/test_checkpoint_contract.py -> passed
```

This confirmed that the new canonical path still satisfies the import contract, model output contract, and checkpoint compatibility contract.

## 7. Legacy Boundary Documentation

Two documentation files were added to make the repository boundaries explicit:

```text
legacy/final_package/README.md
seraph/README_LEGACY.md
```

These files explain that `legacy/final_package/` is preserved as the final packaged historical reference, while `seraph/` contains earlier training and experimental code. Future behavior changes should happen under `src/khuda_cv/` first.

## 8. Full Harness Verification

The full verification harness was rerun after the refactor:

```text
python -m pytest
```

Result:

```text
9 passed
```

This means Day 4 completed without a behavior regression.

## 9. Final Change Set

Day 4 introduced the following intended changes:

```text
src/khuda_cv/__init__.py
src/khuda_cv/model.py
tests/test_imports.py
tests/test_model_contract.py
tests/test_checkpoint_contract.py
legacy/final_package/README.md
seraph/README_LEGACY.md
day4.md
```

## 10. Meaning of Day 4

Day 4 established the canonical development path for the project.

The key result is not a new model architecture or a behavior change. The key result is that the project now has a clear active implementation path, a protected checkpoint contract, and documented legacy boundaries.

In short:

```text
Before Day 4:
legacy code contained the behavior, but the future development path was ambiguous.

After Day 4:
src/khuda_cv/model.py is the canonical development path, verified by the harness and compatible with the trained checkpoint.
```

This is the first safe refactor: behavior-preserving, contract-verified, and ready for future cleanup work.
