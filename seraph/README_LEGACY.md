# Seraph Legacy Code

The `seraph/` directory contains earlier training and experimental implementations from the original CV project.

This code is preserved for historical context and should not be treated as the canonical implementation.

## Important Model Difference

Legacy model:

```text
seraph/models/model.py
-> VideoMAEMultiHead
```

Canonical model:

```text
src/khuda_cv/model.py
-> GarbageDumpingVideoMAE
```

The canonical implementation is the version compatible with the verified `best_model.pt` checkpoint.

Do not delete this directory without first confirming that no historical training or evaluation behavior is still needed.
