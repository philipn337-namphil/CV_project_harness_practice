# Day 7 — Canonical Epoch & Validation Pipeline

Day 6에서는 Canonical Training Pipeline의 최소 단위인 Loss 계산과 Single Train Step까지 구현했다.

Day 7의 목표는 그 위에 여러 batch를 처리하는 **Epoch Training과 Validation Pipeline**을 올리는 것이었다.

이번에도 전체 Training Loop를 한 번에 옮기지 않고, 다음 레이어만 추가했다.

```text
Single Train Step
        ↓
Train Epoch
        ↓
Validation Epoch
        ↓
Epoch Metrics
```

---

## 1. Day 6 Baseline 확인

작업을 시작하기 전에 repository와 Verification Harness 상태를 다시 확인했다.

확인 결과:

```text
main == origin/main
working tree clean
Day 6 commit 존재
11/11 tests passed
```

Day 6에서 만든 canonical training step까지 정상적으로 보호되고 있는 상태에서 Day 7 작업을 시작했다.

---

## 2. 기존 Epoch / Validation 동작 분석

기존 `seraph/train.py`에는 다음 두 함수가 존재했다.

```text
train_one_epoch()
validate()
```

Legacy Training Epoch는 다음 흐름으로 동작했다.

```text
model.train()
    ↓
for batch in train_loader
    ↓
forward
    ↓
clip loss + frame loss
    ↓
backward
    ↓
optimizer step
    ↓
prediction 수집
    ↓
epoch metric 계산
```

Validation은 다음 구조였다.

```text
model.eval()
    ↓
torch.no_grad()
    ↓
for batch in val_loader
    ↓
forward
    ↓
loss 계산
    ↓
prediction 수집
    ↓
epoch metric 계산
```

기존 metric은 다음 네 가지를 반환했다.

```text
epoch loss
clip accuracy
clip macro F1
frame F1
```

하지만 Legacy Model과 Canonical Model의 output contract가 달랐기 때문에 prediction 계산은 그대로 사용할 수 없었다.

Legacy에서는 clip output이 single logit이었기 때문에:

```text
clip_logit > 0
```

으로 classification했다.

Canonical Model은:

```text
clip_logits → (B, 2)
```

형태이므로:

```text
argmax(dim=1)
```

을 사용하도록 변경했다.

---

## 3. Canonical Epoch Contract 정의

Canonical Training Epoch는 Day 6에서 만든 `train_step()`을 반복해서 호출하는 구조로 정의했다.

```text
model
loader
optimizer
clip criterion
frame criterion
device
        ↓
for each batch
        ↓
train_step()
        ↓
prediction 수집
        ↓
epoch metric 계산
        ↓
EpochMetrics
```

Validation Epoch는 optimizer를 사용하지 않고 다음 흐름을 따른다.

```text
model
loader
clip criterion
frame criterion
device
        ↓
model.eval()
        ↓
gradient disabled
        ↓
forward
        ↓
loss 계산
        ↓
prediction 수집
        ↓
EpochMetrics
```

Epoch의 반환값 역시 기존 tuple 대신 이름이 있는 구조로 정리했다.

```text
EpochMetrics
├── loss
├── clip_accuracy
├── clip_f1
└── frame_f1
```

---

## 4. Metric 로직 분리

Training과 Validation 양쪽에서 같은 metric 계산 코드를 반복하지 않도록 별도 module로 분리했다.

새로운 파일:

```text
src/khuda_cv/training/metrics.py
```

여기에 다음 interface를 추가했다.

```text
EpochMetrics
compute_epoch_metrics()
```

Clip metric은 다음과 같이 계산한다.

```text
clip accuracy
clip macro F1
```

Frame prediction은:

```text
frame_logits > 0
```

으로 binary prediction을 만들었다.

그리고 Legacy 코드와 달리 frame prediction과 label을 flatten한 뒤 binary F1을 계산하도록 정리했다.

```text
(B, 16)
   ↓
flatten
   ↓
전체 sampled frame 기준 binary F1
```

간단한 smoke test 결과:

```text
EpochMetrics(
    loss=0.5,
    clip_accuracy=0.6666...,
    clip_f1=0.6666...,
    frame_f1=0.6666...
)
```

로 정상 계산되는 것을 확인했다.

---

## 5~6. Canonical Train Epoch와 Validation 구현

Train Epoch와 Validation은 서로 강하게 연결된 기능이기 때문에 같은 `epoch.py` 안에서 함께 구현했다.

새로운 파일:

```text
src/khuda_cv/training/epoch.py
```

### Train Epoch

`train_epoch()`은 여러 canonical batch를 순회하면서 Day 6에서 만든 `train_step()`을 반복 호출한다.

```text
loader
  ↓
batch
  ↓
train_step()
  ↓
loss 누적
  ↓
prediction 수집
  ↓
EpochMetrics
```

Clip prediction은:

```text
clip_logits.argmax(dim=1)
```

Frame prediction은:

```text
frame_logits > 0
```

을 사용한다.

### Validation Epoch

`validate_epoch()`은 같은 metric contract를 유지하지만 다음 차이를 가진다.

```text
model.eval()
gradient disabled
optimizer 없음
parameter update 없음
```

즉 Training과 Validation의 output format은 동일하게 유지하면서 optimization 여부만 명확하게 분리했다.

---

## 7. Epoch / Validation Harness 추가

새로운 Epoch Contract를 보호하기 위해 다음 테스트를 추가했다.

```text
tests/test_epoch_contract.py
```

실제 VideoMAE 대신 canonical output contract를 따르는 작은 Toy Model과 synthetic DataLoader를 사용했다.

### Train Epoch Test

다음을 확인한다.

```text
여러 batch 처리
model.train() 상태
finite loss
clip accuracy 범위
clip F1 범위
frame F1 범위
parameter 실제 변경
```

즉 단순히 epoch 함수가 끝나는지만 확인하는 것이 아니라 여러 batch를 거친 뒤 실제 optimizer update가 발생했는지까지 검사했다.

### Validation Epoch Test

다음을 확인한다.

```text
여러 batch 처리
model.eval() 상태
finite loss
metric 정상 계산
parameter 변화 없음
```

Validation 전후 model parameter를 비교해 validation 과정에서 학습이 발생하지 않는 것도 확인했다.

Focused Test 결과:

```text
2 passed
```

---

## 8. Legacy와 Canonical 동작 비교

Day 7까지 이전한 Epoch / Validation 동작을 기존 `seraph/train.py`와 비교했다.

| Legacy Training | Canonical Day 7 |
|---|---|
| tuple batch | dict batch |
| `VideoMAEMultiHead` | `GarbageDumpingVideoMAE` contract |
| single-logit clip prediction | 2-class argmax |
| frame metric shape 모호 | flatten 후 binary F1 |
| tuple metric return | `EpochMetrics` |
| train/validation 내부 metric 중복 | 공통 metric module |
| `model.train()` | 유지 |
| `model.eval()` | 유지 |
| validation `no_grad()` | 유지 |
| AMP | 추후 이전 |
| Scheduler | 추후 이전 |
| Progressive Unfreezing | 추후 이전 |
| Checkpoint Save | 추후 이전 |

즉 Legacy behavior를 그대로 복제한 것이 아니라, 현재 Canonical Model/Data Contract에 맞게 의미를 유지하면서 interface를 정리했다.

---

## 9. Full Harness 검증

Epoch / Validation Harness를 추가한 뒤 전체 regression test를 다시 실행했다.

기존 Harness:

```text
11 tests
```

Day 7에서 추가된 Harness:

```text
+ 2 tests
```

최종 결과:

```text
13 passed
```

현재 Harness는 다음 영역까지 보호한다.

```text
Model Contract
Checkpoint Contract
Dataset Contract
Batch Contract
Preprocessing Contract
Training Loss Contract
Single Train Step Contract
Train Epoch Contract
Validation Epoch Contract
```

Day 7 변경 이후에도 기존 기능에 regression이 없음을 확인했다.

---

## 10. Day 7 결과

Day 7에서는 Day 6에서 만든 Single Train Step 위에 Epoch와 Validation이라는 다음 실행 레이어를 추가했다.

현재 canonical training 구조는 다음과 같다.

```text
src/khuda_cv/training/
├── __init__.py
├── losses.py
├── step.py
├── metrics.py
└── epoch.py
```

전체 pipeline은 이제 다음 단계까지 이어진다.

```text
Canonical Dataset
        ↓
Canonical Batch
        ↓
Canonical Model
        ↓
Canonical Loss
        ↓
Single Train Step
        ↓
Train Epoch
        ↓
Validation Epoch
        ↓
Epoch Metrics
```

아직 다음 기능은 Legacy 영역에 남아 있다.

```text
AMP / GradScaler
Scheduler
Progressive Unfreezing
Checkpoint Save
Full Multi-Epoch Runner
```

하지만 Training과 Validation의 핵심 실행 경로는 이제 canonical 영역에서 독립적으로 동작하며 Harness로 보호된다.

최종 Verification Harness 상태:

```text
13 passed
```

현재까지의 흐름은 다음과 같다.

```text
Day 1 — Observe
Day 2 — Define
Day 3 — Enforce
Day 4 — Canonicalize Model
Day 5 — Canonicalize Data Pipeline
Day 6 — Canonicalize Training Step
Day 7 — Canonicalize Epoch & Validation
```

Day 7을 통해 이제 하나의 batch가 아니라 여러 batch로 구성된 전체 epoch까지 canonical pipeline 안에서 실행할 수 있게 되었다.

---

## Day 7 최종 체크리스트

| 단계 | 할 일 | 완료 기준 | 상태 |
|---|---|---|---|
| **1. Day 6 baseline 확인** | repo 이동 → pull/status → Harness 확인 | clean + 최신 main + 11/11 PASS | ✅ |
| **2. 기존 Training 경로 분석** | `seraph/train.py` 분석 | 유지할 behavior 확정 | ✅ |
| **3. Epoch Contract 정의** | train/validation 입력·출력 정의 | canonical interface 확정 | ✅ |
| **4. Metric 로직 분리** | accuracy/F1 계산 공통화 | 독립 metric 계산 가능 | ✅ |
| **5. Canonical Train Epoch 구현** | 여러 batch에 `train_step()` 적용 | epoch 함수 구현/import | ✅ |
| **6. Canonical Validation 구현** | no-grad validation 경로 구현 | validation 함수 구현/import | ✅ |
| **7. Epoch/Validation Harness 추가** | Toy Model + synthetic loader 테스트 | 2/2 PASS | ✅ |
| **8. Legacy와 동작 비교** | legacy/canonical 차이 정리 | migration 차이 명문화 | ✅ |
| **9. Full Harness 검증** | 전체 regression 검사 | **13/13 PASS** | ✅ |
| **10. Day 7 기록 및 Git 반영** | `day7.md` 작성 | commit → push → clean | ✅ |
