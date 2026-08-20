# Day 6 — Canonical Training Pipeline

## 목표

Day 5까지 Model과 Data Pipeline을 `src/khuda_cv/` 아래의 canonical 영역으로 정리했다.

Day 6의 목표는 기존 `seraph/train.py`를 그대로 복사하는 것이 아니라, 현재 canonical Model/Data Contract에 맞는 Training Pipeline의 최소 단위를 새롭게 정의하는 것이다.

특히 이번 단계에서는 전체 epoch loop를 한 번에 이전하지 않고 다음 흐름을 먼저 안정화했다.

```text
Canonical Batch
      ↓
Model Forward
      ↓
Loss 계산
      ↓
Backward
      ↓
Optimizer Step
```

---

## 1. Day 5 Baseline 확인

작업을 시작하기 전에 repository 상태와 기존 Verification Harness를 다시 확인했다.

확인 항목:

- 최신 `main` branch
- clean working tree
- Day 5까지의 변경사항 반영 여부
- 기존 전체 Harness 실행

결과:

```text
9 passed
```

따라서 기존 동작이 정상적으로 보호되고 있는 상태에서 Day 6 작업을 시작했다.

---

## 2. 기존 Training Pipeline 분석

기존 학습 코드는 `seraph/train.py`에 위치하고 있었다.

주요 함수는 다음과 같았다.

```text
train_one_epoch()
validate()
train()
```

기존 Training Pipeline의 흐름은 다음과 같았다.

```text
MyDataset
   ↓
tuple batch
(frames, clip_label, frame_labels)
   ↓
VideoMAEMultiHead
   ↓
(clip_logit, frame_logits)
   ↓
BCE Loss + BCE Loss
   ↓
backward()
   ↓
optimizer.step()
```

하지만 Day 4~5에서 확정한 canonical 구조와 비교하면 여러 차이가 있었다.

| Legacy Training | Canonical Contract |
|---|---|
| `VideoMAEMultiHead` | `GarbageDumpingVideoMAE` |
| tuple batch | dict batch |
| tuple model output | dict model output |
| 기존 Dataset 경로 | `src.khuda_cv.data` |
| clip BCE | canonical Focal Loss |
| 48-frame 중심 코드 | 16-frame model input contract |

따라서 기존 `train.py` 전체를 그대로 복사하는 방식은 사용하지 않기로 했다.

---

## 3. Canonical Training Contract 정의

Training Pipeline을 이전하기 전에 가장 작은 학습 단위의 Contract를 먼저 정의했다.

Canonical batch는 다음 구조를 사용한다.

```text
{
    pixel_values: (B, 16, 3, 224, 224),
    clip_label:   (B,),
    frame_labels: (B, 16),
    clip_id:      list[str],
    video_path:   list[str]
}
```

Model output은 다음 구조를 따른다.

```text
{
    clip_logits:  (B, 2),
    frame_logits: (B, 16),
    ...
}
```

Loss는 두 종류로 구성했다.

```text
clip_logits
    ↓
FocalLoss

frame_logits
    ↓
BCEWithLogitsLoss
```

최종 loss는 기존 정책과 동일하게 두 loss의 합으로 정의했다.

```text
total_loss = clip_loss + frame_loss
```

이를 통해 Dataset → Model → Loss 사이의 Training Contract를 명확하게 고정했다.

---

## 4. Canonical Training 구조 생성

Training 관련 코드를 기존 `seraph/` 영역과 분리하기 위해 새로운 package를 생성했다.

```text
src/
└── khuda_cv/
    └── training/
        ├── __init__.py
        ├── losses.py
        └── step.py
```

각 파일의 역할은 다음과 같다.

```text
losses.py
→ loss 생성 및 계산

step.py
→ single optimization step

__init__.py
→ canonical training interface export
```

전체 training script를 하나의 파일에 넣는 대신 기능 단위로 분리하기 시작했다.

---

## 5. Loss 로직 분리

`losses.py`에 Training Loss 관련 로직을 분리했다.

구현한 주요 interface는 다음과 같다.

```text
LossOutput
build_losses()
compute_losses()
```

`build_losses()`는 canonical loss function을 생성한다.

```text
Clip classification
→ FocalLoss

Frame detection
→ BCEWithLogitsLoss
```

`compute_losses()`는 canonical model output과 batch를 받아 다음 결과를 반환한다.

```text
total_loss
clip_loss
frame_loss
```

Synthetic tensor를 사용해 실제 계산도 확인했다.

```text
training losses import ok
```

또한 세 loss 값이 정상적으로 계산되는 것을 확인했다.

이를 통해 Loss 계산 로직이 전체 Training Loop와 독립적으로 동작하게 되었다.

---

## 6. Single Train Step 구현

다음으로 `step.py`에 하나의 batch를 학습하는 최소 단위를 구현했다.

Canonical train step은 다음 순서로 동작한다.

```text
batch
  ↓
move to device
  ↓
optimizer.zero_grad()
  ↓
model forward
  ↓
compute_losses()
  ↓
total_loss.backward()
  ↓
optimizer.step()
```

이를 `train_step()`이라는 독립적인 함수로 분리했다.

또한 `move_batch_to_device()`를 통해 tensor field만 target device로 이동하도록 구성했다.

이 단계에서는 의도적으로 다음 기능을 포함하지 않았다.

```text
Epoch Loop
Validation
Scheduler
AMP / GradScaler
Progressive Unfreezing
Checkpoint Save
```

Training Pipeline 전체를 한 번에 이전하기보다 가장 작은 optimization unit부터 검증하기 위한 선택이다.

---

## 7. Training Harness 추가

새로운 Training Contract를 보호하기 위해 다음 테스트 파일을 추가했다.

```text
tests/test_training_contract.py
```

두 가지 테스트를 구현했다.

### Training Loss Contract

다음을 검증한다.

```text
clip loss 계산
frame loss 계산
total loss 계산
finite loss 여부
total_loss = clip_loss + frame_loss
```

### Single Train Step Contract

실제 VideoMAE 대신 canonical output contract를 따르는 작은 Toy Model을 사용했다.

테스트 흐름은 다음과 같다.

```text
Synthetic Batch
      ↓
Toy Model
      ↓
Forward
      ↓
Loss
      ↓
Backward
      ↓
Optimizer Step
      ↓
Parameter 변경 확인
```

단순히 함수가 오류 없이 실행되는지만 확인하는 것이 아니라, optimizer 실행 전후의 parameter를 비교해 실제 gradient update가 발생했는지도 검증했다.

Focused test 결과:

```text
2 passed
```

---

## 8. Legacy → Canonical Migration 경계 정리

Training Pipeline 전체를 바로 이전하지 않고 기존 코드에서 어떤 부분을 유지하고 어떤 부분을 교체할지 정리했다.

| Legacy 요소 | 처리 |
|---|---|
| `MyDataset` | canonical Data Pipeline으로 교체 |
| tuple batch | canonical dict batch로 교체 |
| `VideoMAEMultiHead` | canonical Model로 교체 |
| clip BCE | Focal Loss로 교체 |
| frame BCE | 유지 |
| `clip_loss + frame_loss` | 유지 |
| backward / optimizer step | canonical `train_step()`으로 이전 |
| Progressive Unfreezing | 추후 이전 |
| AMP / GradScaler | 추후 이전 |
| Scheduler | 추후 이전 |
| Epoch Loop | 추후 이전 |
| Validation | 추후 이전 |
| Checkpoint Save | 추후 이전 |
| Hard-coded server paths | canonical 영역에서 제거 |

따라서 현재 Training Pipeline의 migration boundary는 다음과 같다.

```text
Day 6 완료 영역

Loss
 ↓
Single Train Step
 ↓
Backward
 ↓
Optimizer Update
```

이후 단계에서는 이 위에 Epoch / Validation / Scheduler / Checkpoint 등의 기능을 순차적으로 쌓을 수 있다.

---

## 9. Full Harness 검증

Training Harness를 추가한 뒤 전체 regression test를 다시 실행했다.

기존 Harness:

```text
9 tests
```

Day 6 Training Harness:

```text
+ 2 tests
```

최종 결과:

```text
11 passed
```

기존 Model, Dataset, Preprocessing, Checkpoint Contract가 깨지지 않은 상태에서 새로운 Training Contract까지 정상적으로 추가된 것을 확인했다.

---

## 10. Day 6 결과

Day 6에서는 기존 `seraph/train.py` 전체를 새로운 위치로 복사하지 않았다.

대신 Training Pipeline을 구성하는 가장 작은 단위부터 canonical 영역으로 분리했다.

최종 구조는 다음과 같다.

```text
src/khuda_cv/
├── model.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── collate.py
│
└── training/
    ├── __init__.py
    ├── losses.py
    └── step.py
```

현재까지 canonical pipeline은 다음 단계까지 연결되었다.

```text
Canonical Dataset
        ↓
Canonical Batch
        ↓
Canonical Model
        ↓
Canonical Loss
        ↓
Canonical Train Step
        ↓
Backward
        ↓
Optimizer Update
```

Verification Harness 역시 다음과 같이 확장되었다.

```text
Day 3 : Contract Verification Harness
Day 4 : Canonical Model
Day 5 : Canonical Data Pipeline
Day 6 : Canonical Training Step
```

최종 검증 결과:

```text
11 passed
```

Day 6를 통해 전체 Training Loop를 이전하기 전에 핵심 optimization behavior를 독립적인 코드와 테스트로 보호할 수 있는 기반을 만들었다.
