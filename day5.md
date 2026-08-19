# Day 5 — Canonical Data Pipeline

Day 4에서는 여러 모델 구현 중 실제 최종 checkpoint와 호환되는 모델을 `src/khuda_cv/model.py`로 이동해 canonical model path를 확립했다.

Day 5의 목표는 같은 원칙을 데이터 영역에도 적용하는 것이었다.

기존 `seraph/` 내부에 흩어져 있던 Dataset과 DataLoader 구현을 분석하고, Day 2에서 정의하고 Day 3에서 테스트로 보호한 data contract를 기준으로 새로운 canonical data pipeline을 구축했다.

최종 목표는 다음과 같다.

```text
Dataset
   ↓
dict sample
   ↓
canonical collate
   ↓
dict batch
   ↓
Model
```

---

## 1. Day 4 Baseline 확인

작업을 시작하기 전에 repository와 Verification Harness 상태를 다시 확인했다.

```text
main == origin/main
working tree clean
9/9 tests passed
```

Day 4에서 canonical model path를 만든 이후에도 전체 Harness가 정상적으로 유지되고 있었다.

따라서 Day 5의 변경으로 문제가 발생할 경우 이번 작업에서 발생한 regression이라고 판단할 수 있는 baseline을 확보했다.

---

## 2. Legacy Dataset 구현 비교

먼저 기존 repository에 존재하는 데이터 관련 구현을 조사했다.

주요 파일은 다음 세 개였다.

```text
seraph/dataset.py
seraph/other/dataset.py
seraph/utils/transforms.py
```

`seraph/dataset.py`와 `seraph/other/dataset.py`를 확인한 결과 두 파일 모두 크기가 8276 bytes였으며 다음 구조를 가지고 있었다.

```text
GarbageDumpingClipDataset
collate_fn
```

SHA256도 비교했다.

```text
68CDBAFA3AD6B5CA6FE820943D62B4228C6E508EDEB9C9562B9BED43782A0C93
```

두 파일의 hash가 완전히 동일했다.

즉 서로 다른 구현이 아니라 동일한 Dataset 코드가 두 위치에 중복되어 있던 것이다.

반면 `seraph/utils/transforms.py`에는 다음과 같은 별도의 데이터 로딩 계층이 존재했다.

```text
_FolderClipDataset
MyDataset
_tuple_collate_fn
```

특히 manifest를 사용할 경우에는 `GarbageDumpingClipDataset`을 생성하지만 DataLoader에서는 `_tuple_collate_fn`을 사용하고 있었다.

이를 통해 다음과 같은 잠재적인 interface mismatch도 발견했다.

```text
Manifest Dataset
→ dict sample

_tuple_collate_fn
→ tuple sample 기대
```

따라서 canonical pipeline에서는 이 두 방식을 섞지 않고 하나의 명확한 contract로 통일할 필요가 있었다.

---

## 3. Canonical Dataset Contract 재확인

Day 2와 Day 3에서 정의한 Dataset Contract를 다시 기준으로 삼았다.

하나의 sample은 다음 구조를 사용한다.

```text
{
    pixel_values,
    clip_label,
    frame_labels,
    clip_id,
    video_path
}
```

각 값의 contract는 다음과 같다.

```text
pixel_values → (16, 3, 224, 224)
clip_label   → scalar
frame_labels → (16,)
clip_id      → str
video_path   → str
```

DataLoader를 통과한 batch 역시 dictionary 형태로 유지한다.

```text
pixel_values → (B, 16, 3, 224, 224)
clip_label   → (B,)
frame_labels → (B, 16)
clip_id      → list[str]
video_path   → list[str]
```

따라서 Day 5에서는 **manifest 기반 dictionary interface를 canonical contract로 확정**했다.

기존 `seraph/`의 tuple 기반 또는 중복 Dataset 구현은 삭제하지 않고 legacy 영역으로 유지하기로 했다.

---

## 4. Canonical Data Package 생성

Day 4에서 model에 적용했던 것과 동일하게 데이터 코드에도 명확한 canonical development path를 만들었다.

새로운 구조는 다음과 같다.

```text
src/
└── khuda_cv/
    ├── model.py
    └── data/
        ├── __init__.py
        ├── dataset.py
        └── collate.py
```

각 파일의 책임도 분리했다.

```text
dataset.py
→ Dataset 구현

collate.py
→ Sample → Batch 변환

__init__.py
→ canonical data API export
```

앞으로 active development에서는 `seraph/`가 아니라 이 경로를 기준으로 사용한다.

---

## 5. Dataset 구현 이전

Canonical Dataset의 기반으로 기존 `seraph/dataset.py`의 `GarbageDumpingClipDataset` 구현을 사용했다.

```text
seraph/dataset.py
        ↓
src/khuda_cv/data/dataset.py
```

이 과정에서 기존 서버 경로를 이용하는 standalone 실행 코드는 canonical package에서 제거했다.

Dataset을 새 경로에서 import하는 과정에서는 환경 관련 문제도 발견했다.

처음에는 다음 dependency가 설치되어 있지 않았다.

```text
ModuleNotFoundError: No module named 'decord'
```

따라서 `decord 0.6.0`을 설치했다.

이후 Windows 환경에서 `decord`가 먼저 import된 뒤 `torch`가 로드되는 과정에서 일시적으로 다음 오류가 발생했다.

```text
WinError 1114
Error loading c10.dll
```

하지만 PyTorch 자체를 확인한 결과:

```text
torch 2.11.0+cu128
```

은 정상적으로 import되고 있었다.

Canonical Dataset의 import 순서를 `torch`가 먼저 로드되도록 정리한 뒤 다시 확인했다.

```text
canonical dataset import ok
```

결과적으로 canonical Dataset을 독립적으로 import할 수 있는 상태를 만들었다.

---

## 6. DataLoader / Collate Contract 통일

다음으로 Dataset sample을 batch로 변환하는 canonical `collate_fn`을 별도의 파일로 분리했다.

```text
src/khuda_cv/data/collate.py
```

Canonical collate는 tensor 데이터는 stack하고 metadata는 list 형태로 유지한다.

```text
pixel_values
→ torch.stack

clip_label
→ torch.stack

frame_labels
→ torch.stack

clip_id
→ list

video_path
→ list
```

Synthetic sample 두 개를 사용해 직접 확인한 결과:

```text
torch.Size([2, 16, 3, 224, 224])
torch.Size([2])
torch.Size([2, 16])
['test', 'test']
['test.mp4', 'test.mp4']
```

로 기대한 Batch Contract가 정확히 생성되었다.

또한 기존 legacy collate에서는 batch 생성 과정에서 `video_path` metadata가 사라졌지만, canonical collate에서는 이를 유지하도록 했다.

최종 데이터 흐름은 다음처럼 단순해졌다.

```text
GarbageDumpingClipDataset
        ↓
dict sample
        ↓
collate_fn
        ↓
dict batch
```

---

## 7. Dataset Harness를 실제 구현에 연결

Day 3에서 만든 `test_dataset_contract.py`는 이전까지 synthetic helper를 이용해 Batch Contract 자체를 검증하고 있었다.

Day 5에서는 이를 실제 canonical implementation과 연결했다.

기존 구조:

```text
Test
→ synthetic canonical_collate
```

변경 후:

```text
Test
→ src.khuda_cv.data.collate_fn
```

즉 테스트가 더 이상 contract를 흉내 내는 helper만 검증하는 것이 아니라 실제 production candidate 코드를 검증하게 되었다.

Dataset Contract Test 결과:

```text
2 passed
```

Canonical data implementation이 기존 Harness의 sample/batch contract를 만족하는 것을 확인했다.

---

## 8. Focused Contract 검증

Dataset 관련 변경이 preprocessing contract까지 영향을 주지 않았는지 확인하기 위해 관련 테스트를 묶어 실행했다.

```text
Dataset Contract       2 tests
Preprocessing Contract 3 tests
```

결과:

```text
5 passed
```

즉 canonical data pipeline을 추가한 이후에도 기존 preprocessing behavior가 유지되고 있었다.

---

## 9. Full Harness 검증

Focused Test가 통과한 뒤 전체 Verification Harness를 실행했다.

```bash
python -m pytest
```

결과:

```text
9 passed in 18.88s
```

검증 대상은 다음과 같다.

```text
Checkpoint Contract
Dataset Sample Contract
DataLoader Batch Contract
Canonical Model Import
Model Output Contract
Preprocessing Shape
ImageNet Normalization
White Pixel Normalization
```

Canonical data path를 추가하고 Dataset Harness를 실제 구현으로 전환한 이후에도 기존 contract가 모두 유지되었다.

---

## 10. Day 5 Result

Day 5에서는 기존 `seraph/` 영역에 중복되어 있던 데이터 구현을 분석하고 새로운 canonical data path를 만들었다.

변경 전에는 다음과 같이 여러 데이터 interface가 섞여 있었다.

```text
seraph/dataset.py
seraph/other/dataset.py
seraph/utils/transforms.py

dict Dataset
tuple Dataset
dict collate
tuple collate
```

변경 후 active development path는 다음처럼 단순해졌다.

```text
src/khuda_cv/data/

GarbageDumpingClipDataset
        ↓
dict sample
        ↓
canonical collate_fn
        ↓
dict batch
```

기존 코드를 바로 삭제하지 않고 `seraph/`에 historical reference로 남겨두면서, 앞으로 개발할 코드의 위치와 interface만 명확하게 분리했다.

그리고 Day 3에서 만든 Harness를 통해 이 변경이 기존 behavior를 깨뜨리지 않았다는 것도 확인했다.

```text
Dataset Contract       PASS
Preprocessing Contract PASS
Full Harness           9/9 PASS
```

현재까지의 Harness Engineering 과정은 다음과 같이 이어진다.

```text
Day 1 — Observe
기존 프로젝트의 구조와 실행 계약을 파악한다.

Day 2 — Define
지켜야 할 Project Contract를 명문화한다.

Day 3 — Enforce
Contract를 자동 테스트로 보호한다.

Day 4 — Canonicalize Model
실제 개발에 사용할 canonical model path를 만든다.

Day 5 — Canonicalize Data Pipeline
Dataset → Collate → Batch 경로를 하나의 contract로 통일한다.
```

Day 4에서 모델의 기준점이 생겼고, Day 5에서는 데이터의 기준점까지 생겼다.

이제 canonical 영역에는 최소한 다음 두 축이 존재한다.

```text
src/khuda_cv/model.py
→ Canonical Model

src/khuda_cv/data/
→ Canonical Data Pipeline
```

다음 단계부터는 이 두 canonical component를 연결해 **Training / Evaluation / Inference 실행 경로를 legacy 코드에서 점진적으로 분리**할 수 있다.