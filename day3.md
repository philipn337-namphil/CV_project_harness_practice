# Day 3 - Contract Test Harness

Day 1에서 기존 프로젝트의 baseline을 파악하고, Day 2에서 canonical model과 project contract를 정의했다.

Day 3의 목표는 문서에만 존재하던 contract를 자동화된 테스트로 바꾸는 것이었다.

문서에 contract를 적어두는 것만으로는 충분하지 않다. Agent가 모델 구조를 수정하거나 데이터 전처리 코드를 바꿀 때, 입력 shape이나 checkpoint 호환성이 깨져도 문서만으로는 자동 감지가 어렵다.

그래서 Day 3에서는 Day 2에서 확정한 정상 동작을 pytest 기반 Verification Harness로 고정했다.

---

## 1. Day 2 Baseline 확인

작업 전 repository가 Day 2 결과와 동기화되어 있는지 확인했다.

- `main == origin/main`
- working tree clean
- `AGENTS.md` 존재
- `docs/` context 문서 존재
- `legacy/final_package/` 존재

Harness는 "현재 정상이라고 판단한 상태"를 보호하는 장치이기 때문에, 테스트를 만들기 전에 기준점 자체가 올바른지 확인하는 과정이 필요했다.

---

## 2. Test Harness 구조 설계

Day 2에서 정의한 contract 중 자동 검증이 필요한 핵심 영역을 테스트 대상으로 정했다.

```text
tests/
├── test_imports.py
├── test_model_contract.py
├── test_checkpoint_contract.py
├── test_dataset_contract.py
└── test_preprocessing_contract.py
```

각 테스트의 역할은 다음과 같이 분리했다.

- `test_imports.py`: canonical model import 가능 여부 확인
- `test_model_contract.py`: model input/output shape 확인
- `test_checkpoint_contract.py`: 기존 checkpoint와 canonical model의 호환성 확인
- `test_dataset_contract.py`: Dataset sample과 Batch schema 확인
- `test_preprocessing_contract.py`: frame 수, resize, tensor type, normalization 확인

테스트는 실제 대용량 artifact가 없어도 가능한 fast test와 checkpoint가 필요한 artifact-dependent test로 구분했다.

---

## 3. Import Smoke Test

가장 먼저 canonical model 파일과 주요 symbol이 정상적으로 import되는지 확인했다.

검증 항목은 다음과 같다.

- `legacy/final_package/model.py` 존재
- `GarbageDumpingVideoMAE` import 가능
- `build_model` import 가능

결과:

```text
2 passed
```

이 테스트는 단순하지만 중요하다. 이후 파일 이동, module path 변경, symbol 이름 변경으로 프로젝트의 기본 import가 깨지는 상황을 빠르게 잡아낼 수 있다.

---

## 4. Model Contract Test

다음으로 Day 2에서 정의한 canonical model contract를 테스트했다.

Dummy input을 실제 canonical model에 전달해 입력과 출력 구조가 유지되는지 확인했다.

```text
Input
(1, 16, 3, 224, 224)

Output
clip_logits       -> (1, 2)
frame_logits      -> (1, 16)
last_hidden_state -> exists
```

또한 model output이 dictionary 형태인지 확인했다.

결과:

```text
1 passed
```

이 테스트는 모델 리팩터링 중 가장 쉽게 깨질 수 있는 interface를 보호한다. 특히 frame 수, channel 순서, clip-level output, frame-level output의 shape이 바뀌는 문제를 바로 감지할 수 있다.

---

## 5. Checkpoint Contract Test

실제 `best_model.pt`를 사용해 checkpoint와 canonical model의 호환성을 테스트했다.

검증 항목은 다음과 같다.

- `model_state_dict` 존재
- `clip_head.*` key 존재
- `frame_head.*` key 존재
- `backbone.*` key 182개 확인
- `GarbageDumpingVideoMAE`에 `strict=True` load 성공

결과:

```text
1 passed
```

이 테스트는 이후 Agent가 architecture나 layer name을 변경해 기존 checkpoint를 깨뜨리는 상황을 감지하기 위한 안전망이다.

모델 코드는 겉으로 보기에는 정상적으로 실행될 수 있지만, 기존 학습 결과물을 더 이상 불러올 수 없다면 프로젝트 관점에서는 큰 회귀다. Checkpoint Contract Test는 이 문제를 자동으로 확인한다.

---

## 6. Dataset Contract Test

Day 2에서 정의한 canonical Dataset/Batch interface를 synthetic sample로 테스트했다.

Sample contract:

```text
pixel_values -> (16, 3, 224, 224)
clip_label   -> scalar
frame_labels -> (16,)
```

Batch contract:

```text
pixel_values -> (B, 16, 3, 224, 224)
clip_label   -> (B,)
frame_labels -> (B, 16)
```

Dataset과 Batch 모두 dictionary interface를 표준으로 한다.

결과:

```text
2 passed
```

이 테스트는 데이터 로딩 코드가 모델이 기대하는 입력 형식을 계속 만족하는지 확인한다. 모델과 데이터셋 사이의 연결부를 contract로 고정한 것이다.

---

## 7. Preprocessing Contract Test

영상 전처리 계약도 테스트로 고정했다.

검증 항목은 다음과 같다.

- 16-frame 유지
- 224 x 224 resize
- `float32` tensor
- ImageNet mean/std normalization
- black pixel normalization 계산
- white pixel normalization 계산

결과:

```text
3 passed
```

Preprocessing은 모델 성능에 직접 영향을 주는 영역이다. 입력 크기나 normalization 값이 바뀌면 코드는 실행되더라도 모델의 의미 있는 추론 결과가 깨질 수 있다.

그래서 resize와 dtype뿐 아니라 black/white pixel 기준 normalization 계산까지 테스트에 포함했다.

---

## 8. pytest Runner 구축

Repository root에 `pytest.ini`를 추가해 전체 Harness를 한 명령으로 실행할 수 있도록 했다.

```bash
python -m pytest
```

최종 결과:

```text
9 passed
```

이제 Agent나 개발자는 개별 테스트 파일을 기억할 필요 없이 repository root에서 pytest만 실행하면 전체 contract를 검증할 수 있다.

---

## 9. AGENTS.md Verification Harness 연결

`AGENTS.md`에 Verification Harness 규칙을 추가했다.

앞으로 non-trivial code change 이후 Agent는 반드시 다음 명령을 실행해야 한다.

```bash
python -m pytest
```

또한 테스트가 실패했을 때의 원칙도 명확히 했다.

테스트를 삭제하거나 약화해서 통과시키는 것이 아니라, 먼저 implementation과 contract 중 무엇이 잘못되었는지 판단해야 한다.

즉 Harness는 단순한 테스트 모음이 아니라 Agent가 프로젝트를 수정할 때 따라야 하는 검증 규칙으로 연결된다.

---

## 10. Day 3 Result

Day 3 종료 시점에서 다음 contract가 자동 테스트로 보호된다.

```text
Canonical model import
Model input/output shape
Checkpoint compatibility
Dataset schema
Batch schema
Preprocessing behavior
ImageNet normalization
```

전체 Harness 상태:

```text
9 passed
```

Day 1과 Day 2에서 사람이 문서로 정의했던 정상 동작을 Day 3에서는 코드가 자동으로 검증하도록 만들었다.

```text
Day 1 - Observe
Day 2 - Define
Day 3 - Enforce
```

이제 다음 단계부터는 이 Harness를 안전망으로 사용해 legacy code를 실제로 리팩터링할 수 있다.
