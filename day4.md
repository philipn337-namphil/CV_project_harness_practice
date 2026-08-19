# Day 4 — Canonicalization & First Safe Refactor

Day 1에서는 기존 프로젝트의 구조와 문제점을 파악했고, Day 2에서는 canonical model과 project contract를 정의했다. Day 3에서는 이 contract를 자동 테스트로 보호하는 Verification Harness를 만들었다.

Day 4의 목표는 이 Harness를 실제 안전망으로 사용해, 여러 곳에 흩어진 legacy 구현 중 어떤 코드를 앞으로 개발할 canonical implementation으로 사용할지 명확하게 정리하는 것이었다.

## 1. Day 3 Baseline 확인

작업 전 repository 상태와 전체 Harness를 다시 확인했다.

확인 결과:

```text
main == origin/main
working tree clean
9/9 tests passed
```

즉 Day 3에서 만든 안전망이 정상적으로 유지되고 있는 상태에서 리팩터링을 시작했다.

## 2. Legacy 모델 구현 비교

현재 repository에는 서로 다른 두 모델 구현이 존재했다.

```text
seraph/models/model.py
→ VideoMAEMultiHead

legacy/final_package/model.py
→ GarbageDumpingVideoMAE
```

두 파일의 구조를 비교한 결과, `seraph/models/model.py`는 과거 학습 과정에서 사용된 legacy implementation이고, `legacy/final_package/model.py`는 실제 최종 package와 `best_model.pt`에 대응하는 canonical implementation임을 다시 확인했다.

`legacy/final_package/model.py`에는 다음 구성 요소가 포함되어 있었다.

```text
FocalLoss
GarbageDumpingVideoMAE
build_model
```

반면 `seraph/models/model.py`에는 `VideoMAEMultiHead`만 존재했다.

## 3. Canonical Package 구조 설계

앞으로 실제 개발 코드를 legacy 폴더에서 직접 수정하는 것은 적절하지 않다고 판단했다.

따라서 새로운 canonical development path를 만들었다.

```text
src/
└── khuda_cv/
    ├── __init__.py
    └── model.py
```

역할은 다음처럼 구분했다.

```text
src/khuda_cv/
→ 앞으로 수정하고 발전시킬 canonical implementation

legacy/final_package/
→ 최종 배포본 historical reference

seraph/
→ 과거 training / experiment implementation
```

## 4. Canonical Model 복사

`legacy/final_package/model.py`를 새로운 canonical path인:

```text
src/khuda_cv/model.py
```

로 복사했다.

원본과 새 파일의 SHA256 hash를 비교했다.

```text
A853D323D5F0777CE39A0D6DC34E86825F47AD7880F4EF06526F2D13F958A69E
```

두 파일의 hash가 완전히 동일한 것을 확인했다.

즉 이 단계에서는 behavior를 변경하지 않고 파일 위치만 canonical development 영역으로 복제했다.

## 5. Harness Import 경로 전환

Day 3에서 만든 Harness는 기존에는:

```text
legacy/final_package/model.py
```

를 직접 검증하고 있었다.

이를 새로운 canonical path로 변경했다.

```text
src/khuda_cv/model.py
```

수정한 테스트:

```text
tests/test_imports.py
tests/test_model_contract.py
tests/test_checkpoint_contract.py
```

Import Smoke Test 결과:

```text
2 passed
```

새 canonical package에서 모델을 정상적으로 import할 수 있음을 확인했다.

## 6. Model Contract 재검증

새로운 `src/khuda_cv/model.py`를 대상으로 기존 Model Contract Test를 다시 실행했다.

검증 대상:

```text
Input
(B, 16, 3, 224, 224)

Output
clip_logits  → (B, 2)
frame_logits → (B, 16)
last_hidden_state → exists
```

결과:

```text
1 passed
```

파일 위치를 변경한 이후에도 기존 model behavior가 그대로 유지되는 것을 확인했다.

## 7. Checkpoint 호환성 재검증

다음으로 실제 `best_model.pt`와 새 canonical model의 호환성을 다시 확인했다.

`strict=True` checkpoint load test 결과:

```text
1 passed
```

즉 `src/khuda_cv/model.py`는 기존 학습 checkpoint와 완전히 호환된다.

이 검증을 통해 코드 위치 변경이 모델 artifact에 영향을 주지 않았음을 확인했다.

## 8. Legacy Boundary 명문화

향후 Agent가 다시 legacy 구현을 canonical model로 오해하지 않도록 각 영역의 역할을 문서로 명시했다.

추가한 파일:

```text
legacy/final_package/README.md
seraph/README_LEGACY.md
```

최종 역할 구분:

```text
seraph/
→ earlier training / experiment code

legacy/final_package/
→ final packaged historical reference

src/khuda_cv/
→ active canonical development implementation
```

기존 legacy code는 삭제하지 않고 historical context로 보존했다.

## 9. Full Harness 검증

Canonical path 전환과 legacy boundary 정리가 끝난 뒤 전체 Verification Harness를 다시 실행했다.

```bash
python -m pytest
```

결과:

```text
9 passed
```

즉 이번 리팩터링으로 기존 model, checkpoint, dataset, preprocessing contract가 깨지지 않았음을 확인했다.

## 10. Day 4 Result

Day 4에서는 처음으로 실제 repository 구조를 변경했지만, Day 3에서 만든 Harness 덕분에 behavior를 유지하면서 안전하게 변경할 수 있었다.

변경 전:

```text
legacy/final_package/model.py
→ 실제 canonical model이 legacy 영역에 존재
```

변경 후:

```text
src/khuda_cv/model.py
→ canonical development path

legacy/final_package/
→ historical reference

seraph/
→ legacy experiment code
```

최종 상태:

```text
Canonical model import      PASS
Model contract              PASS
Checkpoint compatibility    PASS
Full Harness                9/9 PASS
```

Day 4의 핵심은 단순히 파일을 옮긴 것이 아니라, 앞으로 Agent가 작업할 **명확한 canonical development path를 만들었다는 것**이다.

```text
Day 1 — Observe
Day 2 — Define
Day 3 — Enforce
Day 4 — Canonicalize
```

다음 단계부터는 `src/khuda_cv/`를 중심으로 legacy train/test/dataset 코드를 canonical contract에 맞게 점진적으로 리팩터링할 수 있다.
