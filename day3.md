\# Day 3 — Contract Test Harness



Day 1에서 기존 프로젝트의 baseline을 파악하고, Day 2에서 canonical model과 project contract를 정의했다.



Day 3의 목표는 문서에만 존재하던 contract를 자동화된 테스트로 바꾸는 것이었다.



\## 1. Day 2 Baseline 확인



작업 전 repository가 Day 2 결과와 동기화되어 있는지 확인했다.



\- `main == origin/main`

\- working tree clean

\- `AGENTS.md` 존재

\- `docs/` context 문서 존재

\- `legacy/final\_package/` 존재



\## 2. Test Harness 구조 설계



다음 contract를 테스트 대상으로 정했다.



```text

tests/

├── test\_imports.py

├── test\_model\_contract.py

├── test\_checkpoint\_contract.py

├── test\_dataset\_contract.py

└── test\_preprocessing\_contract.py

```



테스트는 실제 대용량 artifact가 없어도 가능한 fast test와 checkpoint가 필요한 artifact-dependent test로 구분했다.



\## 3. Import Smoke Test



Canonical model 파일과 주요 symbol이 정상적으로 import되는지 확인했다.



검증 항목:



\- `legacy/final\_package/model.py` 존재

\- `GarbageDumpingVideoMAE` 존재

\- `build\_model` 존재



Result:



```text

2 passed

```



\## 4. Model Contract Test



Dummy input을 실제 canonical model에 전달해 Day 2에서 정의한 model contract를 검증했다.



```text

Input

(1, 16, 3, 224, 224)



Output

clip\_logits       → (1, 2)

frame\_logits      → (1, 16)

last\_hidden\_state → exists

```



또한 model output이 dictionary인지 확인했다.



Result:



```text

1 passed

```



\## 5. Checkpoint Contract Test



실제 `best\_model.pt`를 사용해 checkpoint와 canonical model의 호환성을 테스트했다.



검증 항목:



\- `model\_state\_dict` 존재

\- `clip\_head.\*` key 존재

\- `frame\_head.\*` key 존재

\- `backbone.\*` 182개 확인

\- `GarbageDumpingVideoMAE`에 `strict=True` load 성공



Result:



```text

1 passed

```



이 테스트를 통해 이후 Agent가 architecture나 layer name을 변경해 기존 checkpoint를 깨뜨리는 상황을 감지할 수 있다.



\## 6. Dataset Contract Test



Day 2에서 정의한 canonical Dataset/Batch interface를 synthetic sample을 이용해 테스트했다.



```text

Sample

pixel\_values → (16, 3, 224, 224)

clip\_label   → scalar

frame\_labels → (16,)



Batch

pixel\_values → (B, 16, 3, 224, 224)

clip\_label   → (B,)

frame\_labels → (B, 16)

```



Dataset과 Batch 모두 dictionary interface를 표준으로 한다.



Result:



```text

2 passed

```



\## 7. Preprocessing Contract Test



영상 전처리 계약도 테스트로 고정했다.



검증 항목:



\- 16-frame 유지

\- 224 × 224 resize

\- `float32` tensor

\- ImageNet mean/std normalization

\- black / white pixel normalization 계산



Result:



```text

3 passed

```



\## 8. Test Runner 구축



Repository root에 `pytest.ini`를 추가해 전체 Harness를 다음 한 명령으로 실행할 수 있도록 했다.



```bash

python -m pytest

```



최종 결과:



```text

9 passed

```



\## 9. Agent Verification Rule 연결



`AGENTS.md`에 Verification Harness 규칙을 추가했다.



앞으로 non-trivial code change 이후 Agent는 반드시:



```bash

python -m pytest

```



를 실행해야 한다.



테스트가 실패하면 테스트를 삭제하거나 약화해 통과시키는 것이 아니라, implementation과 contract 중 무엇이 잘못되었는지 먼저 판단하도록 규칙을 정의했다.



\## 10. Day 3 Result



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

Day 1 — Observe

Day 2 — Define

Day 3 — Enforce

```



다음 단계부터는 이 Harness를 안전망으로 사용해 legacy code를 실제로 리팩터링할 수 있다.

