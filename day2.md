# Day 2 Harness Contracts and Agent Context

Day 2의 목표는 코드를 바로 고치기보다, 앞으로의 refactor와 harness 작업이 따라야 할 기준을 먼저 고정하는 것이었다. Day 1에서 확인한 문제들은 대부분 "무엇이 맞는 인터페이스인지"가 불명확해서 생긴 위험이었다. 그래서 Day 2에서는 canonical model, 데이터 입력, checkpoint 호환성, 작업 경계를 문서로 확정했다.

## 1. Baseline 재확인

먼저 working tree 상태를 확인했다. 기존 production code를 수정하지 않고, 사용자가 복사한 `legacy/final_package/`의 세 파일을 기준 자료로 보존했다.

보존한 파일:

- `legacy/final_package/model.py`
- `legacy/final_package/pipeline.py`
- `legacy/final_package/requirements.txt`

이 파일들은 final package에서 가져온 참조 구현이며, Day 2에서는 내용을 변경하지 않았다.

## 2. Project Contract 확정

Day 2에서 확정한 canonical contract는 다음과 같다.

```text
Canonical model: GarbageDumpingVideoMAE
Input:           (B, 16, 3, 224, 224)
Normalization:   ImageNet mean/std
Output:          dict
  - clip_logits:       (B, 2)
  - frame_logits:      (B, 16)
  - last_hidden_state
Labels:
  - 0 = legal
  - 1 = illegal
Batch interface: dict
```

중요한 결정은 source video window와 model input을 분리한 것이다. 현재 pipeline과 dataset 설명에는 48-frame source window가 등장하지만, VideoMAE에 들어가는 실제 입력은 16 sampled frames다. 앞으로의 harness와 refactor는 이 차이를 보존해야 한다.

## 3. `project_package` 조사와 Canonical Model 결정

`legacy/final_package/model.py`를 확인한 결과 `GarbageDumpingVideoMAE`가 다음 구조를 가진다.

- VideoMAE backbone
- 2-class clip head
- 16-frame frame head
- dict output

반면 현재 `seraph/models/model.py`의 `VideoMAEMultiHead`는 legacy training path로 남긴다.

차이점:

- `VideoMAEMultiHead`는 tuple output을 반환한다.
- clip head가 one-logit BCE 방식이다.
- frame output 길이가 16-frame canonical contract와 다를 수 있다.
- `seraph/train.py`는 tuple batch 흐름을 사용한다.

따라서 Day 2의 결정은 `GarbageDumpingVideoMAE`를 canonical model로 삼고, `VideoMAEMultiHead`는 향후 refactor 대상인 legacy training path로 문서화하는 것이다.

## 4. Checkpoint Key 실검증 반영

`best_model.pt`는 `GarbageDumpingVideoMAE`와 호환되는 checkpoint로 고정했다. 확인된 `model_state_dict` 구조는 다음과 같다.

- `backbone.*` 182 keys
- `clip_head.0.*`
- `clip_head.3.*`
- `frame_head.0.*`
- `frame_head.3.*`

이 구조는 final-package의 canonical MLP head 구조와 맞는다. 그래서 checkpoint contract에는 module name과 head layer 구조를 임의로 바꾸지 말라는 금지 규칙을 명시했다.

## 5. 변경 경계 정리

Day 2에서는 작업 영역을 세 가지로 나눴다.

### PROTECTED

- `legacy/final_package/`
- checkpoint semantics
- label semantics
- 현재 production training/evaluation code

### REFACTOR

- train/test model factory 통합
- dict batch interface 정리
- source window와 model input frame 수 분리
- hard-coded path 정리
- dependency 명세 정리

### NEW

- `tests/`
- `harness/`
- small fixtures
- contract docs

이 경계를 먼저 둔 이유는, 코드를 고치기 전에 어떤 변경이 안전한지 판단할 기준이 필요했기 때문이다.

## 6. `AGENTS.md` 추가

`AGENTS.md`에는 future agent가 따라야 할 작업 규칙을 넣었다.

포함한 내용:

- canonical contract
- PROTECTED / REFACTOR / NEW 경계
- before/while/after editing rules
- checkpoint protection
- large artifact policy
- Definition of Done

이 파일은 앞으로 agent가 저장소를 열었을 때 가장 먼저 읽어야 하는 작업 안내서 역할을 한다.

## 7. Context Docs 추가

`docs/` 아래에 다음 문서를 추가했다.

| File | Purpose |
| --- | --- |
| `docs/architecture.md` | 전체 구조, canonical/legacy 관계, 향후 refactor 방향 |
| `docs/data-contract.md` | input shape, label mapping, frame sampling, dict batch 계약 |
| `docs/checkpoint-contract.md` | checkpoint key 구조, 호환 모델, 금지 변경 |
| `docs/execution-baseline.md` | 로컬 실행 환경, known blockers, smoke-check 명령 |
| `docs/agent-task-template.md` | future agent에게 넘길 작업 명세 템플릿 |

문서들은 production code를 수정하지 않고, 앞으로의 작업 기준을 분명하게 만드는 데 집중했다.

## 8. Execution Baseline

Day 2 기준 환경 정보는 다음과 같다.

| Component | Version |
| --- | --- |
| Python | `3.12.10` |
| torch | `2.11.0+cu128` |
| transformers | `5.4.0` |
| OpenCV | `4.13.0` |
| decord | initially missing |

known blockers:

- local environment에서 `decord`가 처음에는 없어서 dataset import가 실패할 수 있다.
- repo 안에 Linux absolute path가 남아 있다.
- train/test가 서로 다른 model output contract를 기대한다.
- tuple batch와 dict batch가 섞여 있다.

이 문제들은 Day 2에서 바로 수정하지 않고, 향후 harness test를 붙인 뒤 단계적으로 refactor할 대상으로 남겼다.

## 9. Day 2 결론

Day 2의 핵심 산출물은 코드 변경이 아니라 계약의 고정이다.

이제 저장소에는 future agent가 따라야 할 기준이 생겼다.

- 어떤 모델이 canonical인지
- 어떤 checkpoint 구조를 보존해야 하는지
- 실제 입력 frame 수가 무엇인지
- label 의미가 무엇인지
- dataset/batch interface가 무엇인지
- 어떤 파일을 보호하고 어떤 영역부터 refactor할지

다음 단계에서는 이 문서들을 기준으로 small harness tests를 추가하고, production code를 한 번에 크게 바꾸지 않고 contract-by-contract로 정리하면 된다.
