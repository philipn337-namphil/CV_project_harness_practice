# Day 1 Harness Engineering Baseline

이 문서는 `CV_project_harness_practice` 저장소를 처음 안전하게 다루기 위한 Day 1 기준선이다.
목표는 production 코드를 바로 고치지 않고, 현재 구조와 실행 계약을 문서화한 뒤 어떤 영역을 보호하고 어떤 영역을 리팩터링할지 분리하는 것이다.

## 1. Repository Structure

현재 repo 루트 기준 구조는 다음과 같다.

```text
.
├── README.md
├── preprocessing/
│   └── khuda_173/
│       └── README_KHUDA_173_PREPROCESSING.md
└── seraph/
    ├── dataset.py
    ├── test.py
    ├── train.py
    ├── models/
    │   └── model.py
    ├── other/
    │   ├── dataset.py
    │   └── evaluate.py
    └── utils/
        ├── logger.py
        └── transforms.py
```

주요 파일의 현재 역할은 다음과 같다.

| Path | Current role |
| --- | --- |
| `seraph/models/model.py` | Hugging Face VideoMAE backbone + clip/frame multi-head 모델 정의 |
| `seraph/dataset.py` | KHUDA manifest 기반 clip dataset, dict batch용 `collate_fn` 포함 |
| `seraph/utils/transforms.py` | train/val dataloader wrapper, tuple batch용 collate 포함 |
| `seraph/train.py` | progressive unfreezing 기반 학습 루프 |
| `seraph/test.py` | checkpoint 로드, test manifest inference, 단일 video Grad-CAM 시각화 |
| `seraph/other/evaluate.py` | 별도 evaluation script 성격의 보조 파일 |
| `preprocessing/khuda_173/` | KHUDA_173 preprocessing 관련 문서 |

## 2. VideoMAE Multi-head Structure

현재 `seraph/models/model.py`의 production 모델 클래스는 `VideoMAEMultiHead`이다.

```text
Input: pixel_values
  shape: (B, T, C, H, W)
        ↓
VideoMAEModel backbone
        ↓
last_hidden_state
  shape: (B, N, D)
        ↓
├── clip_head
│   └── out.mean(dim=1) → Linear(D, 1) → clip_logit
│
└── frame_head
    └── temporal tokens 평균 → Linear(D, 1) → frame_logits
```

현재 모델의 핵심 계약:

- Backbone: `transformers.VideoMAEModel`
- Default checkpoint constant: `MCG-NJU/videomae-base-finetuned-kinetics`
- `train.py`에서는 `MCG-NJU/videomae-base`를 직접 전달한다.
- `clip_head`: clip 단위 illegal/legal binary logit
- `frame_head`: temporal token 단위 suspicious frame/event logit
- `t_tokens = num_frames // backbone.config.tubelet_size`
- VideoMAE base의 tubelet size가 2이면, `num_frames=48`일 때 frame head 출력 길이는 24가 된다.
- Progressive unfreezing method:
  - `unfreeze_heads()`
  - `unfreeze_last_layers(num_layers=1)`
  - `unfreeze_all()`

## 3. Dataset to Train/Test Flow

현재 코드상 의도된 흐름은 아래와 같다.

```text
KHUDA manifest json
  clips_train_v2.json / clips_val_v2.json / clips_test_v2.json
        ↓
GarbageDumpingClipDataset
  - video_path resolve
  - decord VideoReader
  - sampled frame indices
  - ImageNet normalization
  - clip_label
  - frame_labels
        ↓
DataLoader
        ↓
VideoMAEMultiHead or test-side model
        ↓
clip-level loss / metric
frame-level loss / metric
```

현재 train 경로:

```text
seraph/train.py
  → MyDataset(...)
  → seraph/utils/transforms.py
  → GarbageDumpingClipDataset(...)
  → DataLoader(..., collate_fn=self._tuple_collate_fn)
  → for batch in train_loader:
        frames, clip_label, frame_labels = batch
  → VideoMAEMultiHead(...)
```

현재 test 경로:

```text
seraph/test.py
  → GarbageDumpingClipDataset(...)
  → DataLoader(..., collate_fn=collate_fn)
  → batch["pixel_values"], batch["clip_label"], batch["frame_labels"]
  → GarbageDumpingVideoMAE()
  → outputs["clip_logits"], outputs["frame_logits"]
```

여기서 중요한 점은 train과 test가 같은 모델/출력 형식/배치 형식을 공유하지 않는다는 것이다.

## 4. Environment Baseline

Day 1에서 확인한 로컬 환경 기준선:

| Component | Observed baseline |
| --- | --- |
| Python | `3.12.10` |
| torch | `2.11.0+cu128` |
| transformers | `5.4.0` |
| OpenCV | `4.13.0.92` |
| decord | not installed |

현재 repo에는 production 실행에 필요한 dependency 명세가 고정되어 있지 않다.
특히 `seraph/dataset.py`와 `seraph/utils/transforms.py`는 import 시점에 `decord`를 필요로 하므로, `decord`가 없으면 dataset import부터 실패할 수 있다.

## 5. Problems Found

Day 1에서 발견한 주요 문제점은 다음과 같다.

| Issue | Current observation | Risk |
| --- | --- | --- |
| `decord` 누락 | dataset 계층이 `import decord`를 직접 수행하지만 현재 환경에는 설치되어 있지 않음 | 학습/평가 import 단계에서 즉시 실패 가능 |
| dependency 명세 부재 | repo 루트에 고정된 `requirements.txt` 또는 lock file이 없음 | 다른 환경에서 재현 어려움 |
| Linux 절대경로 하드코딩 | `/data/leecg1219/...`, `/data/philipn337/...`, `/data2/local_datasets/...`가 train/test/dataset에 박혀 있음 | 로컬/Windows/다른 서버에서 바로 실행 불가 |
| `test.py` import 불일치 | `seraph/test.py`가 `GarbageDumpingVideoMAE`를 import하지만 현재 `seraph/models/model.py`에는 `VideoMAEMultiHead`만 존재 | test 실행 시 import error 가능 |
| dataset dict 반환과 tuple collate 불일치 가능성 | `GarbageDumpingClipDataset.__getitem__`은 dict를 반환하지만 `MyDataset._tuple_collate_fn`은 tuple item을 기대 | train dataloader가 manifest dataset을 사용할 때 batch 생성 실패 가능 |
| 48/16 frame contract 혼재 | dataset 주석은 48-frame clip에서 16 frame sampling을 말하지만 train은 `CLIP_LEN=48`을 모델 `num_frames`로 넘김 | frame label 길이와 model frame logits 길이 불일치 가능 |
| train/test version mismatch 가능성 | train은 `VideoMAEMultiHead` tuple-output/BCE 흐름, test는 `GarbageDumpingVideoMAE` dict-output/softmax 흐름을 기대 | checkpoint 호환성 및 metric 해석 불일치 가능 |

## 6. Protected / Refactor / New Areas

Day 1 이후 작업은 아래 영역 구분을 지켜서 진행한다.

### PROTECTED

아직 임의로 수정하지 않는 영역:

- `seraph/train.py`
- `seraph/test.py`
- `seraph/dataset.py`
- `seraph/models/model.py`
- `seraph/utils/transforms.py`
- 기존 checkpoint path, manifest path, dataset path semantics
- 기존 README 내용

이유: 현재 코드는 실행 계약이 섞여 있어 작은 수정도 train/test/checkpoint 호환성에 영향을 줄 수 있다.
먼저 harness로 현상을 고정한 뒤 production code를 고친다.

### REFACTOR

다음 단계에서 정리할 후보:

- config/env 기반 path 주입
- train/test가 공유하는 단일 model factory
- `VideoMAEMultiHead` 출력 형식 통일
- dataset 반환 형식과 collate function 통일
- 48 raw clip frame과 16 model input frame 계약 명시
- frame label downsample 또는 tubelet alignment 로직 통일
- dependency 파일 추가
- import path를 package-safe하게 정리

### NEW

production code와 분리해서 새로 둘 수 있는 harness 영역:

- `tests/` 또는 `harness/`
- import smoke test
- model shape contract test
- dataset sample/collate contract test
- train/test entrypoint dry-run test
- fixture manifest와 tiny fake video/sample tensor
- environment snapshot 문서 또는 script

## 7. Day 1 Checklist

- [x] GitHub repo가 존재하는지 확인
- [x] 로컬 working tree 상태 확인
- [x] production code 수정 없이 문서 파일만 추가
- [x] repository 구조 기록
- [x] VideoMAE multi-head 구조 기록
- [x] dataset to train/test 흐름 기록
- [x] Day 1 환경 baseline 기록
- [x] decord 누락 기록
- [x] dependency 명세 부재 기록
- [x] Linux 절대경로 하드코딩 기록
- [x] `GarbageDumpingVideoMAE` import 불일치 기록
- [x] dict 반환과 tuple collate 불일치 가능성 기록
- [x] 48/16 frame contract 혼재 기록
- [x] train/test 버전 불일치 가능성 기록
- [x] PROTECTED / REFACTOR / NEW 영역 구분

## 8. Next Steps

추천 순서:

1. `day1.md`를 기준선으로 커밋한다.
2. production code를 수정하기 전에 최소 harness를 만든다.
3. 첫 harness는 import smoke test부터 시작한다.
4. `decord`가 없는 환경에서도 어떤 import가 실패하는지 명확히 기록한다.
5. model contract test를 추가해 `VideoMAEMultiHead` 입력/출력 shape을 고정한다.
6. dataset contract test를 추가해 dict batch와 tuple batch 중 어느 쪽을 표준으로 삼을지 결정한다.
7. train/test가 같은 model factory와 같은 output contract를 쓰도록 리팩터링한다.
8. path 하드코딩을 CLI arg 또는 config/env 기반으로 이동한다.
9. dependency baseline을 `requirements.txt` 또는 별도 environment 문서로 고정한다.

Day 1의 결론:

```text
지금은 성능 개선 단계가 아니라 실행 계약을 고정하는 단계다.
production 코드를 바로 고치기보다, 현재 깨질 수 있는 지점을 harness로 먼저 붙잡는다.
```
