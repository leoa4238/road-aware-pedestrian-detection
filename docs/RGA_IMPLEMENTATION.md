# RGA 구현 (2026-10-14, Claude Code)

**채택 구현:** `aihub/road_yolo.py`, `aihub/train_road.py` (2026-10-14 사용자 결정).
같은 날 작성된 `docs/RGA_V0_PROTOCOL_20261014.md`, `aihub/smoke_rga.py`, `tests/test_rga.py`(Codex, 미완성)는 채택하지 않았다.
그 설계의 보도 채널, "마스크 없음"과 "기타 영역"의 구분, 조절 폭 제한은 필요하면 이 구현에 옵션으로 추가한다.

근거: `docs/PROGRESS_FAILURE_MODES_20261014.md` 6-2절. 목표는 작은 보행자 후보의 점수를 배경과 구분되게 올리는 것이다.

## 입력 (A안: 미리 만든 SegFormer 마스크)

- 이미지(RGB)와 도로 마스크를 한 배열의 4채널로 묶어 읽는다. 모자이크, affine, 뒤집기, letterbox가 둘을 똑같이 변형한다.
- 4번째 채널은 도로 확률이다. 도로는 255, 나머지는 0이고, 모자이크 여백은 114로 "알 수 없음"을 뜻한다.
- Ultralytics는 3채널이 아니면 HSV와 Albumentations 증강, BGR→RGB 변환을 조용히 건너뛴다. 이 구현은 이 셋을 앞 3채널에 직접 적용해서 baseline과 같은 증강을 받게 했다.
- 실제 사용 시에는 SegFormer를 함께 돌려야 하므로, 논문에서 그 연산량을 따로 보고한다.

## 모델

YOLOv8s(사전학습 가중치)의 backbone과 neck은 RGB만 받는다. 모듈은 Detect 직전 P3, P4, P5 각각에 붙는다.

```
a  = tanh(Conv3x3(Conv3x3([Conv1x1(F), road, boundary])))
F' = F * (1 + gamma * a)        # gamma = 0으로 시작 → 처음에는 baseline과 같은 함수
```

| MODE | 내용 | 역할 |
|---|---|---|
| (baseline) | `train_baseline.py` | 기준선 |
| `concat` | 마스크를 첫 conv의 4번째 입력 채널로 추가 (가중치 0으로 시작) | 단순 도로 입력 |
| `attn` | 같은 게이트, 도로 입력을 0으로 | 같은 크기의 일반 attention (대조군) |
| `rga` | 도로 확률로 조절 | **제안 모듈** |
| `rgab` | rga + 도로 경계 채널 | 경계 추가 효과 |

- 추가 파라미터: 게이트 방식 22.7K (약 0.2%), concat 288개.
- `--mask-close 31`: 마스크에서 사람이나 차량 모양의 구멍을 메우는 대조 실험이다. SegFormer는 사람을 person으로 칠하므로 도로 마스크에 사람 모양의 구멍이 생긴다. 모델이 "도로 맥락"이 아니라 "SegFormer의 사람 인식"을 힌트로 쓰는지 가르기 위한 실험이다. 커널은 640×360 마스크 기준 31px이라 작은 보행자의 구멍은 메워지지만, 아주 큰 보행자는 일부만 메워진다.

## 확인한 것

| 확인 | 결과 |
|---|---|
| 앞 3채널이 baseline 입력과 같은가 (val, 증강 없음) | 완전히 같음 |
| 4번째 채널이 마스크 파일과 맞는가 | 도로 비율 0.3385 vs 0.3388 |
| 학습 시작 시 출력이 baseline과 같은가 (4개 MODE) | 차이 0.0 |
| 학습 이미지 색이 맞는가 | `train_batch0.jpg` 육안 확인. 노란 차선, 빨간 미등 정상 |
| 학습, 저장, 재로드, test 평가가 끝까지 도는가 | train 5%, 2 epoch, GTX 1650에서 완료 (성능 비교용 아님) |
| 게이트가 학습되는가 | gamma 0 → ±0.0005 (짧은 시험이라 작음) |
| 단위 테스트 | `tests/test_road_yolo.py` 5개 통과 |

전체 학습 뒤에는 `best.pt`의 gamma 값을 확인한다. gamma가 거의 0이면 모듈이 사실상 쓰이지 않은 것이므로 결과를 해석하기 전에 짚어야 한다.

## 실행

Colab: `docs/COLAB_TRAINING.md` 셀 6. 로컬:

```powershell
python aihub\train_road.py --data <DATA>\aihub_f\dataset_session_v1.yaml --mode rga --seed 0
```

Windows에서는 `--workers 0`을 권한다. 결과 JSON은 `outputs/aihub/results/yolov8s_<mode>_e50_s<seed>.json`에 저장된다.

## 실험 계획과 판단 기준

1. 1차: `rga`, `attn`, `concat`, `rgab` 각 seed 0
2. 2차: 모두 seed 1, 2 (baseline 포함). 평균 ± 표준편차로 보고
3. 대조: `rga --mask-close 31` seed 0

제안 모듈이 효과가 있다고 말하려면 다음이 필요하다.
- `rga`가 `attn`(같은 크기, 도로 정보 없음)보다 나아야 한다. baseline보다 나은 것만으로는 부족하다.
- 차이가 시드 간 표준편차보다 커야 한다.
- 주 지표는 보행자 AP50과 32~64px 보행자 재현율이다. 실패 유형 분석(`analyze_failure_modes.py`)을 다시 돌려 below_threshold 미탐이 몇 개 회복됐는지도 본다.

주의: 실패 유형 분석 코드는 아직 3채널 모델 전용이다. RGA 모델용 추론(4채널 입력)은 첫 학습 결과가 나온 뒤 추가한다.
