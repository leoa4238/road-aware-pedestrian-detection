# RCR-Head 설계 (2026-11-16, Claude Code)

**RCR-Head(Road-Context Risk Head, 이름은 제안):** 탐지기가 찾은 보행자 하나하나에 대해 위험·주의·안전을 판별하는 모듈이다.
RGA(도로 정보로 탐지 성능 향상)는 도로 마스크를 거의 쓰지 않았다(`docs/RGA_IMPLEMENTATION.md`).
파일럿에서는 도로 맥락이 위험 판별에 도움이 되었다(`docs/PROGRESS_20261020.md` 4절).
그래서 도로 정보를 **탐지**가 아니라 **위험 판별**에 쓰는 구조로 바꾼다.

코드: `aihub/rcr_head.py`(모델), `aihub/rcr_features.py`(입력 캐시), `aihub/rcr_train.py`(학습·비교), `tests/test_rcr_head.py`.

## 구조

```
이미지 ─▶ YOLOv8s (session_v1 학습, 고정) ─▶ 보행자 박스
              │ neck P3/P4/P5
              ▼
   ① 외형: RoIAlign 7×7 (3단계) ─▶ CNN ─▶ a (128)          자세·몸 방향을 탐지기 특징에서 직접
   ② 도로 래스터 [도로, 보도, 이 사람 박스, 발 위치] 64×112 ─▶ CNN ─▶ r (128)
   ③ 기하: 발 x·y, 박스 높이·폭, 화면 중앙에서의 좌우 거리 ─▶ MLP ─▶ g (32)
              ▼
   융합(FiLM, 제안): a' = a ⊙ (1 + γ(r, g)) + β(r, g),  γ = β = 0으로 시작
              ▼
   [a', r, g] ─▶ MLP ─▶ 위험도 (안전/주의/위험)
                      └▶ 보조 출력: 위치(경로 안/가장자리/차도/보행 공간), 방향(Q W E R), 가림, 경계, 횡단보도
```

설계 근거:

| 선택 | 이유 |
|---|---|
| 탐지기는 고정, 특징만 캐시 | 라벨이 400명이라 탐지기까지 학습하면 과적합. 탐지 성능은 베이스라인과 같게 유지 |
| 외형은 탐지기 neck의 RoI 특징 | 별도 pose 모델 없이 자세·방향 정보를 쓴다. 작은 보행자는 관절 추출이 불안정하다(10-10 결정) |
| 도로 정보는 장면 전체 래스터 | 파일럿에서 고정 경로 띠가 실제 경로를 못 잡았다. 곡선 도로·주차 차량 사이의 도로 모양을 CNN이 직접 본다 |
| FiLM 융합 | "같은 걷는 자세라도 도로 위치에 따라 위험이 다르다"를 구조로 표현한다(JAAD 예비 실험의 pose × road). 0으로 시작해 처음에는 외형만 쓰는 모델과 같다 |
| 조건 보조 출력 | 라벨 기준 v1이 조건을 하나씩 기록한다. 조건을 같이 배우면 적은 라벨로도 판단 근거를 학습하고, 예측한 조건에 규칙 v1을 적용한 위험도도 낼 수 있다(설명 가능성) |
| 좌우 반전 증강 | 위치·방향 라벨은 자차 경로 기준이라 반전해도 그대로다. 반전 이미지의 RoI 특징은 미리 계산한다 |

## 비교 실험 (`rcr_train.py`의 `VARIANTS`)

모든 변형은 **같은 세션 단위 5-fold × 5 seed**로 평가한다. 그래서 변형 사이 차이를 짝지어(paired) 비교할 수 있다.

| 변형 | 질문 |
|---|---|
| 규칙: 발이 도로 위면 위험 | 기존 방식(도로/보도 이분법)의 한계 |
| G | 위치만 |
| R | 도로 맥락만 |
| R+G | 도로 + 위치 (외형 없음) |
| A | 외형만 (도로 없음) |
| A+R+G concat | 단순 결합 |
| **A+R+G film** | 제안 융합 |
| **A+R+G film + conditions** | 전체 RCR-Head |

주 지표는 macro-F1과 **위험 재현율**이다. 보조 지표는 위험 정밀도(오경보)와 조건 정확도다.

제안 구조가 효과 있다고 말하려면 다음 셋이 모두 필요하다.
1. film > concat (같은 정보, 다른 결합). `paired_macro_f1_difference`에서 평균 차이가 seed 표준편차보다 크고, 대부분 seed에서 양수일 것
2. film > A, film > R+G (외형과 도로가 서로 보완)
3. 전체 > 규칙 (도로/보도 이분법보다 낫다)

RGA 때처럼 "정보를 지워도 결과가 같은지"도 확인한다. 테스트 시 래스터의 도로·보도 채널을 0으로 바꿔 성능이 떨어지는지 본다(다음 단계에서 추가).

## 실행

이 PC (GTX 1650, `D:\reserch\jaywalking-risk-recognition\.venv` 에 torch 2.5.1+cu121):

```powershell
$py = "D:\reserch\jaywalking-risk-recognition\.venv\Scripts\python.exe"
& $py aihub\rcr_features.py --set data\labeling\pilot_v0 --view data\views\session_v1 --weights D:\nonmoon\best.pt --out data\rcr\pilot_v0_features.npz
& $py aihub\rcr_train.py --features data\rcr\pilot_v0_features.npz --labels data\labeling\pilot_v0\attributes_pilot_v0.csv --out outputs\aihub\rcr\pilot_v0.json
```

- 이 venv의 ultralytics는 8.4.42로, 저장소 고정 버전(8.3.150)과 다르다. RCR 코드는 탐지기 특징만 읽어서 문제없다.
  다만 `tests/test_road_yolo.py`(RGA)는 이 venv에서 import 오류가 난다.
- `--labels`에 v0 라벨(`labels_partial_20261015.tsv`)을 주면 연구자 종합 판단으로도 같은 비교를 한다. 이 경우 조건 라벨이 없어서 조건 보조 출력은 학습되지 않는다.
- 캐시(`data/rcr/`)는 AI-Hub 이미지에서 나온 특징이라 커밋하지 않는다(`data/`는 .gitignore).

## 검증한 것 (합성 데이터, 실데이터 아님)

| 확인 | 결과 |
|---|---|
| 위험 규칙이 `pilot_risk_analysis.v1_rule`과 같은가 (위치 4 × 방향 4 × 가림 2) | 32개 조합 모두 같음 |
| 래스터: 도로 비율, 발 위치, 좌우 반전이 반전된 장면과 같은가 | 같음 |
| 7개 변형이 모두 돌고 역전파되는가, FiLM이 0에서 시작하는가 | 통과 |
| 무작위 가중치 YOLOv8s로 특징 추출이 끝까지 도는가 | P3/P4/P5 = 128/256/512 채널, 반전 포함 |
| 도로가 위험을 결정하는 합성 데이터에서 학습이 되는가 | 외형(잡음) 0.32, 도로 래스터 0.83, 전체 0.76 (macro-F1, 3-fold) |

## 다음 할 일

1. **(사용자)** 라벨 400명 완성 → 실데이터로 위 두 명령 실행.
2. 도로 채널 제거 시험과 RoI 크기·FiLM 위치 등 세부 ablation 추가.
3. 최종 평가: session_v1 **test** 보행자에 라벨을 따로 붙이고(학습에 쓰지 않음), 탐지기가 찾은 박스로 end-to-end 평가
   (위험 보행자 재현율 = 탐지 성공 × 위험 판별 성공).
4. 연산량: RCR-Head 파라미터·GFLOPs와 SegFormer 비용을 따로 보고.
