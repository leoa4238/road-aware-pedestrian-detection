# 진행 상황

> **최신 (2026-10-14):** 먼저 `docs/PROGRESS_FAILURE_MODES_20261014.md`를 읽는다 (실패 유형 분석, 세션 단위 split `session_v1`, 다음 작업).
> 그 앞 단계의 오류 분석은 `docs/PROGRESS_ERROR_ANALYSIS_20261014.md`. 아래는 2026-10-10 기록이다.

## 2026-10-10 (PC방, RTX 4060 Ti 8GB)

### 한 일

1. AI-Hub 058 생활도로 전방(F) 라벨 분석: 이미지 90,035장, 보행자 포함 60,659장, 보행자 약 17.5만 명.
2. 파이프라인 작성 및 실데이터 검증 (`aihub/`).
   - 라벨 1만 장 처리 7초, clip 1,106개 기준 split, train/test clip 겹침 0.
   - Windows 한글 경로에서 cv2.imread/imwrite가 실패하는 문제 수정.
   - 보행자 발 픽셀을 SegFormer가 person으로 칠해 도로 판정이 틀리는 문제 수정 (발 주변 바닥 창으로 판정).
3. Validation K 이미지만 사용(5,501장: train 3,894 / val 798 / test 809)해 YOLOv8s 베이스라인 학습.
   - 50 epoch 완료. test mAP50 0.734, mAP50-95 0.524, 보행자 mAP50 0.789 (`outputs/aihub/results/yolov8s_e50_s0.json`).
   - 가중치와 예시 그림은 Drive의 `rapd_weights_20261010.zip`.

### 발견

| 보행자 위치(K, 10,340명) | 비율 |
|---|---:|
| 도로 위 | 68% |
| 보도 위 | 18% |
| 기타(차량 탑승자, 건물 앞 등) | 14% |

- 생활도로는 연석이 없어 보행자 대부분이 차도 위에 있다 → 도로/보도 이분법으로는 위험 보행자를 정의할 수 없다.
- Cityscapes로 학습한 SegFormer는 **흰 선으로만 구분된 벽돌 보행로를 차도로 판정**한다.
  → 해외 데이터 기반 분할 모델은 한국 생활도로 보행 공간을 구분하지 못한다는 연구 필요성 근거.
- AI-Hub LiDAR 라벨에는 Road/Sidewalk 클래스가 있으므로, 카메라로 투영해 한국 도로용 도로·보도 분할 정답을 만들 수 있다 (검토 필요).

### 베이스라인 분석 (confusion matrix)

- 보행자의 21%를 놓침(background) → 제안 모델의 개선 목표. 멀리 있거나 가려진 작은 보행자로 추정, 확인 필요.
- 이륜차/PM은 25~28% 놓침 → 데이터 부족. 전체 데이터에서 재확인.
- 오탐의 62%가 승용차 → 라벨이 없는 먼 주차 차량 가능성, 라벨링 기준 확인.
- test(0.734)와 val(0.744)이 비슷하고 30 epoch 이후 수렴 → 누수/과적합 없음.
- 선배 논문 수치와는 데이터·클래스·평가 방식이 달라 직접 비교 불가.

## 결정 사항 (2026-10-10 논의)

### 문제 제기 (3단계)

1. 사회적 문제: 생활도로는 보도·차도 미분리로 보행자 사고 위험이 높다.
   → TAAS(도로교통공단) 도로 폭/유형별 보행자 사고 통계를 직접 찾아 출처와 함께 인용. AI-Hub 데이터 구축 목적 문구 인용.
2. 기술의 한계: 보행자 **존재** 탐지만으로는 위험 보행자를 구분할 수 없다 (보행자 68%가 차도 위).
   해외 데이터 기반 분할 모델은 벽돌 보행로를 차도로 오인.
3. 연구 공백: YOLOP, HybridNets는 도로 영역을 별도 출력만 하고 탐지에 활용하지 않는다.

주의: "생활도로에서 보행자 탐지 자체가 어렵다"는 아직 근거 없음. 주장하려면 아래 실험 필요.

### 생활도로 정의

법적 정의가 명확하지 않으므로 "AI-Hub 058 데이터의 도로 유형 분류(R01~R06)를 따른다"고 정의.
보차도로(R03)가 41.5%로 가장 많음. 경찰청 생활도로구역, 안전속도 5030 자료로 보강(직접 확인).

### 모델 기여 (추천 구성)

| 모델 | 구성 |
|---|---|
| Baseline | YOLOv8s (완료) |
| + Road input | 도로 마스크를 입력 채널로 추가 |
| + RGA | Road-Guided Attention (핵심 제안 모듈) |
| + RGA + Boundary | 도로 경계·진입 구간 강조 prior (distance transform) — 생활도로 차별점 |
| Full | + 위험도 head (보행자별 위험 여부 출력) |

- CBAM vs RGA를 같은 조건에서 직접 비교 (선배 논문에 없는 실험).
- 위험도 head 입력에 **관절(pose)** 추가 검토: 몸 방향, 발목 위치, 고개 방향.
  JAAD 선행 연구(bbox 0.656 → pose 0.711 → pose+road 0.755)를 근거로 확장.
  탐지 단계에는 관절을 넣지 않는다(작은 보행자는 관절도 불안정). 일정이 밀리면 ablation 하나로 축소.
  AI-Hub 서브라벨 keypoint 1만 장(COCO 17점) + 나머지는 YOLOv8-pose로 자동 추출.
- 평가 지표: 보행자 mAP50, **위험 보행자 recall(핵심)**, 위험 판정 precision(오경보), 조건별(야간/우천/R03), GFLOPs/FPS.

### 확장 (여유 있을 때)

- GPS 위험 지점 지도: Other.zip(69MB)에 GPS가 있는지 확인. 실시간 전송(V2X)은 향후 연구.
- 블랙박스 외부 검증: 사용자가 생활도로 블랙박스 영상 촬영 → 200~300 프레임 라벨링(CVAT/Label Studio)
  → 베이스라인/제안 모델 일반화 성능 + 시연 영상. 얼굴·번호판 모자이크, 원본 비공개, IRB 필요 여부 교수님께 확인.
- 영상 기반 추적·이동 방향(접근 속도)은 향후 연구 (JAAD approach_rate와 연결).

### 일정 (약 3개월, 2026-10 ~ 2027-01)

| 기간 | 내용 |
|---|---|
| 1~2주 | 위험 보행자 정의 확정, 전체 데이터 구축, 베이스라인 확정, 교수님 보고 |
| 3~5주 | RGA + 경계 prior + 위험도 head 구현 |
| 6~8주 | 비교 실험, 블랙박스 외부 검증 |
| 9~12주 | 논문 작성 (선배 논문 형식, 5장) |

## 다음 할 일 (우선순위 순)

1. **교수님 보고**: `docs/advisor_brief.md` + Drive `rapd_weights_20261010.zip`의 `figures/zone_2.jpg`(벽돌 보행로 오인식).
   여쭐 것: 주제 적절성, 위험 보행자 정의 기준, GPU 서버, 블랙박스 촬영 IRB.
2. **위험 보행자 정의 확정**: 주행 경로(이미지 하단 중앙)까지 거리 vs 도로 가장자리 기준. 여러 정의를 숫자와 이미지로 비교.
3. **베이스라인이 놓친 보행자 분석**: 도로 위/보도/거리(bbox 크기)별로 놓친 비율.
4. **도로 유형별 보행자 위치 분포**: 특히 보차도로(R03)만 따로 → 문제 제기 근거 강화.
5. **일반 도로 모델 적용 실험**: COCO 사전학습 YOLO를 학습 없이 생활도로 test에 적용 → 성능이 떨어지면
   "일반 도로 모델은 생활도로에서 약하다" 근거 추가. 평가만이라 10분.
6. V 이미지(VS_calib_V_가시광이미지_image_F.zip, 10GB) 추가 후 데이터셋 재생성.
7. RGA 구현 시작.
8. (선택) Other.zip의 GPS·정합 파일 확인, LiDAR Road/Sidewalk 투영 가능성 검토.

## 준비된 환경(Python/PyTorch 있는 PC나 서버)에서 시작하는 법

```bash
git clone https://github.com/leoa4238/road-aware-pedestrian-detection.git && cd road-aware-pedestrian-detection
pip install -r aihub/requirements.txt        # PyTorch는 CUDA 버전에 맞게 별도 설치
# Drive의 rapd_backup_YYYYMMDD.zip을 원하는 곳에 풀기 -> <DATA>/aihub_f
python aihub/train_baseline.py --data <DATA>/aihub_f/dataset.yaml --model yolov8s.pt --epochs 50
```

dataset.yaml 경로는 학습 시작 때 자동으로 실제 위치로 고쳐진다.

## PC방에서 다시 시작하는 법

1. 환경 설치 (15~20분): `aihub/setup_pcbang.ps1` 실행.
2. Google Drive의 `rapd_backup_YYYYMMDD.zip`을 받아 `C:\work\data\`에 압축 해제 → `C:\work\data\aihub_f`.
   - 이미지, YOLO 라벨, 도로 마스크, `objects_road.csv`, 학습 가중치 포함.
   - `dataset.yaml`이 `C:/work/data/aihub_f` 경로를 가리키므로 같은 위치에 풀 것.
3. AI-Hub 원본 zip은 데이터셋을 새로 만들 때만 필요.
4. 끝낼 때: 결과 push → 백업 zip을 Drive에 업로드 → `cmdkey /list`로 GitHub 자격 증명 삭제, 브라우저 로그아웃.
