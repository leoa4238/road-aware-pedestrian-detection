# 진행 상황

## 2026-10-10 (PC방, RTX 4060 Ti 8GB)

### 한 일

1. AI-Hub 058 생활도로 전방(F) 라벨 분석: 이미지 90,035장, 보행자 포함 60,659장, 보행자 약 17.5만 명.
2. 파이프라인 작성 및 실데이터 검증 (`aihub/`).
   - 라벨 1만 장 처리 7초, clip 1,106개 기준 split, train/test clip 겹침 0.
   - Windows 한글 경로에서 cv2.imread/imwrite가 실패하는 문제 수정.
   - 보행자 발 픽셀을 SegFormer가 person으로 칠해 도로 판정이 틀리는 문제 수정 (발 주변 바닥 창으로 판정).
3. Validation K 이미지만 사용(5,501장: train 3,894 / val 798 / test 809)해 YOLOv8s 베이스라인 학습.
   - 7 epoch 시점 val mAP50 0.676, mAP50-95 0.460 (`outputs/aihub/results/yolov8s_baseline_partial_e7.csv`).

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

### 다음 할 일

1. V 이미지(VS_calib_V_가시광이미지_image_F.zip, 10GB) 추가 후 데이터셋 재생성.
2. 위험 보행자 정의 확정: 주행 경로(이미지 하단 중앙)까지의 거리 또는 도로 가장자리 기준.
3. LiDAR Road/Sidewalk → 카메라 투영 가능성 확인 (Other.zip의 정합 파일).
4. 제안 모듈(Road-Guided Attention) 구현 및 베이스라인과 비교.

## PC방에서 다시 시작하는 법

1. 환경 설치 (15~20분): `aihub/setup_pcbang.ps1` 실행.
2. Google Drive의 `rapd_backup_YYYYMMDD.zip`을 받아 `C:\work\data\`에 압축 해제 → `C:\work\data\aihub_f`.
   - 이미지, YOLO 라벨, 도로 마스크, `objects_road.csv`, 학습 가중치 포함.
   - `dataset.yaml`이 `C:/work/data/aihub_f` 경로를 가리키므로 같은 위치에 풀 것.
3. AI-Hub 원본 zip은 데이터셋을 새로 만들 때만 필요.
4. 끝낼 때: 결과 push → 백업 zip을 Drive에 업로드 → `cmdkey /list`로 GitHub 자격 증명 삭제, 브라우저 로그아웃.
