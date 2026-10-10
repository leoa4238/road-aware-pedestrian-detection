# CLAUDE.md

석사 졸업 논문 연구 저장소. 새 세션은 이 파일과 `PROGRESS.md`(최신 진행 상황, 다음 할 일)를 먼저 읽고 이어서 작업한다.

## 연구

- 주제(가제): 보차 혼용 생활도로에서의 도로 영역 기반 위험 보행자 탐지
- 학위 요건: **새 모델 구조 제안** 필요. 계획: YOLO + 도로 마스크로 탐지 특징을 강조하는 Road-Guided Attention 모듈.
- 연구실 선행 논문: 신상민(한양대 공학대학원, 2025.2) "도로 주행 환경에서의 객체 탐지 성능 향상을 위한 하이브리드 모델"
  (YOLOv5 + DETR + ASAM, AI-Hub 도로주행영상). 같은 형식(5장, 단계별 비교)을 따른다.
  그 논문의 약점(베이스라인 크기 불공정, 단독 ablation 없음, 수치 불일치)은 반복하지 않는다.
- 예비 실험: github.com/leoa4238/Researchai (JAAD, pose + SegFormer road relation, F1 0.7108 → 0.7545).
- 일정: 2026-10 기준 약 3개월. 실험 2개월 + 작성 1개월. LiDAR 기반 도로/보도 분할은 여유 있을 때만.

## 데이터

- AI-Hub 058 생활도로 객체인식 자율주행 데이터, 전방 카메라(image_F)만 사용.
- 카메라 라벨에 Road 클래스 없음 → SegFormer(Cityscapes)로 도로/보도 마스크 생성.
- 이용약관상 재배포 불가: 이미지, 라벨, 이를 학습한 가중치는 커밋하지 않는다(.gitignore 처리됨).
- 가공 데이터 백업은 사용자 Google Drive의 `rapd_backup_YYYYMMDD.zip` (풀면 `aihub_f/`).

## 실행 환경

두 가지 환경을 모두 지원해야 한다.

1. PC방 (Windows, 재부팅 시 C: 초기화, GPU RTX 4060 Ti 8GB): `aihub/setup_pcbang.ps1`로 매번 재설치.
   Python 미설치, winget/git 없음. 작업 위치 `C:\work`. 끝낼 때 GitHub 자격 증명 삭제를 꼭 안내.
2. 준비된 환경 (Python/PyTorch가 있는 PC나 서버): `pip install -r aihub/requirements.txt` 후 동일 스크립트 사용.

경로는 하드코딩하지 않고 인자로 받는다. `train_baseline.py`는 dataset.yaml의 path를 실제 위치로 자동 수정한다.

## 주의

- Windows 한글 경로: cv2.imread/imwrite 대신 `make_road_masks.imread/imwrite`(imdecode/imencode) 사용.
- SegFormer는 보행자 발 픽셀을 person으로 칠한다 → 도로 판정은 발 주변 바닥 창으로 한다.
- split은 clip(촬영 구간) 단위. 프레임 단위로 섞으면 성능이 부풀려진다.
- 사용자는 한국어로 소통한다.
