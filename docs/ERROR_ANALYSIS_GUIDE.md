# 1차 개발: 기존 보행자 탐지 실패 분석

이번 코드는 **RGA 구현이나 새 탐지기 학습이 아니다**. 기존 최종 가중치와 고정 test 분할에서 실패 조건을 확인한다. 원본 ZIP, dataset.yaml, 이미지, 라벨, 기존 모델 코드는 수정하지 않는다.

## 생성한 코드

- `aihub/extract_analysis_backup.py`: 안전한 압축 해제, 덮어쓰기 금지, 원본 ZIP SHA256 기록.
- `aihub/analysis_sanity.py`: 기존 `model.val(split="test")` 성능 재현 확인.
- `aihub/analyze_detection_errors.py`: 실행 진입점, 재현 확인 게이트, 추론 캐시, 매칭, 조건별 통계, 시각화.
- `aihub/error_analysis_core.py`: 좌표 변환, 일대일 매칭, 부호 거리, 그룹 신뢰구간.
- `aihub/analysis_dataset_audit.py`: clip/주행 세션 중복, 파일명 충돌, 메타데이터·YOLO 라벨 비교.
- `aihub/analysis_statistics.py`: 크기 층화 비교와 FN 설명용 정규화 로지스틱 회귀.
- `aihub/analysis_visuals.py`: FN/FP, 수동 마스크 검수 자료, 300dpi 그림.
- `aihub/verify_error_analysis.py`: 집계 보존, 일대일 매칭, 캐시 수, 이미지 디코딩, 원본 ZIP·기존 소스 보존 확인.
- `tests/test_error_analysis.py`: 실제 데이터·가중치가 없는 단위 테스트.

2026-10-13 실제 검증: 단위 테스트 24개 통과. test 809장 / GT 7274개 / 예측 7766개의 중복·누락 집계를 독립 점검했고, 생성 이미지180개가 모두 디코딩 가능하며 비어 있지 않음을 확인했다. 원본 ZIP 두 개의 SHA256과 기존 소스7개의 내용도 보존됐다.

기존 파일의 변경은 `aihub/requirements.txt`의 직접 의존성 `scipy` 추가와 `.gitignore`의 비공개 분석 산출물·Ultralytics 설정 제외뿐이다. 기존 학습 및 전처리 코드는 그대로다. 패키지 설치를 기존 가상환경에 강제로 적용하지 않았다.

## 현재 PC에서 재실행

PowerShell:

```powershell
$repo = 'C:\Users\LENOVO\Documents\Codex\2026-05-06\d-reserch\road-aware-pedestrian-detection'
$py = 'D:\reserch\jaywalking-risk-recognition\.venv\Scripts\python.exe'
$data = Join-Path $repo 'data\analysis_backup_20261013\dataset\aihub_f'
$weights = Join-Path $repo 'data\analysis_backup_20261013\weights\yolov8s_baseline_best.pt'
$out = Join-Path $repo 'outputs\aihub\error_analysis'
$packages = Join-Path $repo 'data\reproduction_packages'

& $py (Join-Path $repo 'aihub\analyze_detection_errors.py') `
  --data $data --weights $weights --output $out `
  --expected (Join-Path $repo 'outputs\aihub\results\yolov8s_e50_s0.json') `
  --sanity-dir (Join-Path $out 'reproduction_original') `
  --package-dir $packages --stage analyze --batch 2 --device 0 --resume

& $py -m unittest discover -s (Join-Path $repo 'tests') -v
```

`--resume`는 동일 가중치·입력 이미지 해시·추론 설정에서만 기존 이미지별 예측을 재사용한다. confidence/IoU/배치/가중치가 달라지면 새 출력 폴더를 사용한다. 통계와 그림은 캐시에서 다시 생성할 수 있다. `--half`는 기본 False다. `--batch`는 4GB GPU 기준 기본 2이며 다른 GPU에서는 조정할 수 있다. 배치는 추론 시 사용되고 매칭/통계는 CPU에서 수행된다.

## 새 환경에서의 절차

1. 원본 ZIP을 새 압축 해제 폴더에 풀어 보존한다. 이미 있는 파일에는 덮어쓰지 않는다.

```powershell
& $py (Join-Path $repo 'aihub\extract_analysis_backup.py') `
  --data-zip 'D:\nonmoon\rapd_backup_20261010.zip' `
  --weights-zip 'D:\nonmoon\rapd_weights_20261010.zip' `
  --destination '<새 압축 해제 위치의 절대 경로>'
```

2. 기존 재현 버전이 없는 환경에서는 기존 환경을 교체하지 않고 별도 경로에 설치할 수 있다.

```powershell
& $py -m pip install --no-deps --target $packages ultralytics==8.3.150
```

3. 재현 확인부터 실행한다. 기존 체크포인트/원래 스크립트의 batch는 16이었다. 이 PC의 4GB GPU에서 FP32 평가가 실제로 완료됐지만 다른 환경의 메모리 여유는 보장하지 않는다.

```powershell
& $py (Join-Path $repo 'aihub\analyze_detection_errors.py') `
  --data $data --weights $weights --output '<새 결과 폴더>' `
  --package-dir $packages --stage all --batch 2 --sanity-batch 16 --device 0
```

Linux에서도 같은 인자를 사용하되 실행 파일과 경로를 해당 환경으로 바꾼다. Python 3.9 호환이며 실행 경로는 CLI로 전달한다. 기본 test 분할을 바꾸는 옵션은 없다. 기존 YAML의 예전 PC 경로는 출력 폴더의 별도 평가 YAML에만 덮어쓴다.

## 재현 결과의 해석

- 원래 기록: 전체 mAP50 `0.7339355554`, 보행자 `0.7893284284`.
- 현재 8.4.42 / batch2의 첫 실행: 전체 `0.7310362900`, 보행자 `0.7862426896`. 허용 오차를 넘어 상세 분석이 차단됐다.
- 원래 8.3.150 / batch16 조건: 전체 `0.7339702622`, 보행자 `0.7893734621`. 사전에 고정한 절대 허용 오차 `0.00005` 안에서 재현됐다. 완전히 동일한 실수 값은 아니다.
- 라이브러리와 batch를 모두 바꿨으므로 첫 차이를 라이브러리 하나만의 원인이라고 확정하지 않는다.
- 원래 환경과 현재 GPU가 다르며 전체 5개 클래스의 개별 값까지 일치했다고 주장하지 않는다. 게이트의 판정 대상은 요청된 전체·보행자 mAP50 두 값이다.
- 상세 분석은 통과한 재현 버전으로 수행한다. mAP의 confidence 적분 지표와 confidence 0.25에서의 P/R/F1을 혼동하지 않는다.

## 통계와 거리 정의

- 매칭: 예측 confidence 내림차순, 같은 클래스의 미매칭 GT 중 가장 높은 IoU. GT/예측 각각 한 번만 사용한다. 중복 예측은 FP다.
- 오분류 진단: 미매칭 GT와 미매칭 다른 클래스 예측을 IoU 0.5 이상에서 별도 일대일 연결한다. FN/FP를 TP로 재분류하지 않는다. `no_correct_class_match`는 기준을 만족하는 정탐이 없다는 뜻이며 아예 아무 box도 없었다는 뜻은 아니다. CSV의 최대 IoU를 함께 확인한다.
- 좌표: GT는 원본 CSV의 좌상단 x/y+w/h. YOLO box는 실제 디코딩된 저장 이미지 크기에서 원본 폭/높이로 복원한다. 비율이나 1280px를 일괄 가정하지 않는다.
- 정답이 빈 이미지의 원본 해상도는 `--empty-original-size W H`로 지정할 수 있다. 미지정이고 전체 원본 메타데이터 해상도가 하나면 그 크기를 추론해 `coordinate_status`에 기록한다. 그렇지 않으면 저장 이미지 좌표임을 명시한다.
- 기존 `road_dist_norm`은 바닥 창의 가장 가까운 road 픽셀 거리/bbox 높이이며 경계 거리가 아니다.
- 새 부호 거리: nearest opposite road/non-road 픽셀 중심까지의 Euclidean Distance Transform. road 안 −, 밖 +. 도로 전체/도로 없음처럼 내부 경계가 없는 마스크는 NaN이다. bbox 높이와 마스크 높이로 각각 정규화한다.
- 경계 분석의 첫 정의는 GT instance foot 픽셀에서 계산한다. 마스크가 사람을 other로 칠하면 person hole도 경계에 포함된다. nearest road/sidewalk 바닥 지지점 분석은 이를 점검하는 민감도 분석이지 실제 연석 복원이 아니다.
- 크기 구간: 원본 bbox 높이 `[0,32), [32,64), [64,128), [128,256), [256,inf)` px.
- 경계 거리 구간: abs(distance)/bbox 높이 `[0,.25), [.25,.5), [.5,1), [1,2), [2,inf)`; 가설 비교는 `<=.25` 대 `>.25`. 경계값 0.25의 소속은 구간표와 이진 비교에서 다름을 명시한다.
- 이미지 높이 기준 구간 `[0,.01), [.01,.025), [.025,.05), [.05,.1), [.1,inf)`도 별도 제공한다.
- Recall 95% CI는 clip bootstrap 2000회, seed42. 세션 CI를 민감도 열로 함께 제공한다. GT30 미만 또는 cluster 부족은 표시한다. GT0은 Recall0이 아니라 n/a다. percentile bootstrap에서 관측 성공/실패가 한 종류뿐이면 `[0,0]`/`[1,1]`이 나올 수 있으나 모집단의 불확실성이 없다는 의미는 아니다.
- 로지스틱 회귀: FN을 종속변수로 거리·높이·야간·날씨 설명 변수, L2 C=1, 표준화, clip/세션 bootstrap 각 200회. test의 설명적 탐색 분석이며 검증된 예측 모델이나 인과 분석이 아니다. 공선성/행렬 rank를 JSON에 보고한다.

## 중요한 데이터 제한과 수정 제안

- 하위 clip의 split 중복은 없지만 상위 주행 세션 99개가 여러 split에 걸친다. 같은 주행 세션의 상관이 학습/시험 양쪽에 남으며 새 주행 세션 일반화로 해석할 수 없다.
- train의 서로 다른 두 clip에 `CK_A01_R03_erh_rainy_01023394_F`라는 같은 파일명이 있다. images.csv는 train3894행이지만 실제 train 이미지는3893장이다. 객체 키도20행 중복이고 TXT는10개 객체로, 두 메타데이터 묶음 중 어느 이미지가 저장됐는지는 원본 ZIP/JSON 검수 필요하다.
- 이 충돌 이미지의 좌표 비교는 객체 수 불일치로 별도 기록한다. 다른 비교 가능한 라벨의 0.01px 초과 좌표 차이는0개다. 이 두 사실을 혼동하지 않는다.
- 데이터/분할을 자동 수정하지 않았다. 다음 데이터 버전에서는 파일 키를 `(주행 세션, clip, 원본 파일명)`으로 만들고 주행 세션 단위 split을 먼저 확정한 뒤 다시 구축/학습하는 것을 권한다. 현재 버전과 별도 결과로 관리해야 한다.
- 원본 가림 주석이 없어 가림별 분석 불가다. GT가 없는 R06, GT15개뿐인 night/foggy는 일반화 결론을 내릴 수 없다.
- `foot_zone`는 SegFormer 추정값이다. 150개 검수 샘플의 정답 칸은 비워 두었으며 분할 정확도는 측정하지 않았다.
- test 조건을 보고 RGA를 설계한다면 이 test는 개발 분석 자료가 된다. 최종 주장은 새 세션 단위 홀드아웃이나 독립 자료에서 확인해야 한다.

## 주요 산출물

`outputs/aihub/error_analysis/`:

- `summary.json`, `analysis_report.md`: 실제 결과와 가설 판단.
- `detections_matched.csv`: GT/예측 일대일 매칭과 좌표, FN 진단.
- `pedestrian_failure_analysis.csv`: 조건별 GT/TP/FN/Recall와 clip·세션 CI.
- `road_boundary_analysis.csv`, `boundary_size_stratified_contrasts.csv`: 경계 거리와 크기 층화.
- `logistic_failure_analysis.json`: 조정된 탐색 분석과 공선성 진단.
- `dataset_audit.json`, `duplicate_*`, `session_overlap_images.csv`: 데이터 정합성 문제.
- `mask_review_sheet.csv`, `mask_review/`: 사람이 작성할 분할 검수 시트와 오버레이.
- `failure_examples/`: seed42로 고른 FN/FP 예시. 사례 빈도나 대표성을 입증하지 않는다.
- `figures/`: 조건별 Recall, 300dpi PNG 및 PDF.
- `prediction_cache/`, `analysis_manifest.json`: 예측 재시작과 추론 설정·소스 해시.

분석 자료에는 원본 객체 좌표가 있으므로 폴더 전체를 Git에서 제외했다. 데이터/라벨/가중치를 커밋하거나 업로드하지 않았다.

## RGA의 다음 단계

1. 검수 시트로 road/sidewalk/other와 person hole 경계를 확인한다.
2. 작은 객체·우천 미탐을 우선 설계 근거로 검토한다. 현재 거리 정의에서 경계 근처 미탐 증가 가설은 지지되지 않았다.
3. 입력 prior가 틀려도 원래 특징이 유지되는 residual RGA를 후보로 둔다. road/sidewalk/unknown, 부호·절대 거리, 신뢰도를 분리해 ablation한다.
4. 새 세션 단위 평가 버전에서 YOLO, 단순 road 입력, 동일 용량 일반 attention, RGA, RGA+boundary를 동일 학습 조건/시드로 비교한다.
5. 보행자 AP/Recall뿐 아니라 FP, 작은 객체, R03, 우천, 마스크 오류 조건, 연산량·FPS를 함께 본다. 성능 향상은 아직 주장하지 않는다.
