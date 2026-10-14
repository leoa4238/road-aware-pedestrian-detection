# 실패 유형 분석과 세션 단위 재분할 (2026-10-14)

작업자: Claude Code. 이전 문서 `docs/PROGRESS_ERROR_ANALYSIS_20261014.md`(Codex)의 후속 작업이다.
기존 분석 결과, 예측 캐시, 원본 split, 이미지, 라벨은 수정하지 않았다. 새 탐지 모델 학습은 아직 하지 않았다.

## 1. 요약

1. **보행자 미탐 291개 중 44%(127개)는 모델이 정답 위치에 박스를 냈지만 점수가 0.25보다 낮았다.**
   점수 중앙값은 0.074였다. 박스 후보가 아예 없던 경우는 26%(77개)다.
2. **임계값을 낮추는 것으로는 해결되지 않는다.** 0.1로 낮추면 미탐 81개를 되찾지만, 0.1~0.25 구간의 보행자 박스가 545개 새로 나온다.
   즉 모델이 진짜 보행자 후보와 배경을 점수로 구분하지 못하는 문제다.
3. **보행자 오탐 377개 중 52%(195개)는 어떤 정답과도 겹치지 않는 배경 오탐**이고, 그중 172개가 128px 미만의 작은 박스다.
   배경 오탐의 68%(132개)는 추정 마스크상 도로 위에 있다. 따라서 "도로 위면 점수를 올린다"는 단순한 규칙은 오탐도 함께 늘린다.
4. **세션 단위 새 split `session_v1`을 만들었다.** 세션 교차 0개, 파일명 충돌 이미지 2행은 제외했다. 기존 split은 그대로 남아 있다.

논문에 쓸 수 있는 동기 문장(초안):

> 기존 탐지기가 놓친 보행자의 44%는 후보 박스가 생성되었으나 낮은 점수로 제거된 경우였고, 임계값을 낮추면 미탐 1개를 회복할 때마다 약 6개의 저점수 박스가 추가되었다. 이는 후보 생성보다 후보와 배경의 구분이 병목임을 뜻하며, 본 연구는 도로 맥락을 이용해 이 구분을 돕는 모듈을 제안한다.

주의: 마지막 문장은 가설이다. 도로 맥락이 실제로 이 구분을 돕는지는 RGA 비교 실험으로 확인해야 한다.

## 2. 미탐(FN) 유형

조건: 기존 분석과 같은 가중치, confidence 0.25, 매칭 IoU 0.5. 저점수 판정은 별도 추론(confidence 0.001)으로 했다.
별도 추론의 0.25 이상 보행자 박스 수(1,434개)가 기존 캐시와 정확히 같아 추론 조건이 일치함을 확인했다.

| 유형 | 정의 | 개수 | 비율 |
|---|---|---:|---:|
| below_threshold | IoU 0.5 이상 보행자 박스가 0.001~0.25 점수로 존재 | 127 | 43.6% |
| no_candidate | 0.001에서도 보행자 박스 없음 | 77 | 26.5% |
| localization | 매칭 안 된 보행자 박스가 IoU 0.1~0.5로 겹침 | 48 | 16.5% |
| merged_with_neighbor | 겹치는 보행자 박스가 이웃 보행자에 매칭됨 (군중) | 28 | 9.6% |
| wrong_class | 다른 클래스 박스가 IoU 0.5 이상 | 11 | 3.8% |

크기별 (원본 bbox 높이):

| 높이 | below_threshold | no_candidate | localization | merged | wrong_class |
|---|---:|---:|---:|---:|---:|
| 32px 미만 | 3 | 9 | 0 | 0 | 0 |
| 32~64px | 60 | 44 | 7 | 2 | 2 |
| 64~128px | 50 | 18 | 22 | 15 | 5 |
| 128~256px | 10 | 3 | 9 | 6 | 3 |
| 256px 이상 | 4 | 3 | 10 | 5 | 1 |

- 작은 보행자(32~64px)의 미탐은 저점수(60)와 후보 없음(44)이 대부분이다.
- 큰 보행자의 미탐은 위치 오차와 군중 병합이 주를 이룬다. 제안 모듈의 대상이 아니다.

## 3. 오탐(FP) 유형

| 유형 | 정의 | 개수 |
|---|---|---:|
| background | 어떤 정답과도 의미 있게 겹치지 않음 | 195 |
| localization | 보행자 정답과 겹치지만 매칭 기준 미달 | 86 |
| class_confusion | 다른 클래스 정답과 IoU 0.5 이상 (PM, 이륜차 탑승자 등) | 48 |
| duplicate | 이미 매칭된 보행자 정답과 IoU 0.5 이상 | 48 |

추정 발 위치 영역별:

| 영역 | background | localization | class_confusion | duplicate |
|---|---:|---:|---:|---:|
| road | 132 | 54 | 44 | 36 |
| sidewalk | 36 | 23 | 3 | 10 |
| other | 27 | 9 | 1 | 2 |

- 배경 오탐 점수 사분위: 0.31 / 0.41 / 0.54. 0.6 이상은 35개다.
- 점수가 높은 배경 오탐 일부는 **라벨이 빠진 실제 사람**일 수 있다. 상위 40개 목록을 `fp_background_review.csv`에 두었다. 사람이 이미지를 확인해야 한다.
- class_confusion 48개는 대부분 128px 이상의 큰 박스다. PM이나 이륜차 탑승자를 보행자로도 낸 경우로 보이며, 라벨링 기준(탑승자를 보행자로 보지 않음)과 관련된다.

## 4. 영역별 보행자 재현율

| 추정 발 위치 영역 | 전체 GT | 재현율 | 32~64px GT | 32~64px 재현율 |
|---|---:|---:|---:|---:|
| road | 946 | 0.864 | 102 | 0.529 |
| sidewalk | 226 | 0.735 | 24 | 0.292 |
| other | 176 | 0.421 | 68 | 0.265 |

- 마스크가 도로나 보도가 아니라고 판단한 위치(other)의 보행자가 가장 많이 놓쳐진다. 작은 보행자가 많고(68/176), 차량 뒤에 가려졌거나 마스크가 틀렸을 가능성이 있다.
- 이 영역 구분은 추정 마스크 기반이다. 150개 수동 검수(이전 문서 8장 우선 1)가 끝나야 해석할 수 있다.

## 5. 세션 단위 재분할 `session_v1`

세션 = clip 이름에서 `(n)`을 뗀 촬영 시작 시각. 같은 세션의 clip은 한 주행의 연속 구간이므로 같은 split에 둔다.
(time, weather) 조합별로 세션을 70/15/15에 맞춰 배정하고, 500개 시드 중 이미지·보행자·우천·안개·야간 비율 편차가 가장 작은 시드(4)를 골랐다.

| split | 세션 | 이미지 | 보행자 |
|---|---:|---:|---:|
| train | 158 | 3,854 | 7,478 |
| val | 63 | 830 | 1,367 |
| test | 60 | 815 | 1,492 |

- 세션 교차: 0 (기존 split은 99개 세션이 교차)
- 제외: `CK_A01_R03_erh_rainy_01023394_F` (서로 다른 clip의 같은 파일명, 디스크에 1장만 존재) 메타데이터 2행
- 야간 이미지: train 228 / val 49 / test 49. **이 데이터의 야간 세션 9개는 모두 안개**라 야간과 안개의 효과는 분리할 수 없다. 분리하려면 데이터를 추가로 받아야 한다.
- Ultralytics 8.3.150으로 세 split을 모두 로드해 라벨 누락 0개를 확인했다 (train 3,854장 33,386개, val 830장 6,683개, test 815장 6,654개 객체).

참고: Ultralytics는 라벨 캐시를 첫 라벨 파일의 폴더 이름으로 저장한다. 새 split은 세 개 모두 `labels/test.cache`를 쓰게 되어 매번 다시 스캔한다(약 5초). 결과에는 영향이 없다. 이 확인 과정에서 데이터 폴더의 `labels/test.cache`가 다시 만들어졌다.

## 6. 파일 위치

| 용도 | 경로 |
|---|---|
| 실패 유형 분석 코드 | `aihub/analyze_failure_modes.py` |
| 재분할 코드 | `aihub/make_session_split.py` |
| 단위 테스트 (신규 3개, 전체 27개 통과) | `tests/test_failure_modes_and_split.py` |
| 실패 유형 요약 | `outputs/aihub/failure_modes/failure_modes_summary.json` |
| 미탐별 유형 | `outputs/aihub/failure_modes/pedestrian_fn_modes.csv` |
| 오탐별 유형 | `outputs/aihub/failure_modes/pedestrian_fp_modes.csv` |
| 검토용 고점수 배경 오탐 40개 | `outputs/aihub/failure_modes/fp_background_review.csv` |
| 저점수 추론 캐시 | `outputs/aihub/failure_modes/low_conf_cache/` |
| 새 split 요약 | `outputs/aihub/splits/session_v1/split_summary.json` |
| 새 split 이미지별 배정 | `outputs/aihub/splits/session_v1/split_manifest.csv` |
| 학습용 yaml (데이터 폴더 안) | `data/analysis_backup_20261013/dataset/aihub_f/dataset_session_v1.yaml` |
| 학습용 이미지 목록 | 같은 폴더의 `split_session_v1_{train,val,test}.txt` |

`.gitignore`에 `outputs/aihub/failure_modes/`, `outputs/aihub/splits/`를 추가했다. 라벨 좌표가 들어 있기 때문이다. 커밋은 하지 않았다.

## 7. 다음 작업

1. **(사용자) 마스크 150개 검수**와 `fp_background_review.csv` 상위 40개 육안 확인. 라벨 누락 비율을 알면 배경 오탐 수치를 어떻게 해석할지 정해진다.
2. **(사용자 확인 후) `session_v1`로 baseline 재학습.** 기존 split의 수치(mAP50 0.734)와 직접 비교하지 않고, 이후 모든 모델을 이 split에서 같은 조건으로 학습한다. 4GB GPU(GTX 1650 SUPER)에서는 batch를 줄여야 하므로, 가능하면 PC방(RTX 4060 Ti) 또는 서버에서 한다.

   ```powershell
   python aihub\train_baseline.py --data <DATA>\aihub_f\dataset_session_v1.yaml --model yolov8s.pt --epochs 50 --seed 0 --name yolov8s_session_v1_s0
   ```

   주의: `train_baseline.py`는 yaml의 `path:` 줄을 실제 위치로 고쳐 쓴다. 의도된 동작이다.
3. **RGA 설계 기준.** 이번 결과로 정한 목표는 "작은 보행자 후보와 배경 오탐의 점수 구분"이다. 비교는 이전 문서 8장 우선 3의 설계(기존 YOLO / 단순 road 입력 / 같은 크기 일반 attention / residual RGA / RGA+경계)를 따른다. 추가로 볼 지표는 다음과 같다.
   - 32~64px 보행자 재현율
   - below_threshold 미탐이 몇 개 회복됐는지
   - background 오탐 수
4. 새 split에서 baseline을 다시 학습하면 이 실패 유형 분석을 새 test로 한 번 더 돌려 수치를 갱신한다. 지금 수치는 기존 split의 test 기준이다.
