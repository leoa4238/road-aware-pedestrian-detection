# Road-Aware Pedestrian Detection

생활도로 환경에서 주행가능영역(도로) 정보를 활용해 도로 위·도로 경계의 위험 보행자를 더 잘 탐지하는 모델 연구.

## 연구 질문

일반 객체 탐지 모델은 보도 위 보행자와 차도에 진입한 보행자를 같은 클래스로 취급한다.
자율주행 차량에 실제로 위험한 것은 도로 위·도로 경계의 보행자이므로, 도로 영역 정보를 탐지 특징에 직접 반영하면
위험 보행자 탐지 성능이 향상되는가?

## 데이터

AI-Hub `058. 생활도로 객체인식 자율주행 데이터` 전방 카메라(image_F)

| split(AI-Hub) | 이미지 | 보행자 포함 이미지 | 보행자 수 |
|---|---:|---:|---:|
| Training | 80,001 | 53,923 | 155,492 |
| Validation | 10,034 | 6,736 | 19,506 |

- 이미지 라벨에 Road 클래스가 없으므로 SegFormer(Cityscapes)로 도로/보도 마스크를 생성한다.
- 데이터는 AI-Hub 이용약관상 재배포 불가이므로 이 저장소에 포함하지 않는다.

## 파이프라인

```text
aihub/prepare_dataset.py   AI-Hub COCO 라벨 -> YOLO 형식, clip 단위 split, 보행자 발 위치 기록
aihub/make_road_masks.py   SegFormer 도로/보도 마스크, 보행자별 도로 위 여부와 도로까지 거리
aihub/train_baseline.py    YOLO 베이스라인 학습, test split 클래스별 mAP 저장
aihub/setup_pcbang.ps1     초기화되는 PC에서 Python/CUDA 환경 재설치
```

실행 순서:

```powershell
$py = "C:\work\venv\Scripts\python.exe"
$root = "<AI-Hub 다운로드 경로>\3.개방데이터\1.데이터\Validation"
& $py aihub\prepare_dataset.py --label-root "$root\02.라벨링데이터" --image-root "$root\01.원천데이터" --out C:\work\data\aihub_f
& $py aihub\make_road_masks.py --data C:\work\data\aihub_f
& $py aihub\train_baseline.py --data C:\work\data\aihub_f\dataset.yaml --model yolov8s.pt --epochs 50
```

## 관련 작업

- 예비 실험: [leoa4238/Researchai](https://github.com/leoa4238/Researchai) — JAAD에서 pose feature에 SegFormer 기반
  road relation feature를 결합했을 때 crossing 인식 F1이 0.7108 → 0.7545로 향상됨을 확인.
