# Colab(무료)에서 session_v1 학습하기

무료 Colab(T4 16GB) 기준이다. 비교할 모델은 모두 같은 환경(Colab T4, Ultralytics 8.3.150)에서 학습한다.

준비: 런타임 → 런타임 유형 변경 → **T4 GPU**. Drive에 `rapd_backup_20261010.zip`(약 2GB, 풀면 `aihub_f/`)이 있어야 한다.
데이터가 든 Drive 폴더와 노트북은 다른 사람과 공유하지 않는다 (AI-Hub 재배포 금지).

## 셀 1. Drive 연결과 설정

```python
from google.colab import drive
drive.mount('/content/drive')

BACKUP_ZIP = '/content/drive/MyDrive/rapd_backup_20261010.zip'  # Drive 안의 실제 위치로 수정
DRIVE_OUT  = '/content/drive/MyDrive/rapd_runs'                # 학습 결과가 저장될 Drive 폴더
BRANCH     = 'analysis/failure-modes-session-split'
NAME       = 'yolov8s_session_v1_s0'
```

## 셀 2. 환경, 코드, 데이터 준비 (세션이 새로 시작될 때마다 실행)

```python
import os
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install ultralytics==8.3.150
if not os.path.exists('/content/rapd'):
    !git clone -q -b {BRANCH} https://github.com/leoa4238/road-aware-pedestrian-detection.git /content/rapd
if not os.path.exists('/content/data/aihub_f/images.csv'):
    # Drive에서 바로 읽으면 느리므로 로컬 디스크로 복사해서 푼다 (몇 분 걸림)
    !cp "{BACKUP_ZIP}" /content/backup.zip && unzip -q /content/backup.zip -d /content/data && rm /content/backup.zip

# 학습 결과 폴더를 Drive로 연결: 세션이 끊겨도 last.pt가 남아 이어서 학습할 수 있다
os.makedirs('/content/rapd/outputs/aihub', exist_ok=True)
for sub in ('runs', 'results'):
    os.makedirs(f'{DRIVE_OUT}/{sub}', exist_ok=True)
    path = f'/content/rapd/outputs/aihub/{sub}'
    if not os.path.islink(path):
        !rm -rf {path}
        os.symlink(f'{DRIVE_OUT}/{sub}', path)
print('ready')
```

## 셀 3. session_v1 split 재생성과 동일성 확인

```python
%cd /content/rapd
!python aihub/make_session_split.py --data /content/data/aihub_f --version session_v1 --report-dir /content/split_report --overwrite

import hashlib
expected = {'train': (3854, 'f3c7b99ef4a6ef13'), 'val': (830, 'cb4b445ac6a23e87'), 'test': (815, '24ff0ee707189eca')}
for split, want in expected.items():
    lines = open(f'/content/data/aihub_f/split_session_v1_{split}.txt').read().split()
    got = (len(lines), hashlib.sha256('\n'.join(sorted(lines)).encode()).hexdigest()[:16])
    assert got == want, (split, got, want)
print('로컬 PC에서 만든 session_v1과 동일')
```

## 셀 4. 학습 (끊기면 셀 2 → 셀 3 → 셀 4를 다시 실행하면 이어서 학습)

```python
%cd /content/rapd
resume = '--resume' if os.path.exists(f'{DRIVE_OUT}/runs/{NAME}/weights/last.pt') else ''
!python aihub/train_baseline.py --data /content/data/aihub_f/dataset_session_v1.yaml \
    --model yolov8s.pt --epochs 50 --seed 0 --batch 16 --workers 2 --name {NAME} {resume}
```

## 셀 5. 결과 확인

```python
import json
print(json.dumps(json.load(open(f'{DRIVE_OUT}/results/{NAME}.json')), indent=2, ensure_ascii=False))
```

## 무료 Colab 주의점

- 예상 시간: 50 epoch에 1~2시간.
- 브라우저 탭을 닫거나 오래 조작하지 않으면 연결이 끊긴다. 끊기면 위 순서대로 다시 실행한다. 결과는 Drive에 남는다.
- GPU 사용량 한도에 걸리면 하루 정도 GPU를 못 받을 수 있다. 여러 모델을 연달아 학습할 때 일정에 반영한다.
- 학습이 이미 끝난 상태에서 셀 4를 다시 실행하면 "이어서 할 것이 없다"는 오류가 날 수 있다. 셀 5로 결과만 확인하면 된다.
- 이 결과는 기존 split의 수치(mAP50 0.734)와 직접 비교하지 않는다. 이후 모든 모델의 기준선이 된다.
