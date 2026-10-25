"""여러 탐지 모델을 같은 평가 코드로 생활도로 test split에 적용해 클래스별 AP50을 비교한다.

목적: COCO로 사전학습한 일반 모델(생활도로 학습 없음)과 생활도로로 학습한 베이스라인의 차이를 보여
"일반 도로 모델은 생활도로에서 약하다"는 문제 제기 근거를 만든다.

- COCO 모델 클래스는 eval_common.COCO_TO_OURS로 대응시킨다 (person -> pedestrian, bus/truck -> truck_bus 등).
- PM(personal_mobility)은 COCO에 대응 클래스가 없으므로 공통 평균(mAP50_shared)에서 뺀다.
- 모든 모델을 같은 매칭/AP 코드로 평가하므로 ultralytics val의 수치와 소수점 단위로 다를 수 있다.

예시:
    python aihub/eval_zeroshot.py --data /content/data/aihub_f \
        --models yolov8s.pt yolo11s.pt /content/drive/MyDrive/rapd/runs/yolov8s_e50_s0/weights/best.pt \
        --out /content/drive/MyDrive/rapd/results/zeroshot.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval_common import CLASS_NAMES, ap50_per_class, load_gt, predict, split_images

SHARED = [c for c in CLASS_NAMES if c != "personal_mobility"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="prepare_dataset.py 출력 폴더")
    parser.add_argument("--models", nargs="+", required=True, help="가중치 경로 또는 ultralytics 모델 이름")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None, help="결과 CSV (기본: <data>/eval/zeroshot_<split>.csv)")
    args = parser.parse_args()

    gt = load_gt(args.data, args.split)
    images = split_images(args.data, args.split)
    print(f"[data] {args.split}: images={len(images)} objects={len(gt)}")

    rows = []
    for weights in args.models:
        preds = predict(weights, images, imgsz=args.imgsz, batch=args.batch)
        ap = ap50_per_class(preds, gt, CLASS_NAMES)
        shared = [ap[c] for c in SHARED]
        row = {"model": str(weights), **{f"AP50_{c}": round(v, 4) for c, v in ap.items()},
               "mAP50_shared": round(sum(shared) / len(shared), 4)}
        rows.append(row)
        print(row)

    table = pd.DataFrame(rows)
    out = args.out or args.data / "eval" / f"zeroshot_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
