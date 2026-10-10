"""YOLO 베이스라인 학습 후 test split 클래스별 성능을 outputs/aihub/에 저장한다.

예시:
    python aihub/train_baseline.py --data C:/work/data/aihub_f/dataset.yaml --model yolov8s.pt --epochs 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="weights/last.pt가 있으면 이어서 학습")
    args = parser.parse_args()

    # 백업 zip을 다른 PC/경로에 풀어도 동작하도록 dataset.yaml의 path를 실제 위치로 맞춘다
    lines = args.data.read_text(encoding="utf-8").splitlines()
    lines = [f"path: {args.data.resolve().parent.as_posix()}" if l.startswith("path:") else l for l in lines]
    args.data.write_text("\n".join(lines) + "\n", encoding="utf-8")

    name = args.name or f"{Path(args.model).stem}_e{args.epochs}_s{args.seed}"
    project = REPO / "outputs" / "aihub" / "runs"
    last = project / name / "weights" / "last.pt"
    if args.resume and last.exists():
        # 중단된 학습을 같은 설정(epochs, optimizer 상태)으로 이어서 진행한다
        YOLO(last).train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=args.seed,
                    deterministic=True, workers=args.workers, project=str(project), name=name, exist_ok=True)

    best = YOLO(project / name / "weights" / "best.pt")
    metrics = best.val(data=str(args.data), split="test", imgsz=args.imgsz, batch=args.batch,
                       project=str(project), name=f"{name}_test", exist_ok=True)
    result = {
        "name": name, "model": args.model, "epochs": args.epochs, "imgsz": args.imgsz, "seed": args.seed,
        "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
        "per_class_mAP50": {metrics.names[int(c)]: float(ap) for c, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)},
    }
    out = REPO / "outputs" / "aihub" / "results" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
