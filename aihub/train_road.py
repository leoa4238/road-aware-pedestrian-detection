"""Train a road-module variant (see road_yolo.MODES) and save test metrics like train_baseline.py.

Uses the same defaults as train_baseline.py (yolov8s.pt, 50 epochs, 640, batch 16, deterministic)
so that only the road module differs from the baseline.

Example:
    python aihub/train_road.py --data <DATA>/aihub_f/dataset_session_v1.yaml --mode rga --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # pickled checkpoints reference module "road_yolo"

from road_yolo import MODES, RoadTrainer, RoadValidator  # noqa: E402
from ultralytics.nn.tasks import attempt_load_one_weight  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def evaluate(weights, data, imgsz, batch, mask_close, project, name, workers):
    model, _ = attempt_load_one_weight(str(weights))
    validator = RoadValidator(args=dict(data=str(data), split="test", imgsz=imgsz, batch=batch, workers=workers,
                                        project=str(project), name=name, exist_ok=True, plots=False))
    validator.mask_close = mask_close
    validator(model=model)
    box = validator.metrics.box
    names = validator.metrics.names
    return {"mAP50": float(box.map50), "mAP50_95": float(box.map),
            "per_class_mAP50": {names[int(c)]: float(ap) for c, ap in zip(box.ap_class_index, box.ap50)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mask-close", type=int, default=0,
                        help="close road-mask holes with this kernel (px, mask scale); control for object hints")
    parser.add_argument("--fraction", type=float, default=1.0, help="train on a fraction (smoke tests only)")
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="skip training, evaluate existing best.pt")
    args = parser.parse_args()

    # Same path fix as train_baseline.py so the yaml works wherever the backup was unpacked.
    lines = args.data.read_text(encoding="utf-8").splitlines()
    lines = [f"path: {args.data.resolve().parent.as_posix()}" if l.startswith("path:") else l for l in lines]
    args.data.write_text("\n".join(lines) + "\n", encoding="utf-8")

    suffix = "_close%d" % args.mask_close if args.mask_close else ""
    name = args.name or f"{Path(args.model).stem}_{args.mode}{suffix}_e{args.epochs}_s{args.seed}"
    project = REPO / "outputs" / "aihub" / "runs"
    RoadTrainer.road_mode = args.mode
    RoadTrainer.mask_close = args.mask_close
    last = project / name / "weights" / "last.pt"
    if not args.eval_only:
        if args.resume and last.exists():
            RoadTrainer(overrides=dict(model=str(last), resume=str(last))).train()
        else:
            RoadTrainer(overrides=dict(model=args.model, data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
                                       batch=args.batch, seed=args.seed, deterministic=True, workers=args.workers,
                                       fraction=args.fraction, project=str(project), name=name,
                                       exist_ok=True)).train()

    metrics = evaluate(project / name / "weights" / "best.pt", args.data, args.imgsz, args.batch,
                       args.mask_close, project, name + "_test", args.workers)
    result = {"name": name, "model": args.model, "mode": args.mode, "mask_close": args.mask_close,
              "epochs": args.epochs, "imgsz": args.imgsz, "seed": args.seed, "fraction": args.fraction, **metrics}
    out = REPO / "outputs" / "aihub" / "results" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
