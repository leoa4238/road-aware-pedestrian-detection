"""Compare checkpoints on the test split through one identical 4-channel validation path.

For each run: pedestrian AP50 (Ultralytics), and at a fixed confidence threshold the
pedestrian recall per original-pixel height bin, TP/FN/FP, plus how strongly the RGA
gates modulate features. `--zero-mask` re-evaluates road models with the road input
removed, to test whether the gates actually use the road mask.

Example:
    python aihub/diagnose_rga.py --data <DATA>/aihub_f/dataset_session_v1.yaml \
        --run baseline=D:/nonmoon/best.pt --run rga=D:/nonmoon/rga_best.pt --zero-mask \
        --output outputs/aihub/rga_diagnostics/s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_analysis_core import HEIGHT_EDGES, bin_value, pairwise_iou  # noqa: E402
from road_yolo import RoadDetectionModel, RoadValidator  # noqa: E402
from ultralytics.nn.tasks import attempt_load_one_weight  # noqa: E402
from ultralytics.utils import ops  # noqa: E402

PED = 0


class CollectingValidator(RoadValidator):
    """RoadValidator that also keeps per-image predictions and GT in native image pixels."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.records = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred, pbatch) if len(pred) else pred
            self.records.append({
                "gt_boxes": pbatch["bbox"].cpu().numpy(), "gt_cls": pbatch["cls"].cpu().numpy().astype(int),
                "pred": predn[:, :6].cpu().numpy() if len(pred) else np.zeros((0, 6)),
                "ori_shape": tuple(pbatch["ori_shape"])})


def as_road_model(model):
    """Wrap a stock DetectionModel so it accepts (and ignores) the 4th channel."""
    if isinstance(model, RoadDetectionModel):
        return model
    wrapped = RoadDetectionModel(model.yaml, nc=model.nc, verbose=False).add_road_module("none")
    wrapped.load_state_dict(model.state_dict())
    wrapped.names, wrapped.args, wrapped.stride = model.names, model.args, model.stride
    return wrapped


def fixed_threshold(records, conf, iou, original_width):
    """Class-aware greedy matching at one confidence; pedestrian counts by original-pixel height."""
    rows, fp = [], 0
    for r in records:
        scale = original_width / r["ori_shape"][1]
        pred = r["pred"][r["pred"][:, 4] >= conf]
        pred = pred[np.argsort(-pred[:, 4], kind="stable")]
        gt_boxes, gt_cls = r["gt_boxes"], r["gt_cls"]
        overlaps = pairwise_iou(gt_boxes, pred[:, :4]) if len(pred) and len(gt_boxes) else np.zeros((len(gt_boxes), len(pred)))
        used = set()
        for p in range(len(pred)):
            eligible = [g for g in range(len(gt_boxes)) if g not in used and gt_cls[g] == int(pred[p, 5])
                        and overlaps[g, p] >= iou]
            if eligible:
                used.add(max(eligible, key=lambda g: overlaps[g, p]))
            elif int(pred[p, 5]) == PED:
                fp += 1
        for g in np.flatnonzero(gt_cls == PED):
            height = (gt_boxes[g, 3] - gt_boxes[g, 1]) * scale
            rows.append((bin_value(height, HEIGHT_EDGES), g in used))
    bins = {}
    for name, hit in rows:
        bins.setdefault(name, []).append(hit)
    tp = sum(hit for _, hit in rows)
    return {"tp": int(tp), "fn": len(rows) - int(tp), "fp": fp,
            "recall": tp / len(rows) if rows else None,
            "recall_by_height": {k: {"gt": len(v), "recall": round(float(np.mean(v)), 4)}
                                 for k, v in sorted(bins.items())}}


def gate_strength(model, records_hint=None):
    gates = getattr(model, "road_gates", None)
    return None if gates is None else [round(float(g.gamma), 4) for g in gates]


def evaluate(name, weights, data, args, zero_mask=False):
    model, _ = attempt_load_one_weight(str(weights))
    model = as_road_model(model.float())
    if zero_mask:
        if model.road_mode not in ("rga", "rgab"):
            return None
        model._road_inputs = lambda mask, size, dtype: RoadDetectionModel._road_inputs(model, None, size, dtype)
    validator = CollectingValidator(args=dict(data=str(data), split="test", imgsz=640, batch=args.batch,
                                              workers=args.workers, plots=False, verbose=False,
                                              project=str(args.output.parent), name="val_" + name, exist_ok=True))
    validator(model=model)
    box = validator.metrics.box
    names = validator.metrics.names
    per_class = {names[int(c)]: round(float(ap), 4) for c, ap in zip(box.ap_class_index, box.ap50)}
    return {"weights": str(weights), "mode": model.road_mode, "zero_mask": zero_mask,
            "gamma": gate_strength(model), "mAP50": round(float(box.map50), 4),
            "pedestrian_AP50": per_class.get("pedestrian"), "per_class_AP50": per_class,
            "fixed_threshold": fixed_threshold(validator.records, args.conf, args.iou, args.original_width)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="name=path/to/best.pt")
    parser.add_argument("--zero-mask", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--original-width", type=float, default=1920.)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for item in args.run:
        name, path = item.split("=", 1)
        results[name] = evaluate(name, path, args.data, args)
        if args.zero_mask:
            zero = evaluate(name + "_zero_mask", path, args.data, args, zero_mask=True)
            if zero:
                results[name + "_zero_mask"] = zero
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, r in results.items():
        ft = r["fixed_threshold"]
        small = ft["recall_by_height"].get("[32,64)", {})
        print("%-16s pedAP50 %.4f  R %.4f  TP/FN/FP %d/%d/%d  R[32,64) %s  gamma %s" % (
            name, r["pedestrian_AP50"], ft["recall"], ft["tp"], ft["fn"], ft["fp"], small.get("recall"), r["gamma"]))


if __name__ == "__main__":
    main()
