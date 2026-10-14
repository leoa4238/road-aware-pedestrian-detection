"""Pedestrian FN/FP failure-mode breakdown on top of the existing error analysis.

Reads `detections_matched.csv` (conf 0.25 run) and, optionally, runs a separate
low-confidence inference pass so a missed pedestrian can be split into
"scored below threshold" vs "no candidate box at all". The original analysis
outputs and prediction cache are never modified.

Example:
    python aihub/analyze_failure_modes.py \
        --matched outputs/aihub/error_analysis/detections_matched.csv \
        --data data/analysis_backup_20261013/dataset/aihub_f \
        --weights data/analysis_backup_20261013/weights/yolov8s_baseline_best.pt \
        --output outputs/aihub/failure_modes --low-conf 0.001 --device 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from error_analysis_core import pairwise_iou, read_image, rescale_xyxy, xywh_to_xyxy

PED = "pedestrian"
LOC_IOU = .1          # below the 0.5 match IoU but clearly the same object
SMALL = "[32,64)"


def gt_boxes(frame):
    return xywh_to_xyxy(frame[["x", "y", "w", "h"]].to_numpy(float))


def pred_boxes(frame):
    return frame[["prediction_x1_original", "prediction_y1_original",
                  "prediction_x2_original", "prediction_y2_original"]].to_numpy(float)


def classify_fn(gt, preds, low):
    """One label per missed pedestrian, checked in order of how 'close' the detector came."""
    gt = gt.copy()
    gt["fn_mode"] = ""
    gt["best_ped_iou_any"] = np.nan
    gt["best_ped_iou_fp"] = np.nan
    gt["low_conf_score"] = np.nan
    for image, rows in gt.groupby("image"):
        p = preds[preds["image"] == image]
        ped = p[p["pred_class"] == PED]
        boxes = gt_boxes(rows)
        any_iou = pairwise_iou(boxes, pred_boxes(ped)) if len(ped) else np.zeros((len(rows), 0))
        fp = ped[ped["status"] == "FP"]
        fp_iou = pairwise_iou(boxes, pred_boxes(fp)) if len(fp) else np.zeros((len(rows), 0))
        gt.loc[rows.index, "best_ped_iou_any"] = any_iou.max(axis=1) if any_iou.size else 0.
        gt.loc[rows.index, "best_ped_iou_fp"] = fp_iou.max(axis=1) if fp_iou.size else 0.
        if low is not None:
            lp = low.get(image)
            if lp is not None and len(lp["boxes"]):
                keep = lp["classes"] == 0
                iou = pairwise_iou(boxes, lp["boxes"][keep])
                scores = lp["scores"][keep]
                for k, index in enumerate(rows.index):
                    hits = scores[iou[k] >= .5]
                    gt.loc[index, "low_conf_score"] = hits.max() if len(hits) else np.nan
    for index, row in gt.iterrows():
        if isinstance(row.get("confused_prediction_class"), str) and row["confused_prediction_class"]:
            mode = "wrong_class"
        elif row["best_ped_iou_fp"] >= LOC_IOU:
            mode = "localization"            # an unmatched pedestrian box sits on it with IoU < 0.5
        elif row["best_ped_iou_any"] >= LOC_IOU:
            mode = "merged_with_neighbor"    # its pedestrian box was taken by a nearby GT
        elif np.isfinite(row["low_conf_score"]):
            mode = "below_threshold"         # a correct box exists, but scored < conf threshold
        else:
            mode = "no_candidate" if low is not None else "no_candidate_or_low_score"
        gt.loc[index, "fn_mode"] = mode
    return gt


def classify_fp(fp, gts):
    fp = fp.copy()
    fp["fp_mode"] = ""
    fp["best_gt_iou"] = np.nan
    fp["best_gt_class"] = ""
    for image, rows in fp.groupby("image"):
        g = gts[gts["image"] == image]
        if g.empty:
            fp.loc[rows.index, "fp_mode"] = "background"
            continue
        iou = pairwise_iou(pred_boxes(rows), gt_boxes(g))
        for k, index in enumerate(rows.index):
            ped = (g["cls"] == PED).to_numpy()
            ped_iou = iou[k][ped].max() if ped.any() else 0.
            other_iou = iou[k][~ped].max() if (~ped).any() else 0.
            j = int(np.argmax(iou[k]))
            fp.loc[index, "best_gt_iou"] = iou[k, j]
            fp.loc[index, "best_gt_class"] = g["cls"].iloc[j]
            if ped_iou >= .5:
                tp_hit = g[ped].iloc[int(np.argmax(iou[k][ped]))]["status"] == "TP"
                mode = "duplicate" if tp_hit else "localization"
            elif other_iou >= .5:
                mode = "class_confusion"     # e.g. rider of a PM / two-wheeler labelled as vehicle
            elif ped_iou >= LOC_IOU:
                mode = "localization"
            else:
                mode = "background"          # includes possibly unlabeled people
            fp.loc[index, "fp_mode"] = mode
    return fp


def run_low_conf(args, images):
    from ultralytics import YOLO
    cache = args.output / "low_conf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    if model.names[0] != PED:
        raise ValueError("Class 0 must be pedestrian")
    sizes = images.set_index("image")[["img_w", "img_h"]]
    results = {}
    for offset, image in enumerate(sizes.index):
        path = cache / (image + ".npz")
        if not path.exists():
            decoded = read_image(args.data / "images" / "test" / (image + ".jpg"))
            if decoded is None:
                raise ValueError("Undecodable test image: " + image)
            r = model.predict(source=decoded, imgsz=640, conf=args.low_conf, iou=.7, max_det=300,
                              device=args.device, verbose=False, save=False)[0]
            h, w = r.orig_shape
            boxes = rescale_xyxy(r.boxes.xyxy.cpu().numpy(), (w, h), tuple(sizes.loc[image]))
            np.savez(path, boxes=boxes, scores=r.boxes.conf.cpu().numpy(),
                     classes=r.boxes.cls.cpu().numpy().astype(int))
        saved = np.load(path)
        results[image] = {k: saved[k] for k in ("boxes", "scores", "classes")}
        if offset % 100 == 0:
            print("[low-conf] %d/%d" % (offset, len(sizes)), flush=True)
    return results


def share(frame, column, by=None):
    if by is None:
        counts = frame[column].value_counts()
        return pd.DataFrame({"count": counts, "share": (counts / counts.sum()).round(4)})
    table = pd.crosstab(frame[by], frame[column], margins=True)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-conf", type=float, default=None,
                        help="extra inference pass at this confidence; omit to skip")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.matched)
    gts = d[d["record_type"] == "gt"]
    preds = d[d["record_type"] == "prediction"]
    # TP predictions are stored on the GT row; rebuild them as prediction rows for IoU checks.
    tp = gts[gts["status"] == "TP"].assign(status="TP", record_type="prediction")
    all_preds = pd.concat([preds, tp], ignore_index=True)

    low = None
    if args.low_conf is not None:
        if args.weights is None:
            parser.error("--weights is required with --low-conf")
        test_images = gts[["image", "img_w", "img_h"]].drop_duplicates("image")
        empty = pd.read_csv(args.data / "images.csv")
        empty = empty[(empty["split"] == "test") & ~empty["image"].isin(test_images["image"])]
        if len(empty):
            print("[low-conf] %d test images without GT are skipped (FN analysis only)" % len(empty))
        low = run_low_conf(args, test_images)

    fn = classify_fn(gts[(gts["cls"] == PED) & (gts["status"] == "FN")], all_preds, low)
    fp = classify_fp(preds[(preds["pred_class"] == PED) & (preds["status"] == "FP")], gts)
    fn.to_csv(args.output / "pedestrian_fn_modes.csv", index=False, encoding="utf-8-sig")
    fp.to_csv(args.output / "pedestrian_fp_modes.csv", index=False, encoding="utf-8-sig")

    ped_gt = gts[gts["cls"] == PED]
    small = ped_gt[ped_gt["height_bin"] == SMALL]
    summary = {
        "source": str(args.matched), "low_conf": args.low_conf,
        "definitions": {
            "fn.wrong_class": "IoU>=0.5 with an unmatched prediction of another class",
            "fn.localization": "an unmatched pedestrian box overlaps with %.1f<=IoU<0.5" % LOC_IOU,
            "fn.merged_with_neighbor": "only overlapping pedestrian box is matched to another GT",
            "fn.below_threshold": "IoU>=0.5 pedestrian box exists at conf in [low_conf, 0.25)",
            "fn.no_candidate": "no pedestrian box even at low_conf",
            "fp.duplicate": "IoU>=0.5 with an already-matched pedestrian GT",
            "fp.localization": "overlaps a pedestrian GT but not enough to match",
            "fp.class_confusion": "IoU>=0.5 with a non-pedestrian GT",
            "fp.background": "no meaningful overlap with any GT; may include unlabeled people",
        },
        "fn_total": len(fn), "fp_total": len(fp),
        "fn_modes": fn["fn_mode"].value_counts().to_dict(),
        "fn_modes_by_height": pd.crosstab(fn["height_bin"], fn["fn_mode"]).to_dict(orient="index"),
        "fn_below_threshold_score_quantiles": (fn["low_conf_score"].dropna().quantile([.25, .5, .75]).round(3)
                                               .to_dict() if low is not None else None),
        "fp_modes": fp["fp_mode"].value_counts().to_dict(),
        "fp_modes_by_zone": pd.crosstab(fp["foot_zone"], fp["fp_mode"]).to_dict(orient="index"),
        "fp_modes_by_height": pd.crosstab(fp["height_bin"], fp["fp_mode"]).to_dict(orient="index"),
        "fp_background_confidence_quantiles": fp.loc[fp["fp_mode"] == "background", "prediction_confidence"]
                                              .quantile([.25, .5, .75]).round(3).to_dict(),
        "fp_background_high_conf_ge_0_6": int(((fp["fp_mode"] == "background")
                                               & (fp["prediction_confidence"] >= .6)).sum()),
        "small_pedestrian_recall_by_zone": {
            zone: {"gt": len(g), "recall": round(g["is_tp"].mean(), 4)}
            for zone, g in small.groupby("foot_zone")},
        "pedestrian_recall_by_zone": {
            zone: {"gt": len(g), "recall": round(g["is_tp"].mean(), 4)}
            for zone, g in ped_gt.groupby("foot_zone")},
    }
    (args.output / "failure_modes_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    # Highest-confidence background FPs first: these are the ones to eyeball for missing labels.
    review = fp[fp["fp_mode"] == "background"].sort_values("prediction_confidence", ascending=False)
    review[["image", "clip", "prediction_confidence", "foot_zone", "height_bin",
            "prediction_x1_original", "prediction_y1_original", "prediction_x2_original",
            "prediction_y2_original"]].head(40).to_csv(args.output / "fp_background_review.csv",
                                                        index=False, encoding="utf-8-sig")
    print(json.dumps({k: summary[k] for k in ("fn_modes", "fp_modes", "fp_modes_by_zone",
                                              "small_pedestrian_recall_by_zone")},
                     indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
