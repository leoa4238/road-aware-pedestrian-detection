"""Private review overlays and high-resolution descriptive figures."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from error_analysis_core import read_image, write_image


def overlay(data, row, destination, color):
    data = Path(data)
    picture = read_image(data / "images" / row["split"] / (row["image"] + ".jpg"))
    if picture is None:
        raise ValueError("Review image could not be decoded: " + row["image"])
    ih, iw = picture.shape[:2]
    mask = read_image(data / "road_masks" / row["split"] / (row["image"] + ".png"), cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
        tint = picture.copy()
        tint[mask == 1] = (60, 160, 70)
        tint[mask == 2] = (190, 90, 30)
        picture = cv2.addWeighted(picture, .7, tint, .3, 0)
        boundaries = cv2.morphologyEx((mask == 1).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        picture[boundaries > 0] = (0, 210, 255)
    sx, sy = iw / float(row["img_w"]), ih / float(row["img_h"])
    x, y, w, h = [float(row[k]) for k in ["x", "y", "w", "h"]]
    cv2.rectangle(picture, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)), color, 2)
    if np.isfinite([row["foot_x"], row["foot_y"]]).all():
        cv2.circle(picture, (int(row["foot_x"] * sx), int(row["foot_y"] * sy)), 5, (255, 0, 255), -1)
    title = "%s obj:%s %s zone:%s" % (row["status"], row.get("obj_idx", "FP"), row["image"], row.get("foot_zone", "unknown"))
    cv2.rectangle(picture, (0, 0), (iw, 66), (15, 15, 15), -1)
    cv2.putText(picture, title[:115], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(picture, "Estimated mask only: green=road blue=sidewalk yellow=boundary; not ground truth",
                (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    write_image(destination, picture)


def make_review_outputs(data, detections, output, review_per_zone=50, examples_per_type=12, seed=42):
    output = Path(output)
    (output / "failure_examples").mkdir(parents=True, exist_ok=True)
    (output / "mask_review").mkdir(parents=True, exist_ok=True)
    reviews, examples = [], []
    gt = detections[(detections["record_type"] == "gt") & (detections["gt_class"] == "pedestrian")]
    for zone, group in gt.groupby("foot_zone"):
        chosen = group.sample(n=min(review_per_zone, len(group)), random_state=seed)
        for _, row in chosen.iterrows():
            name = "%s_%s_%s.jpg" % (zone, row["image"], int(row["obj_idx"]))
            destination = output / "mask_review" / name
            overlay(data, row, destination, (255, 255, 255))
            reviews.append({"image": row["image"], "split": row["split"], "obj_idx": row["obj_idx"],
                            "estimated_foot_zone": zone, "overlay": str(destination.resolve()),
                            "human_true_zone": "", "human_mask_correct": "", "human_note": ""})
    selections = [("FN", gt[gt["status"] == "FN"]),
                  ("FP", detections[(detections["status"] == "FP") & (detections["pred_class"] == "pedestrian")])]
    for status, group in selections:
        chosen = group.sample(n=min(examples_per_type, len(group)), random_state=seed)
        for index, (_, row) in enumerate(chosen.iterrows()):
            destination = output / "failure_examples" / ("%s_%02d_%s.jpg" % (status, index, row["image"]))
            overlay(data, row, destination, (0, 0, 255) if status == "FN" else (0, 170, 255))
            examples.append({"status": status, "image": row["image"], "overlay": str(destination.resolve()),
                             "selection": "seeded_random_not_representative_frequency_estimate"})
    pd.DataFrame(reviews).to_csv(output / "mask_review_sheet.csv", index=False, encoding="utf-8-sig")
    (output / "failure_examples_index.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return {"mask_review_samples": len(reviews), "failure_examples": len(examples), "human_reviews_completed": 0}


def make_figures(groups, output, confidence=.25, iou=.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for dimension in ("height_bin", "foot_zone", "road", "day_night", "weather", "boundary_bin"):
        group = groups[groups["dimension"] == dimension].copy()
        if group.empty:
            continue
        if dimension in ("height_bin", "boundary_bin"):
            group = group.assign(_order=group["group"].map(
                lambda value: float(value.split(",")[0][1:]) if value.startswith("[") else float("inf")))
            group = group.sort_values("_order")
        else:
            group = group.sort_values("group")
        figure, axis = plt.subplots(figsize=(max(6, .9 * len(group)), 4.2))
        values = group["recall"].to_numpy()
        axis.bar(range(len(group)), values, color="#287D70", width=.65)
        for i, row in enumerate(group.itertuples(index=False)):
            if np.isfinite([row.recall_ci_low, row.recall_ci_high]).all():
                axis.plot([i, i], [row.recall_ci_low, row.recall_ci_high], color="#222222", linewidth=1.5)
            annotation_y = row.recall_ci_high if np.isfinite(row.recall_ci_high) else row.recall
            axis.text(i, min(annotation_y + .04, 1.08) if np.isfinite(annotation_y) else .04,
                      "n=%d%s" % (row.gt, "*" if row.gt < 30 else ""), ha="center", fontsize=9)
        axis.set_xticks(range(len(group)))
        axis.set_xticklabels(group["group"], rotation=20, ha="right")
        axis.set_ylim(0, 1.13)
        axis.set_ylabel("Pedestrian Recall")
        axis.set_title(dimension + " | 95% clip-bootstrap CI", fontsize=12)
        axis.set_xlabel("IoU>=%g, conf>=%g | * GT<30; absent GT = n/a" % (iou, confidence), fontsize=9)
        figure.tight_layout()
        figure.savefig(output / (dimension + ".png"), dpi=300)
        figure.savefig(output / (dimension + ".pdf"))
        plt.close(figure)
        count += 1
    return count
