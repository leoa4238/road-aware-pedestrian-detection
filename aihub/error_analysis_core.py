"""Pure matching, coordinate, mask and clustered-statistics helpers (no YOLO)."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HEIGHT_EDGES = (0, 32, 64, 128, 256, float("inf"))
BOUNDARY_EDGES = (0, .25, .5, 1, 2, float("inf"))
IMAGE_DISTANCE_EDGES = (0, .01, .025, .05, .1, float("inf"))


def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    except (OSError, ValueError):
        return None


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError("Image encoding failed: " + str(path))
    encoded.tofile(str(path))


def rescale_xyxy(boxes, source_size, target_size):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    sw, sh = source_size
    tw, th = target_size
    if min(sw, sh, tw, th) <= 0:
        raise ValueError("Image dimensions must be positive")
    return boxes * np.array([tw / sw, th / sh, tw / sw, th / sh])


def xywh_to_xyxy(boxes):
    result = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    result[:, 2:] += result[:, :2]
    return result


def pairwise_iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    intersection = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:])
                              - np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(axis=2)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    union = aa[:, None] + bb[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_detections(gt_boxes, gt_classes, pred_boxes, pred_classes, scores, threshold=.5):
    """Class-aware, descending-confidence greedy matching; every GT/pred used once."""
    overlaps = pairwise_iou(gt_boxes, pred_boxes)
    gt_classes, pred_classes = np.asarray(gt_classes), np.asarray(pred_classes)
    remaining = set(range(len(gt_classes)))
    matched = []
    used_predictions = set()
    for p in np.argsort(-np.asarray(scores), kind="stable"):
        eligible = [g for g in sorted(remaining)
                    if gt_classes[g] == pred_classes[p] and overlaps[g, p] >= threshold]
        if eligible:
            g = max(eligible, key=lambda index: overlaps[index, p])
            remaining.remove(g)
            used_predictions.add(int(p))
            matched.append((g, int(p), float(overlaps[g, p])))
    unmatched_predictions = sorted(set(range(len(pred_classes))) - used_predictions)
    # Secondary one-to-one association diagnoses wrong-class overlaps, not extra TPs.
    candidates = [(float(overlaps[g, p]), g, p) for g in remaining for p in unmatched_predictions
                  if gt_classes[g] != pred_classes[p] and overlaps[g, p] >= threshold]
    confused, used_gt, used_pred = {}, set(), set()
    for overlap, g, p in sorted(candidates, reverse=True):
        if g not in used_gt and p not in used_pred:
            confused[g] = (p, overlap)
            used_gt.add(g)
            used_pred.add(p)
    return matched, sorted(remaining), unmatched_predictions, confused


def signed_distance(mask):
    """Distance to nearest opposite road/non-road pixel center; outside +, inside -."""
    road = (np.asarray(mask) == 1).astype(np.uint8)
    if not road.any() or road.all():
        return np.full(road.shape, np.nan, dtype=np.float32)
    outside = cv2.distanceTransform(1 - road, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    inside = cv2.distanceTransform(road, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return outside - inside


def boundary_features(mask, point, box_wh, image_size, distance=None):
    blank = {"signed_boundary_mask_px": np.nan, "signed_boundary_bbox_norm": np.nan,
             "signed_boundary_image_norm": np.nan, "boundary_abs_bbox_norm": np.nan,
             "boundary_abs_image_norm": np.nan, "point_zone": "unknown",
             "boundary_status": "missing_mask", "mask_pixel_width_original": np.nan,
             "support_signed_boundary_bbox_norm": np.nan, "support_zone": "unknown"}
    if mask is None:
        return blank
    mh, mw = mask.shape
    iw, ih = image_size
    px, py = point
    if not np.isfinite([px, py]).all() or not (0 <= px < iw and 0 <= py < ih):
        blank["boundary_status"] = "invalid_or_outside_point"
        return blank
    sx, sy = mw / iw, mh / ih
    x, y = min(int(px * sx), mw - 1), min(int(py * sy), mh - 1)
    distance = signed_distance(mask) if distance is None else distance
    value = float(distance[y, x])
    height = float(box_wh[1]) * sy
    blank.update(point_zone={0: "other", 1: "road", 2: "sidewalk"}.get(int(mask[y, x]), "unknown"),
                 mask_pixel_width_original=iw / mw,
                 boundary_status="ok" if np.isfinite(value) and height > 0 else "no_boundary_or_invalid_height")
    if np.isfinite(value) and height > 0:
        blank.update(signed_boundary_mask_px=value, signed_boundary_bbox_norm=value / height,
                     signed_boundary_image_norm=value / mh, boundary_abs_bbox_norm=abs(value) / height,
                     boundary_abs_image_norm=abs(value) / mh)
    # Sensitivity only: nearest road/sidewalk support pixel in the original foot window.
    # This reduces person-hole artifacts but does NOT reconstruct a physical curb.
    w = max(float(box_wh[0]) * sx, 1)
    h = max(height, 1)
    x0, x1 = max(0, int(x - .4 * w)), min(mw, int(x + .4 * w) + 1)
    y0, y1 = max(0, int(y - .05 * h)), min(mh, int(y + .10 * h) + 1)
    yy, xx = np.nonzero(np.isin(mask[y0:y1, x0:x1], [1, 2]))
    if len(xx):
        index = np.argmin((xx + x0 - px * sx) ** 2 + (yy + y0 - py * sy) ** 2)
        xx, yy = int(xx[index] + x0), int(yy[index] + y0)
        blank["support_zone"] = "road" if mask[yy, xx] == 1 else "sidewalk"
        if np.isfinite(distance[yy, xx]) and height > 0:
            blank["support_signed_boundary_bbox_norm"] = float(distance[yy, xx]) / height
    return blank


def bin_value(value, edges):
    if not np.isfinite(value) or value < edges[0]:
        return "unknown"
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return "[%g,%s)" % (lo, "inf" if np.isinf(hi) else "%g" % hi)
    return "unknown"


def session_id(clip):
    return re.sub(r"\(\d+\)$", "", str(clip))


def rate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "gt": int(tp + fn),
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "miss_rate": fn / (tp + fn) if tp + fn else None}


def cluster_recall_ci(group, repetitions=2000, seed=42):
    if group.empty:
        return (None, None, 0)
    counts = group.groupby("clip", dropna=False)["is_tp"].agg(["sum", "count"]).to_numpy(float)
    if len(counts) < 2:
        return (None, None, len(counts))
    rng = np.random.default_rng(seed)
    selection = rng.integers(0, len(counts), size=(repetitions, len(counts)))
    sums = counts[selection].sum(axis=1)
    values = sums[:, 0] / sums[:, 1]
    lo, hi = np.quantile(values, [.025, .975])
    return (float(lo), float(hi), len(counts))


def group_table(gt, dimensions, repetitions=2000):
    rows = []
    for name in dimensions:
        for value, group in gt.groupby(name, dropna=False, sort=True):
            lo, hi, clips = cluster_recall_ci(group, repetitions)
            session_group = group.copy()
            session_group["clip"] = session_group["clip"].map(session_id)
            slo, shi, sessions = cluster_recall_ci(session_group, repetitions)
            n, tp = len(group), int(group["is_tp"].sum())
            rows.append({"dimension": name, "group": str(value), "gt": n, "tp": tp,
                         "fn": n - tp, "recall": tp / n, "miss_rate": 1 - tp / n,
                         "recall_ci_low": lo, "recall_ci_high": hi, "clips": clips,
                         "recall_session_ci_low": slo, "recall_session_ci_high": shi, "sessions": sessions,
                         "sample_status": "insufficient_gt_lt_30" if n < 30 else
                         ("insufficient_clusters" if clips < 2 else "ok")})
        expected = {"road": ["R01", "R02", "R03", "R04", "R05", "R06"],
                    "time": ["mrh", "day", "lunch", "afterschool", "erh", "night"],
                    "day_night": ["daytime", "night"], "weather": ["clear", "foggy", "rainy"],
                    "foot_zone": ["road", "sidewalk", "other", "unknown"]}.get(name, [])
        seen = {r["group"] for r in rows if r["dimension"] == name}
        for value in expected:
            if value not in seen:
                rows.append({"dimension": name, "group": value, "gt": 0, "tp": 0, "fn": 0,
                             "recall": None, "miss_rate": None, "recall_ci_low": None, "recall_ci_high": None,
                             "clips": 0, "sessions": 0, "recall_session_ci_low": None,
                             "recall_session_ci_high": None, "sample_status": "absent_gt"})
    return pd.DataFrame(rows)


def clustered_boundary_contrast(gt, repetitions=2000, seed=42):
    """Near-minus-far miss rate, resampling whole clips; no independence assumption."""
    data = gt[np.isfinite(gt["boundary_abs_bbox_norm"])].copy()
    data["near"] = data["boundary_abs_bbox_norm"] <= .25
    if data.empty or data["near"].nunique() < 2:
        return {"status": "insufficient_near_or_far"}
    grouped = []
    for _, group in data.groupby("clip", sort=True):
        near, far = group[group["near"]], group[~group["near"]]
        grouped.append([len(near), int((1 - near["is_tp"]).sum()),
                        len(far), int((1 - far["is_tp"]).sum())])
    counts = np.asarray(grouped, dtype=float)
    total = counts.sum(axis=0)
    estimate = total[1] / total[0] - total[3] / total[2]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        n = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        if n[0] and n[2]:
            estimates.append(n[1] / n[0] - n[3] / n[2])
    ci = np.quantile(estimates, [.025, .975]).tolist() if len(counts) > 1 and estimates else [None, None]
    return {"status": "exploratory", "near_rule": "abs(signed distance)/bbox_height <= 0.25",
            "near_gt": int(total[0]), "far_gt": int(total[2]), "near_fn": int(total[1]),
            "far_fn": int(total[3]), "near_minus_far_miss_rate": float(estimate),
            "ci95": ci, "clips": len(counts), "bootstrap_valid_replicates": len(estimates)}
