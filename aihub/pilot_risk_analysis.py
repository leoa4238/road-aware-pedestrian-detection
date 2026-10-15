"""Pilot: do ego-path geometry and road-mask context help predict pedestrian risk labels?

Uses only numbers (bbox, foot point, SegFormer mask values), never image pixels.
Targets: researcher v0 labels (risk/caution/safe) and, separately, the AI-draft attribute labels
converted with the v1 rule. Session-grouped cross-validation, repeated over seeds.

Example:
    python aihub/pilot_risk_analysis.py --set data/labeling/pilot_v0 --view data/views/session_v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_analysis_core import read_image  # noqa: E402

# Same approximate ego-path band as the labeling page (fractions of width/height).
BAND_BOTTOM, BAND_TOP, BAND_Y_TOP = (.32, .68), (.494, .506), .38


def path_geometry(fx, fy):
    """Signed lateral offset of the foot from the path centre, in half-path-widths (|x|<=1 = inside band)."""
    t = np.clip((1 - fy) / (1 - BAND_Y_TOP), 0, 1)        # 0 at image bottom, 1 at band top
    left = BAND_BOTTOM[0] + t * (BAND_TOP[0] - BAND_BOTTOM[0])
    right = BAND_BOTTOM[1] + t * (BAND_TOP[1] - BAND_BOTTOM[1])
    centre, half = (left + right) / 2, (right - left) / 2
    return (fx - centre) / half, t


def mask_context(mask, fx, fy, h):
    """Road / sidewalk share in a window around the feet and in the ego band between car and feet."""
    mh, mw = mask.shape
    x, y, hh = int(fx * mw), int(min(fy, .999) * mh), max(int(h * mh), 4)
    win = mask[max(0, y - hh // 8): min(mh, y + hh // 6), max(0, x - hh // 3): min(mw, x + hh // 3)]
    road = float((win == 1).mean()) if win.size else np.nan
    side = float((win == 2).mean()) if win.size else np.nan
    # Is there road between the pedestrian and the ego path centre at the same image row?
    row = mask[y, :]
    c = mw // 2
    seg = row[min(x, c): max(x, c) + 1]
    gap_road = float((seg == 1).mean()) if seg.size else np.nan
    return road, side, gap_road


def features(items, view):
    rows = []
    for r in items.itertuples():
        fx, fy = r.foot_x / r.img_w, r.foot_y / r.img_h
        h = r.h / r.img_h
        lat, depth = path_geometry(fx, fy)
        mask = read_image(view / "road_masks" / r.split / (r.image + ".png"), cv2.IMREAD_GRAYSCALE)
        road, side, gap = mask_context(mask, fx, fy, h) if mask is not None else (np.nan,) * 3
        foot_zone = mask[int(min(fy, .999) * mask.shape[0]), int(fx * mask.shape[1])] if mask is not None else -1
        rows.append({"id": r.id, "abs_lat": abs(lat), "lat": lat, "depth": depth, "height": h, "foot_y": fy,
                     "road_win": road, "side_win": side, "gap_road": gap, "foot_on_road": float(foot_zone == 1),
                     "session": re.sub(r"\(\d+\)$", "", r.clip)})
    return pd.DataFrame(rows)


SETS = {
    "position (path geometry)": ["abs_lat", "lat", "depth", "height", "foot_y"],
    "road mask only": ["road_win", "side_win", "gap_road", "foot_on_road"],
    "position + road mask": ["abs_lat", "lat", "depth", "height", "foot_y",
                             "road_win", "side_win", "gap_road", "foot_on_road"],
}


def evaluate(df, target, seeds=20):
    y = df[target].to_numpy()
    out = {}
    risky = y.max()
    # Rule baseline: foot on road mask -> risky (highest class), else lowest class.
    rule = np.where(df["foot_on_road"] == 1, risky, y.min())
    out["rule: foot on road = risky"] = summarize([(y, rule)], risky)
    for name, cols in SETS.items():
        pairs = []
        for seed in range(seeds):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            pred = np.empty_like(y)
            X = df[cols].fillna(0).to_numpy()
            for tr, te in cv.split(X, y, df["session"]):
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
                pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
            pairs.append((y, pred))
        out[name] = summarize(pairs, risky)
    return out


def summarize(pairs, risky):
    m = np.array([[f1_score(y, p, average="macro"),
                   recall_score(y == risky, p == risky, zero_division=0),
                   precision_score(y == risky, p == risky, zero_division=0)] for y, p in pairs])
    mean, sd = m.mean(0), m.std(0)
    return {"macro_f1": f"{mean[0]:.3f} ± {sd[0]:.3f}", "risky_recall": f"{mean[1]:.3f} ± {sd[1]:.3f}",
            "risky_precision": f"{mean[2]:.3f} ± {sd[2]:.3f}"}


def v1_rule(r):
    p, d = int(r.position), r.direction
    if p == 1 or (p == 2 and d == "q") or (r.occluded and d == "q"):
        return 3
    if p == 2 or (p == 3 and d == "q"):
        return 2
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--view", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    items = pd.read_csv(args.set / "items.csv")
    feats = features(items, args.view)
    result = {}

    user = pd.read_csv(args.set / "labels_partial_20261015.tsv", sep="\t")
    d = feats.merge(user, on="id")
    d = d[d["label"] > 0]
    result["researcher v0 labels (risk/caution/safe)"] = {
        "n": len(d), "sessions": d["session"].nunique(),
        "class_counts": d["label"].value_counts().sort_index().to_dict(), **evaluate(d, "label")}

    draft = pd.read_csv(args.set / "attributes_pilot_v0_claude_draft.csv", dtype={"position": "string"})
    draft = draft[draft["excluded"] == 0].copy()
    draft["risk"] = draft.apply(v1_rule, axis=1)
    d2 = feats.merge(draft[["id", "risk", "position"]], on="id")
    result["AI draft, v1 rule risk (reference only)"] = {
        "n": len(d2), "sessions": d2["session"].nunique(),
        "class_counts": d2["risk"].value_counts().sort_index().to_dict(), **evaluate(d2, "risk")}
    # How well does the approximate band reproduce the drafted position attribute?
    d2["in_band"] = d2["abs_lat"] <= 1
    result["band check: AI position vs |lateral| <= 1"] = pd.crosstab(d2["position"], d2["in_band"]).to_dict()

    text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
