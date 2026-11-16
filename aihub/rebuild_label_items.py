"""Rebuild items.csv for an existing labeling page (when only index*.html and its jpg folders survive).

The page lists the pedestrian ids (`<image>__<obj_idx>`); coordinates come from objects_road.csv
and clip/scenario from images.csv of a dataset folder (e.g. the Drive backup's aihub_f). The
`split` column is the dataset's own split folder, so --view for later scripts is that dataset folder.

Example:
    python aihub/rebuild_label_items.py --page D:/nonmoon/index_attr.html --data data/aihub_f \
        --output data/labeling/pilot_v0
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

KEEP = ["id", "image", "obj_idx", "split", "clip", "scenario", "road", "time", "weather",
        "x", "y", "w", "h", "foot_x", "foot_y", "img_w", "img_h"]


def page_ids(page):
    html = Path(page).read_text(encoding="utf-8")
    m = re.search(r"const ITEMS=(\[.*?\]), KEY=", html, re.S)
    if not m:
        raise ValueError("No ITEMS list in %s" % page)
    return [t["id"] for t in json.loads(m.group(1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="folder with objects_road.csv and images.csv")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    ids = pd.DataFrame({"id": page_ids(args.page)})
    road = pd.read_csv(args.data / "objects_road.csv")
    road = road[road["cls"] == "pedestrian"].copy()
    road["id"] = road["image"].astype(str) + "__" + road["obj_idx"].astype(int).astype(str)
    images = pd.read_csv(args.data / "images.csv")[["image", "clip", "scenario", "road", "time", "weather"]]
    items = ids.merge(road, on="id", how="left").merge(images, on="image", how="left")
    missing = items["x"].isna().sum()
    if missing:
        raise SystemExit("%d page ids not found in %s/objects_road.csv" % (missing, args.data))
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "items.csv"
    if out.exists():
        raise FileExistsError("%s already exists" % out)
    items[KEEP].to_csv(out, index=False, encoding="utf-8-sig")
    print("items:", len(items), "images:", items["image"].nunique(), "splits:", items["split"].value_counts().to_dict())
    print("saved:", out)


if __name__ == "__main__":
    main()
