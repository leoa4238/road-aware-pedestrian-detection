"""Materialize a split version as a standard images/{split} dataset folder via hard links.

Tools that expect `images/test`, `labels/test`, `road_masks/test` and a `split`
column (e.g. analyze_detection_errors.py) can then run on a new split without
changes. Files are hard-linked (copied if linking fails); the source dataset is
never modified.

Example:
    python aihub/make_split_view.py --data data/analysis_backup_20261013/dataset/aihub_f \
        --manifest outputs/aihub/splits/session_v1/split_manifest.csv --output data/views/session_v1
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

FOLDERS = (("images", ".jpg"), ("labels", ".txt"), ("road_masks", ".png"))


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="split_manifest.csv from make_session_split.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    if output == data or data in output.parents:
        parser.error("Output must be outside the source dataset")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output folder is not empty: %s" % output)

    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    move = manifest.set_index("image")[["source_split", "split"]]
    if move.index.duplicated().any():
        raise ValueError("Manifest has duplicate image names")

    for image, (source_split, split) in move.iterrows():
        for folder, suffix in FOLDERS:
            source = data / folder / source_split / (image + suffix)
            if not source.exists():
                if folder == "road_masks":
                    continue          # analysis treats a missing mask as unknown zone
                raise FileNotFoundError(source)
            link(source, output / folder / split / (image + suffix))

    images = manifest.drop(columns=["session", "source_split"])
    images.to_csv(output / "images.csv", index=False, encoding="utf-8-sig")
    for name in ("objects.csv", "objects_road.csv"):
        table = pd.read_csv(data / name, keep_default_na=False)
        table = table[table["image"].isin(move.index)].copy()
        # Look up by (old split, image) so the excluded filename collision cannot leak in.
        key = set(zip(move["source_split"], move.index))
        table = table[[k in key for k in zip(table["split"], table["image"])]]
        table["split"] = table["image"].map(move["split"])
        table.to_csv(output / name, index=False, encoding="utf-8-sig")
    lines = (data / "dataset.yaml").read_text(encoding="utf-8").splitlines()
    names = lines[[i for i, line in enumerate(lines) if line.startswith("names:")][0]:]
    (output / "dataset.yaml").write_text(
        "\n".join(["path: " + output.as_posix(), "train: images/train", "val: images/val", "test: images/test"]
                  + names) + "\n", encoding="utf-8")
    counts = images.groupby("split").size().to_dict()
    print("view written:", output, counts)


if __name__ == "__main__":
    main()
