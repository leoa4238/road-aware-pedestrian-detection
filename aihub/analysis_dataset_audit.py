"""Read-only audits of the prepared backup; never rewrite labels or splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from error_analysis_core import session_id, xywh_to_xyxy


def load_tables(data, validate_test=True):
    images = pd.read_csv(Path(data) / "images.csv", keep_default_na=False)
    objects = pd.read_csv(Path(data) / "objects.csv")
    road = pd.read_csv(Path(data) / "objects_road.csv")
    checked_images = images[images["split"] == "test"] if validate_test else images.iloc[:0]
    checked_objects = objects[objects["split"] == "test"] if validate_test else objects.iloc[:0]
    checked_road = road[road["split"] == "test"] if validate_test else road.iloc[:0]
    if checked_images.duplicated(["split", "image"]).any():
        raise ValueError("Ambiguous duplicate image metadata keys; report/fix upstream, not automatically")
    if checked_objects.duplicated(["split", "image", "obj_idx"]).any():
        raise ValueError("Duplicate object keys")
    if checked_road.duplicated(["split", "image", "obj_idx"]).any():
        raise ValueError("Duplicate road-object keys")
    return images, objects, road


def audit_dataset(data, output, class_names):
    data, output = Path(data), Path(output)
    images, objects, road = load_tables(data, validate_test=False)
    duplicate_image_keys = images[images.duplicated(["split", "image"], keep=False)]
    duplicate_object_keys = objects[objects.duplicated(["split", "image", "obj_idx"], keep=False)]
    duplicate_road_keys = road[road.duplicated(["split", "image", "obj_idx"], keep=False)]
    duplicate_object_keys.to_csv(output / "duplicate_object_metadata.csv", index=False)
    metadata = images.copy()
    metadata["session"] = metadata["clip"].map(session_id)
    leak_clips = metadata.groupby("clip")["split"].nunique()
    leak_sessions = metadata.groupby("session")["split"].nunique()
    sessions = set(leak_sessions[leak_sessions > 1].index)
    repeated_stems = metadata[metadata.duplicated("image", keep=False)].copy()
    metadata[metadata["session"].isin(sessions)].to_csv(output / "session_overlap_images.csv", index=False)
    repeated_stems.to_csv(output / "duplicate_filename_metadata.csv", index=False)
    orphaned, missing = [], []
    expected_keys = set(zip(images["split"], images["image"]))
    actual_counts = {}
    for split in ("train", "val", "test"):
        files = list((data / "images" / split).glob("*.jpg"))
        actual_counts[split] = len(files)
        for path in files:
            if (split, path.stem) not in expected_keys:
                orphaned.append({"split": split, "image": path.stem})
    source_bounds = {"out_of_bounds_boxes": 0, "nonpositive_boxes": 0}
    coordinates, mismatches, missing_labels = [], [], []
    for (split, image), group in objects.groupby(["split", "image"], sort=True):
        group = group.sort_values("obj_idx")
        iw, ih = float(group["img_w"].iloc[0]), float(group["img_h"].iloc[0])
        gt = xywh_to_xyxy(group[["x", "y", "w", "h"]].to_numpy())
        source_bounds["out_of_bounds_boxes"] += int(((gt[:, :2] < 0).any(axis=1)
                                                      | (gt[:, 2:] > [iw, ih]).any(axis=1)).sum())
        source_bounds["nonpositive_boxes"] += int((group[["w", "h"]].to_numpy() <= 0).any(axis=1).sum())
        path = data / "labels" / split / (image + ".txt")
        if not path.exists():
            missing_labels.append({"split": split, "image": image})
            continue
        content = path.read_text(encoding="utf-8").splitlines()
        try:
            label = np.asarray([[float(x) for x in line.split()] for line in content if line.strip()]).reshape(-1, 5)
        except (ValueError, TypeError):
            mismatches.append({"split": split, "image": image, "reason": "malformed_label"})
            continue
        if len(label) != len(group):
            mismatches.append({"split": split, "image": image, "reason": "object_count_mismatch"})
            continue
        gt_cls = np.array([class_names.index(str(c)) for c in group["cls"]])
        if not np.array_equal(label[:, 0].astype(int), gt_cls):
            mismatches.append({"split": split, "image": image, "reason": "class_order_mismatch"})
        cx, cy, w, h = label[:, 1], label[:, 2], label[:, 3], label[:, 4]
        restored = np.column_stack([(cx - w / 2) * iw, (cy - h / 2) * ih,
                                    (cx + w / 2) * iw, (cy + h / 2) * ih])
        errors = np.max(np.abs(gt - restored), axis=1)
        for obj_idx, error in zip(group["obj_idx"], errors):
            if error > .01:
                coordinates.append({"split": split, "image": image, "obj_idx": int(obj_idx),
                                    "max_edge_error_original_px": float(error)})
    for row in images.itertuples(index=False):
        if not (data / "images" / row.split / (row.image + ".jpg")).exists():
            missing.append({"split": row.split, "image": row.image})
    pd.DataFrame(coordinates, columns=["split", "image", "obj_idx", "max_edge_error_original_px"]).to_csv(
        output / "label_coordinate_mismatches.csv", index=False)
    pd.DataFrame(missing, columns=["split", "image"]).to_csv(output / "missing_images.csv", index=False)
    ped = objects[objects["cls"] == "pedestrian"]
    # Ambiguous road rows cannot be joined safely. Report them, never pick a first row.
    comparable_road = road[~road.duplicated(["split", "image", "obj_idx"], keep=False)]
    joined = ped.merge(comparable_road, on=["split", "image", "obj_idx"], how="left", indicator=True,
                       suffixes=("", "_road"), validate="many_to_one")
    disagreement = 0
    test_disagreement = 0
    for column in ("x", "y", "w", "h", "img_w", "img_h", "foot_x", "foot_y"):
        differs = ~np.isclose(joined[column], joined[column + "_road"], equal_nan=True) & (joined["_merge"] == "both")
        disagreement += int(differs.sum())
        test_disagreement += int((differs & (joined["split"] == "test")).sum())
    result = {"metadata_images": len(images), "disk_images": actual_counts,
              "metadata_split_counts": images["split"].value_counts().to_dict(),
              "clip_split_overlap_count": int((leak_clips > 1).sum()),
              "session_split_overlap_count": len(sessions),
              "session_overlap_image_rows": int(metadata["session"].isin(sessions).shape[0]),
              "duplicate_filename_rows": len(repeated_stems),
              "duplicate_object_key_rows": len(duplicate_object_keys),
              "duplicate_road_key_rows": len(duplicate_road_keys),
              "duplicate_filename_across_clips": int((repeated_stems.groupby("image")["clip"].nunique() > 1).sum()),
              "missing_images": missing, "unindexed_images": orphaned,
              "missing_labels": missing_labels, "label_structure_mismatches": mismatches,
              "label_coordinate_error_gt_0_01_px": len(coordinates),
              "label_coordinate_max_error_px": max([r["max_edge_error_original_px"] for r in coordinates], default=0),
              "source_box_checks": source_bounds, "pedestrian_objects": len(ped),
              "pedestrian_road_missing_rows": int((joined["_merge"] != "both").sum()),
              "objects_road_coordinate_disagreements": disagreement,
              "objects_road_test_coordinate_disagreements": test_disagreement,
              "occlusion": {"status": "unavailable", "reason": "No original annotation JSON or occlusion column supplied"},
              "filename_collision_limitation": "Prepared metadata cannot reveal records overwritten before export; source ZIP/JSON required for complete proof.",
              "split_policy": "Report only; no split, metadata or label modified"}
    result["test_analysis_blocking"] = bool(any(r["split"] == "test" for r in missing + missing_labels + mismatches)
                                            or (leak_clips > 1).any() or test_disagreement
                                            or any((frame["split"] == "test").any() for frame in
                                                   [duplicate_image_keys, duplicate_object_keys, duplicate_road_keys]))
    (output / "dataset_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
