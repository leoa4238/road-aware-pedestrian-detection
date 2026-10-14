"""Independent artifact/count checks; no detector training or inference."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_sanity import sha256
from analysis_visuals import make_figures
from error_analysis_core import read_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--refresh-figures", action="store_true")
    args = parser.parse_args()
    output, data = args.output.resolve(), args.data.resolve()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    records = pd.read_csv(output / "detections_matched.csv", keep_default_na=False)
    gt = records[records.record_type == "gt"]
    assert not gt.duplicated(["split", "image", "obj_idx"]).any(), "GT duplicated"
    actual_gt = pd.read_csv(data / "objects.csv")
    actual_gt = actual_gt[actual_gt.split == "test"]
    assert len(gt) == len(actual_gt), "GT dropped or duplicated"
    used = records[records.status.isin(["TP", "FP"])]
    assert not used.duplicated(["image", "prediction_index"]).any(), "Prediction used more than once"
    cache = list((output / "prediction_cache").glob("*.json"))
    assert len(cache) == summary["test_images"]
    predictions = sum(len(json.loads(p.read_text())["classes"]) for p in cache)
    assert predictions == len(used), "Predictions dropped or double-counted"
    counts = summary["fixed_threshold_metrics"]["all_classes_micro"]
    assert counts["tp"] + counts["fn"] == len(gt)
    assert counts["tp"] + counts["fp"] == predictions
    groups = pd.read_csv(output / "pedestrian_failure_analysis.csv")
    ped_count = summary["fixed_threshold_metrics"]["pedestrian"]["gt"]
    for _, group in groups.groupby("dimension"):
        assert group["gt"].sum() == ped_count, "Grouping lost GT"
        assert np.array_equal(group["gt"].to_numpy(), (group["tp"] + group["fn"]).to_numpy())
    if args.refresh_figures:
        make_figures(groups, output, summary["settings"]["conf"], summary["settings"]["iou"])
    images = list((output / "failure_examples").glob("*.jpg")) + list((output / "mask_review").glob("*.jpg"))
    images += list((output / "figures").glob("*.png"))
    for path in images:
        decoded = read_image(path)
        assert decoded is not None and decoded.size and decoded.max() > decoded.min(), "Broken/blank image: " + str(path)
    manifest = json.loads(args.extraction_manifest.read_text(encoding="utf-8"))
    for archive in manifest:
        assert sha256(archive["archive"]) == archive["sha256"], "Original ZIP changed"
    repo = Path(__file__).resolve().parents[1]
    preserved_sources = ["README.md", "PROGRESS.md", "CLAUDE.md", "aihub/prepare_dataset.py",
                         "aihub/make_road_masks.py", "aihub/train_baseline.py", "outputs/aihub/results/yolov8s_e50_s0.json"]
    for name in preserved_sources:
        archived = subprocess.check_output(["git", "-c", "safe.directory=" + repo.as_posix(),
                                            "show", "HEAD:" + name], cwd=str(repo))
        current = (repo / name).read_bytes()
        assert current.replace(b"\r\n", b"\n") == archived.replace(b"\r\n", b"\n"), "Original source changed: " + name
    result = {"passed": True, "gt_rows": len(gt), "prediction_rows": predictions,
              "test_image_caches": len(cache), "decoded_nonblank_images": len(images),
              "original_zip_hashes_unchanged": True, "preserved_original_source_files": preserved_sources,
              "figure_source_sha256": sha256(repo / "aihub/analysis_visuals.py"),
              "human_segmentation_review_completed": False, "no_new_detector_training": True}
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
