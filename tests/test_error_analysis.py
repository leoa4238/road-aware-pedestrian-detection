"""Synthetic tests: no real images, weights, or GPU needed."""
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aihub"))
from error_analysis_core import (rescale_xyxy, xywh_to_xyxy, pairwise_iou, match_detections,
                                 signed_distance, boundary_features, session_id, bin_value,
                                 rate_metrics, cluster_recall_ci, clustered_boundary_contrast)
from error_analysis_core import group_table
from analysis_statistics import logistic_failure_analysis
from extract_analysis_backup import extract
from analysis_dataset_audit import audit_dataset
from analyze_detection_errors import validate_sanity


class MatchingTests(unittest.TestCase):
    def test_coordinate_resize(self):
        actual = rescale_xyxy([[10, 20, 110, 220]], (1280, 720), (1920, 1080))
        np.testing.assert_allclose(actual, [[15, 30, 165, 330]])

    def test_xywh_top_left(self):
        np.testing.assert_equal(xywh_to_xyxy([[10, 20, 30, 40]]), [[10, 20, 40, 60]])

    def test_duplicate_prediction_is_fp(self):
        matched, fn, fp, confused = match_detections([[0, 0, 10, 10]], [0],
                                                    [[0, 0, 10, 10]] * 2, [0, 0], [.4, .9])
        self.assertEqual(matched, [(0, 1, 1.)])
        self.assertEqual(fp, [0])
        self.assertEqual(fn, [])

    def test_prediction_cannot_match_two_gt(self):
        matched, fn, fp, _ = match_detections([[0, 0, 10, 10]] * 2, [0, 0],
                                             [[0, 0, 10, 10]], [0], [.9])
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(fn), 1)

    def test_misclassified_not_tp(self):
        matched, fn, fp, confused = match_detections([[0, 0, 10, 10]], [0],
                                                    [[0, 0, 10, 10]], [3], [.9])
        self.assertEqual(matched, [])
        self.assertEqual((fn, fp), ([0], [0]))
        self.assertEqual(confused[0], (0, 1.))

    def test_confusion_is_one_to_one(self):
        _, _, _, confused = match_detections([[0, 0, 10, 10]] * 2, [0, 0],
                                             [[0, 0, 10, 10]], [3], [.9])
        self.assertEqual(len(confused), 1)

    def test_empty(self):
        self.assertEqual(match_detections([], [], [], [], []), ([], [], [], {}))
        self.assertEqual(pairwise_iou([], []).shape, (0, 0))

    def test_nonoverlap(self):
        self.assertEqual(pairwise_iou([[0, 0, 1, 1]], [[2, 2, 3, 3]])[0, 0], 0)


class MaskTests(unittest.TestCase):
    def test_signs(self):
        mask = np.zeros((10, 10), np.uint8)
        mask[:, :5] = 1
        distance = signed_distance(mask)
        self.assertLess(distance[5, 2], 0)
        self.assertGreater(distance[5, 8], 0)
        self.assertAlmostEqual(distance[5, 2], -3.)

    def test_two_normalizations(self):
        mask = np.zeros((10, 10), np.uint8)
        mask[:, :5] = 1
        result = boundary_features(mask, (20, 50), (10, 20), (100, 100))
        self.assertAlmostEqual(result["signed_boundary_bbox_norm"], -1.5)
        self.assertAlmostEqual(result["signed_boundary_image_norm"], -.3)
        self.assertEqual(result["mask_pixel_width_original"], 10)

    def test_homogeneous_no_physical_boundary(self):
        self.assertTrue(np.isnan(signed_distance(np.ones((5, 5), np.uint8))).all())
        self.assertTrue(np.isnan(signed_distance(np.zeros((5, 5), np.uint8))).all())

    def test_missing_mask(self):
        self.assertEqual(boundary_features(None, (2, 2), (1, 1), (10, 10))["boundary_status"], "missing_mask")

    def test_out_of_bounds_not_clipped(self):
        result = boundary_features(np.zeros((5, 5), np.uint8), (10, 20), (1, 1), (10, 10))
        self.assertEqual(result["boundary_status"], "invalid_or_outside_point")


class StatisticsTests(unittest.TestCase):
    def test_missing_metrics_not_zero(self):
        self.assertIsNone(rate_metrics(0, 0, 0)["recall"])
        self.assertEqual(rate_metrics(0, 1, 2)["f1"], 0)

    def test_group_bootstrap_seed(self):
        data = pd.DataFrame({"clip": ["a", "a", "b"], "is_tp": [1, 1, 0]})
        self.assertEqual(cluster_recall_ci(data, 100), cluster_recall_ci(data, 100))
        lo, hi, n = cluster_recall_ci(data, 100)
        self.assertEqual(n, 2)
        self.assertTrue(0 <= lo <= hi <= 1)

    def test_one_cluster_ci_unavailable(self):
        self.assertEqual(cluster_recall_ci(pd.DataFrame({"clip": ["a"], "is_tp": [1]})), (None, None, 1))

    def test_absent_road_is_not_zero_recall(self):
        frame = pd.DataFrame({"road": ["R03"], "clip": ["a(1)"], "is_tp": [1]})
        result = group_table(frame, ["road"], 10)
        absent = result[result["group"] == "R06"].iloc[0]
        self.assertEqual(absent["gt"], 0)
        self.assertTrue(pd.isna(absent["recall"]))
        self.assertEqual(absent["sample_status"], "absent_gt")

    def test_logistic_collinearity_reported(self):
        frame = pd.DataFrame({"boundary_abs_bbox_norm": [.1 + .1 * (i % 5) for i in range(40)],
                              "h": [100.] * 40, "is_tp": [int(i % 3 != 0) for i in range(40)],
                              "time": ["night" if i < 20 else "day" for i in range(40)],
                              "weather": ["foggy" if i < 20 else "clear" for i in range(40)],
                              "clip": ["c%d" % (i // 10) for i in range(40)]})
        result = logistic_failure_analysis(frame, repetitions=5)
        self.assertTrue(result["rank_deficient"])
        self.assertIn(["night", "weather_foggy"], result["perfectly_collinear_pairs"])

    def test_contrast(self):
        data = pd.DataFrame({"clip": ["a", "a", "b", "b"], "is_tp": [0, 1, 0, 1],
                             "boundary_abs_bbox_norm": [.1, .5, .1, .5]})
        result = clustered_boundary_contrast(data, 50)
        self.assertEqual(result["near_minus_far_miss_rate"], 1)
        self.assertEqual(result["ci95"], [1, 1])

    def test_bins_and_session(self):
        self.assertEqual(bin_value(.25, [0, .25, .5]), "[0.25,0.5)")
        self.assertEqual(session_id("230906_095031_K(1)"), "230906_095031_K")
        self.assertEqual(session_id("session"), "session")

    def test_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../bad.txt", "x")
            with self.assertRaises(ValueError):
                extract(path, Path(folder) / "output")

    def test_normal_zip_extraction_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ok.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("dataset/file.txt", "test")
            destination = Path(folder) / "out"
            self.assertEqual(extract(path, destination), 1)
            self.assertEqual((destination / "dataset/file.txt").read_text(), "test")
            with self.assertRaises(FileExistsError):
                extract(path, destination)

    def test_failed_sanity_blocks_analysis(self):
        with self.assertRaises(RuntimeError):
            validate_sanity({"passed": False}, Path("missing"), Path("missing"), Path("missing"), "none")

    def test_audit_flags_session_overlap_without_changing_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            data, output = Path(folder) / "data", Path(folder) / "results"
            data.mkdir()
            output.mkdir()
            images = pd.DataFrame([{"split": "train", "image": "a", "clip": "session(1)"},
                                   {"split": "test", "image": "b", "clip": "session(2)"}])
            objects = pd.DataFrame([{"split": s, "image": i, "obj_idx": 0, "cls": "pedestrian",
                                    "x": 10, "y": 20, "w": 30, "h": 40, "foot_x": 25, "foot_y": 60,
                                    "img_w": 100, "img_h": 100} for s, i in [("train", "a"), ("test", "b")]])
            images.to_csv(data / "images.csv", index=False)
            objects.to_csv(data / "objects.csv", index=False)
            road = objects.copy()
            road["foot_zone"] = "road"
            road.to_csv(data / "objects_road.csv", index=False)
            for s, i in [("train", "a"), ("test", "b")]:
                (data / "images" / s).mkdir(parents=True)
                (data / "labels" / s).mkdir(parents=True)
                (data / "images" / s / (i + ".jpg")).touch()
                (data / "labels" / s / (i + ".txt")).write_text("0 .25 .40 .30 .40")
            before = (data / "labels/test/b.txt").read_bytes()
            result = audit_dataset(data, output, ["pedestrian"])
            self.assertEqual(result["session_split_overlap_count"], 1)
            self.assertEqual(result["label_coordinate_error_gt_0_01_px"], 0)
            self.assertFalse(result["test_analysis_blocking"])
            self.assertEqual(before, (data / "labels/test/b.txt").read_bytes())


if __name__ == "__main__":
    unittest.main()
