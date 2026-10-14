"""Synthetic tests for failure-mode labelling and session split assignment."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aihub"))
from analyze_failure_modes import classify_fn, classify_fp
from make_session_split import SPLITS, assign, score

PRED_COLUMNS = ["prediction_x1_original", "prediction_y1_original",
                "prediction_x2_original", "prediction_y2_original"]


def gt_row(image, box, status="FN", cls="pedestrian", confused=""):
    x, y, w, h = box
    return {"image": image, "cls": cls, "x": x, "y": y, "w": w, "h": h, "status": status,
            "confused_prediction_class": confused}


def pred_row(image, xyxy, status="FP", cls="pedestrian"):
    return {"image": image, "pred_class": cls, "status": status, **dict(zip(PRED_COLUMNS, xyxy))}


class FailureModeTests(unittest.TestCase):
    def test_fn_modes(self):
        gts = pd.DataFrame([
            gt_row("a", (0, 0, 10, 20)),                        # partial unmatched box -> localization
            gt_row("a", (100, 0, 10, 20)),                      # nothing nearby
            gt_row("a", (200, 0, 10, 20), confused="car"),      # wrong class wins first
        ])
        preds = pd.DataFrame([pred_row("a", (5, 0, 15, 20))])   # IoU 1/3 with first GT
        low = {"a": {"boxes": np.array([[100, 0, 110, 20.]]), "scores": np.array([.05]),
                     "classes": np.array([0])}}
        modes = classify_fn(gts, preds, low)["fn_mode"].tolist()
        self.assertEqual(modes, ["localization", "below_threshold", "wrong_class"])
        self.assertEqual(classify_fn(gts, preds, None)["fn_mode"].iloc[1], "no_candidate_or_low_score")

    def test_fp_modes(self):
        gts = pd.DataFrame([gt_row("a", (0, 0, 10, 20), status="TP"),
                            gt_row("a", (100, 0, 40, 40), status="TP", cls="two_wheeler")])
        fps = pd.DataFrame([pred_row("a", (0, 0, 10, 20)), pred_row("a", (100, 0, 140, 40)),
                            pred_row("a", (500, 500, 510, 520)), pred_row("b", (0, 0, 5, 5))])
        modes = classify_fp(fps, gts)["fp_mode"].tolist()
        self.assertEqual(modes, ["duplicate", "class_confusion", "background", "background"])


class SessionSplitTests(unittest.TestCase):
    def sessions(self):
        rng = np.random.default_rng(0)
        rows = [{"images": int(rng.integers(5, 40)), "pedestrians": int(rng.integers(0, 60)),
                 "time": t, "weather": w}
                for t, w in [("day", "clear")] * 40 + [("night", "foggy")] * 10 + [("erh", "rainy")] * 20]
        frame = pd.DataFrame(rows, index=["s%d" % i for i in range(len(rows))])
        for column in ("rainy", "foggy"):
            frame[column] = np.where(frame["weather"] == column, frame["images"], 0)
        frame["night"] = np.where(frame["time"] == "night", frame["images"], 0)
        return frame

    def test_every_session_gets_one_split_and_each_stratum_reaches_test(self):
        sessions = self.sessions()
        ratios = dict(zip(SPLITS, (.7, .15, .15)))
        split = assign(sessions, ratios, seed=1)
        self.assertEqual(set(split.index), set(sessions.index))
        self.assertTrue(set(split) <= set(SPLITS))
        for _, stratum in sessions.assign(split=split).groupby("time"):
            self.assertIn("test", set(stratum["split"]))
        self.assertLess(score(sessions, split, ratios), .1)


if __name__ == "__main__":
    unittest.main()
