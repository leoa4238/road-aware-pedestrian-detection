"""Position suggestions for the labeling page, and the page's review flow (via Node, if installed)."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aihub"))
import rcr_train  # noqa: E402
from make_label_set import suggest_position, write_attribute_page  # noqa: E402


def road_mask():
    m = np.zeros((360, 640), np.uint8)
    m[200:, 100:540] = 1          # road
    m[200:, :100] = 2             # sidewalk on the left
    return m


class SuggestTests(unittest.TestCase):
    def test_positions(self):
        m = road_mask()
        self.assertEqual(suggest_position(0.50, 0.95, 0.3, m)[0], "1")    # centre, near: in path
        self.assertEqual(suggest_position(0.70, 0.95, 0.3, m)[0], "2")    # just outside the band (|lat| ~1.2)
        self.assertEqual(suggest_position(0.82, 0.95, 0.3, m)[0], "3")    # road, well off the path
        self.assertEqual(suggest_position(0.05, 0.95, 0.3, m)[0], "4")    # sidewalk
        self.assertEqual(suggest_position(0.50, 0.30, 0.1, m)[0], "4")    # above the road (not road)
        self.assertEqual(suggest_position(0.5, 0.9, 0.3, None), ("", 0.0, "마스크 없음"))

    def test_confidence_is_low_near_thresholds(self):
        m = road_mask()
        clear = suggest_position(0.50, 0.95, 0.3, m)[1]
        border = suggest_position(0.653, 0.95, 0.3, m)[1]             # |lateral| close to 0.85
        self.assertGreater(clear, 0.8)
        self.assertLess(border, 0.5)


class PageTests(unittest.TestCase):
    def make_set(self, tmp):
        view, out = tmp / "view", tmp / "set"
        (view / "road_masks" / "train").mkdir(parents=True)
        out.mkdir()
        cv2.imwrite(str(view / "road_masks" / "train" / "im0.png"), road_mask())
        feet = [0.653, 0.50, 0.05, 0.82]                               # border case first in file order
        rows = [dict(id=f"im0__{k}", image="im0", obj_idx=k, split="train", clip="c(1)", x=fx * 1920 - 40,
                     y=0.95 * 1080 - 300, w=80, h=300, foot_x=fx * 1920, foot_y=0.95 * 1080, img_w=1920, img_h=1080)
                for k, fx in enumerate(feet)]
        pd.DataFrame(rows).to_csv(out / "items.csv", index=False)
        return view, out

    def test_page_orders_by_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            view, out = self.make_set(Path(tmp))
            write_attribute_page(out, view)
            html = (out / "index_attr.html").read_text(encoding="utf-8")
        items = json.loads(html.split("const ITEMS=", 1)[1].split(", KEY=", 1)[0])
        self.assertEqual(len(items), 4)
        confs = [t["conf"] for t in items]
        self.assertEqual(confs, sorted(confs, reverse=True))
        self.assertEqual(items[-1]["id"], "im0__0")                    # the border case goes last
        self.assertEqual({t["id"]: t["sug"] for t in items}["im0__2"], "4")

    @unittest.skipIf(shutil.which("node") is None, "node not installed")
    def test_review_flow_in_browser_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            view, out = self.make_set(Path(tmp))
            write_attribute_page(out, view)
            res = subprocess.run(["node", str(ROOT / "tests" / "label_page_harness.js"), str(out / "index_attr.html")],
                                 capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(res.returncode, 0, res.stderr)
        r = json.loads(res.stdout)
        self.assertEqual(r["first"], "im0__1")                          # clearest suggestion first
        self.assertEqual(r["prefilled"], "1")
        self.assertIn("위치 자동 제안", r["sugline"])
        self.assertTrue(r["blocked"])                                   # Space needs a direction
        self.assertEqual(r["accepted"], {"checked": True, "sug_accepted": True, "pos": "1"})
        self.assertEqual(r["changed"], {"checked": True, "sug_accepted": False, "pos": "4"})
        rows = [l.split(",") for l in r["rows"][1:]]
        checked = {x[0] for x in rows if x[8] == "1"}
        self.assertEqual(len(checked), 2)
        for x in rows:                                                  # unchecked rows carry no label
            if x[0] not in checked:
                self.assertEqual((x[1], x[2]), ("", ""))
        self.assertEqual(r["draftLoaded"], 2)                           # the two reviewed, different ones kept
        self.assertEqual(r["afterDraft"], {"firstKept": "1", "firstChecked": True,
                                           "thirdDraft": True, "thirdChecked": False})
        self.assertTrue(r["legacyChecked"])
        self.assertEqual(r["toggleOff"]["after"], None)
        self.assertEqual(r["toggleOff"]["stored"], "off")

    def test_rcr_train_uses_checked_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.csv"
            pd.DataFrame([dict(id="a", position="1", direction="q", crosswalk=0, occluded=0, boundary=0, excluded=0,
                               mask_error=0, checked=1, from_draft=0, sug_accepted=1, pos_suggested="1", note='""'),
                          dict(id="b", position="", direction="", crosswalk=0, occluded=0, boundary=0, excluded=0,
                               mask_error=0, checked=0, from_draft=1, sug_accepted=0, pos_suggested="", note='""')]
                         ).to_csv(path, index=False)
            labels = rcr_train.load_labels(path)
        self.assertEqual(labels["id"].tolist(), ["a"])
        self.assertEqual(labels["risk"].tolist(), [2])


if __name__ == "__main__":
    unittest.main()
