"""RCR-Head checks on synthetic data (no AI-Hub images, labels or trained weights needed)."""
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aihub"))
import rcr_train  # noqa: E402
from pilot_risk_analysis import v1_rule  # noqa: E402
from rcr_head import (DIRECTION_VALUES, POSITION_VALUES, RASTER_HW, RCRConfig, RCRHead, flip_inputs,  # noqa: E402
                      make_geometry, make_raster, rcr_loss, v1_risk)


def synthetic_cache(path, n=240, seed=0, channels=(8, 8, 8)):
    """Features whose road raster decides the risk: foot on road near the centre -> risky, etc."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((360, 640), np.uint8)
    mask[180:, 160:480] = 1                       # road: lower centre
    mask[180:, :120] = 2                          # sidewalk: lower left
    rows, labels = {k: [] for k in ("raster", "geometry", "p3", "p4", "p5", "p3_flip", "p4_flip", "p5_flip")}, []
    for i in range(n):
        fx, fy = rng.uniform(0.05, 0.95), rng.uniform(0.55, 0.98)
        hgt = 0.1 + 0.3 * (fy - 0.5)
        box = (fx - hgt / 6, fy - hgt, fx + hgt / 6, fy)
        on_road = mask[int(fy * 359), int(fx * 639)] == 1
        central = abs(fx - 0.5) < 0.12
        position = "1" if on_road and central else "2" if on_road and abs(fx - 0.5) < 0.2 else "3" if on_road else "4"
        direction = DIRECTION_VALUES[rng.integers(0, 4)]
        rows["raster"].append(make_raster(mask, box, (fx, fy)).astype(np.float16))
        rows["geometry"].append(make_geometry(box, (fx, fy)))
        for k, c in zip(("p3", "p4", "p5"), channels):
            a = rng.normal(0, 1, (c, 7, 7)).astype(np.float16)
            rows[k].append(a)
            rows[k + "_flip"].append(a[..., ::-1].copy())
        labels.append(dict(id=f"img{i}__0", position=position, direction=direction, crosswalk=0,
                           occluded=int(rng.random() < .1), boundary=0, excluded=0, mask_error=0, note='""'))
    np.savez_compressed(path, id=np.array([l["id"] for l in labels]),
                        session=np.array([f"s{i % 40}" for i in range(n)]),
                        has_mask=np.ones(n, bool), **{k: np.stack(v) for k, v in rows.items()})
    return pd.DataFrame(labels)


class RuleTests(unittest.TestCase):
    def test_v1_risk_matches_pilot_rule(self):
        for p in POSITION_VALUES:
            for d in DIRECTION_VALUES:
                for occ in (0, 1):
                    row = Namespace(position=p, direction=d, occluded=occ)
                    self.assertEqual(v1_risk(p, d, occ), v1_rule(row) - 1, (p, d, occ))


class InputTests(unittest.TestCase):
    def test_raster_channels(self):
        mask = np.zeros((360, 640), np.uint8)
        mask[:, :320] = 1
        r = make_raster(mask, (0.6, 0.5, 0.7, 0.9), (0.65, 0.9))
        self.assertEqual(r.shape, (4,) + RASTER_HW)
        self.assertAlmostEqual(float(r[0].mean()), 0.5, places=2)       # half the image is road
        self.assertEqual(float(r[1].sum()), 0.0)                        # no sidewalk
        y, x = np.unravel_index(r[3].argmax(), r[3].shape)
        self.assertLess(abs((x + .5) / RASTER_HW[1] - 0.65), 0.02)
        self.assertLess(abs((y + .5) / RASTER_HW[0] - 0.9), 0.02)
        self.assertEqual(float(r[2][int(0.7 * 64), int(0.65 * 112)]), 1.0)
        self.assertEqual(make_raster(None, (0, 0, .1, .1), (.05, .1))[:2].sum(), 0)

    def test_flip_matches_flipped_scene(self):
        mask = np.zeros((360, 640), np.uint8)
        mask[200:, 100:400] = 1
        box, foot = (0.2, 0.4, 0.3, 0.8), (0.25, 0.8)
        r, g, _ = flip_inputs(make_raster(mask, box, foot), make_geometry(box, foot), None)
        fbox, ffoot = (0.7, 0.4, 0.8, 0.8), (0.75, 0.8)
        np.testing.assert_allclose(r, make_raster(mask[:, ::-1], fbox, ffoot), atol=1e-6)
        np.testing.assert_allclose(g, make_geometry(fbox, ffoot), atol=1e-6)


class ModelTests(unittest.TestCase):
    def inputs(self, b=4, ch=(8, 8, 8)):
        return dict(appearance=[torch.randn(b, c, 7, 7) for c in ch], raster=torch.rand(b, 4, *RASTER_HW),
                    geometry=torch.rand(b, 6))

    def test_every_variant_runs_and_backprops(self):
        x = self.inputs()
        target = dict(risk=torch.tensor([0, 1, 2, 2]), position=torch.tensor([0, -1, 2, 3]),
                      direction=torch.tensor([1, 2, -1, 0]),
                      binary=torch.tensor([[0., 1, float("nan")]] * 4))
        for name, kw in rcr_train.VARIANTS.items():
            model = RCRHead(RCRConfig(appearance_channels=(8, 8, 8), **kw))
            out = model(**x)
            self.assertEqual(out["risk"].shape, (4, 3), name)
            self.assertEqual("position" in out, kw.get("condition_heads", True), name)
            loss = rcr_loss(out, target)
            self.assertTrue(torch.isfinite(loss), name)
            loss.backward()

    def test_film_starts_as_identity(self):
        torch.manual_seed(0)
        film = RCRHead(RCRConfig(appearance_channels=(8, 8, 8), fusion="film")).eval()
        x = self.inputs()
        a = film.app(x["appearance"])
        ctx = torch.cat([film.raster(x["raster"]), film.geom(x["geometry"])], 1)
        gamma, beta = film.film(ctx).chunk(2, 1)
        self.assertEqual(float(gamma.abs().max()), 0.0)
        self.assertEqual(float(beta.abs().max()), 0.0)
        self.assertEqual(a.shape, gamma.shape)


class TrainTests(unittest.TestCase):
    def test_cross_validation_learns_road_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = synthetic_cache(Path(tmp) / "f.npz")
            labels.to_csv(Path(tmp) / "attributes.csv", index=False)
            args = Namespace(features=Path(tmp) / "f.npz", labels=Path(tmp) / "attributes.csv", out=None,
                             only=["G (geometry)", "R (road raster)", "A (appearance)", "A+R+G film + conditions"],
                             folds=3, seeds=1, epochs=40, lr=3e-3, weight_decay=1e-2, batch_size=32, dim=32,
                             dropout=0.1, cond_weight=0.5, device="cpu")
            result = rcr_train.run(args)
        v = result["variants"]
        f1 = {k: v[k]["risk"]["macro_f1"]["mean"] for k in v}
        # Appearance here is noise; the road raster carries the rule.
        self.assertLess(f1["A (appearance)"], 0.5)
        self.assertGreater(f1["R (road raster)"], 0.6)
        self.assertGreater(f1["A+R+G film + conditions"], f1["A (appearance)"] + 0.15)
        self.assertIn("risk_via_conditions", v["A+R+G film + conditions"])
        self.assertGreater(v["A+R+G film + conditions"]["conditions"]["position_acc"]["mean"], 0.5)


class FeatureExtractionTests(unittest.TestCase):
    def test_extract_with_random_yolo(self):
        import rcr_features
        from ultralytics import YOLO
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            view = tmp / "view"
            (view / "images" / "train").mkdir(parents=True)
            (view / "road_masks" / "train").mkdir(parents=True)
            rng = np.random.default_rng(0)
            rows = []
            for i in range(2):
                cv2.imwrite(str(view / "images" / "train" / f"im{i}.jpg"), rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8))
                m = np.zeros((360, 640), np.uint8)
                m[200:] = 1
                cv2.imwrite(str(view / "road_masks" / "train" / f"im{i}.png"), m)
                for j in range(2):
                    rows.append(dict(id=f"im{i}__{j}", image=f"im{i}", obj_idx=j, split="train", clip=f"c{i}(1)",
                                     x=300 + 400 * j, y=600, w=80, h=200, foot_x=340 + 400 * j, foot_y=800,
                                     img_w=1920, img_h=1080))
            (tmp / "set").mkdir()
            pd.DataFrame(rows).to_csv(tmp / "set" / "items.csv", index=False)
            weights = tmp / "rand.pt"
            YOLO("yolov8s.yaml").save(str(weights))
            out = tmp / "f.npz"
            argv = sys.argv
            sys.argv = ["x", "--set", str(tmp / "set"), "--view", str(view), "--weights", str(weights),
                        "--out", str(out), "--device", "cpu"]
            try:
                rcr_features.main()
            finally:
                sys.argv = argv
            with np.load(out) as f:
                z = {k: f[k] for k in f.files}
            self.assertEqual([z[k].shape[1] for k in ("p3", "p4", "p5")], [128, 256, 512])
            self.assertEqual(z["p3"].shape, (4, 128, 7, 7))
            self.assertEqual(z["p3_flip"].shape, (4, 128, 7, 7))
            self.assertEqual(z["raster"].shape, (4, 4) + RASTER_HW)
            self.assertEqual(list(z["session"]), ["c0", "c0", "c1", "c1"])
            self.assertTrue(np.isfinite(z["p5"].astype(np.float32)).all())
            # foot at y=800/1080 is well below the road line (200/360) -> road under the foot
            self.assertGreater(float(z["raster"][0, 0, int(800 / 1080 * 64), int(340 / 1920 * 112)]), 0.5)


if __name__ == "__main__":
    unittest.main()
