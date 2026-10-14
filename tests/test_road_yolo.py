"""road_yolo checks with a randomly initialised YOLOv8s (no weights, images or GPU needed)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aihub"))
from road_yolo import MODES, RoadDetectionModel, RoadFormat, mask_path  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402


def build(mode):
    torch.manual_seed(0)
    base = DetectionModel("yolov8s.yaml", nc=5, verbose=False).eval()
    model = RoadDetectionModel("yolov8s.yaml", nc=5, verbose=False).add_road_module(mode)
    model.load_state_dict(base.state_dict(), strict=False)
    if mode == "concat":
        model.expand_first_conv()
    return base, model.eval()


class RoadYoloTests(unittest.TestCase):
    def test_mask_path(self):
        self.assertEqual(mask_path("/d/aihub_f/images/train/a_b.jpg"), "/d/aihub_f/road_masks/train/a_b.png")
        self.assertEqual(mask_path("C:\\d\\images\\test\\x.jpg"), "C:/d/road_masks/test/x.png")

    def test_every_mode_is_identity_at_init(self):
        x = torch.rand(2, 4, 128, 160)
        for mode in MODES:
            base, model = build(mode)
            with torch.no_grad():
                diff = (base(x[:, :3])[0] - model(x)[0]).abs().max().item()
            self.assertEqual(diff, 0.0, mode)

    def test_three_channel_input_is_accepted(self):
        for mode in MODES:
            _, model = build(mode)
            with torch.no_grad():
                model(torch.rand(1, 3, 64, 64))

    def test_gate_reacts_to_mask_only_in_road_modes(self):
        x = torch.rand(1, 4, 128, 160)
        y = x.clone()
        y[:, 3] = 1 - y[:, 3]
        for mode, should_change in (("attn", False), ("rga", True), ("rgab", True)):
            _, model = build(mode)
            for gate in model.road_gates:
                gate.gamma.data.fill_(1.0)
            with torch.no_grad():
                changed = not torch.equal(model(x)[0], model(y)[0])
            self.assertEqual(changed, should_change, mode)

    def test_format_flips_only_color_channels(self):
        formatter = RoadFormat.__new__(RoadFormat)
        formatter.bgr = 0.0
        img = np.stack([np.full((2, 2), v, np.uint8) for v in (1, 2, 3, 9)], axis=-1)
        out = formatter._format_img(img)
        self.assertEqual(out[:, 0, 0].tolist(), [3, 2, 1, 9])


if __name__ == "__main__":
    unittest.main()
