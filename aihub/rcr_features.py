"""Cache RCR-Head inputs for a labeled pedestrian set: detector RoI features, road raster, geometry.

The detector (YOLOv8s trained on session_v1, e.g. best.pt) is frozen, so its neck features are
computed once here and the head is trained on the cache (rcr_train.py) in seconds on a CPU.

Inputs
    --set     folder with items.csv (make_label_set.py): id, image, split, clip, x, y, w, h,
              foot_x, foot_y, img_w, img_h (original-pixel coordinates)
    --view    data view with images/<split>/<image>.jpg and road_masks/<split>/<image>.png
    --weights detector weights

Output: one .npz with, per item, RoI features of P3/P4/P5 (float16), the road raster, the geometry
vector and the same features for the horizontally flipped image (training augmentation).

Example:
    python aihub/rcr_features.py --set data/labeling/pilot_v0 --view data/views/session_v1 \
        --weights D:/nonmoon/best.pt --out data/rcr/pilot_v0_features.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import roi_align

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_analysis_core import read_image, session_id  # noqa: E402
from rcr_head import make_geometry, make_raster  # noqa: E402

STRIDES = (8, 16, 32)


def preprocess(frame_bgr, width=640, pad_to=32):
    """BGR frame -> [1, 3, H, W] RGB float tensor, resized to `width`, padded bottom/right with 114.
    Returns the tensor and the resize scale (network pixels per frame pixel)."""
    scale = width / frame_bgr.shape[1]
    h = round(frame_bgr.shape[0] * scale)
    img = cv2.resize(frame_bgr, (width, h), interpolation=cv2.INTER_AREA)
    ph = int(np.ceil(h / pad_to) * pad_to)
    canvas = np.full((ph, width, 3), 114, np.uint8)
    canvas[:h] = img
    x = torch.from_numpy(np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
    return x, scale


class NeckFeatures:
    """Wraps an ultralytics DetectionModel and returns the inputs of its Detect layer (P3, P4, P5)."""

    def __init__(self, model):
        self.model = model.eval()
        self._feats = None
        self.model.model[-1].register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self._feats = [t.detach() for t in inputs[0]]

    @torch.no_grad()
    def __call__(self, x):
        self.model(x)
        return self._feats


def load_detector(weights, device):
    from ultralytics import YOLO
    model = YOLO(str(weights)).model.float().to(device)
    return NeckFeatures(model)


def roi_features(feats, boxes_xyxy_net, roi_size=7):
    """feats: list of [1, C, h, w]; boxes in network pixels [N, 4] -> list of [N, C, s, s] float16 arrays."""
    rois = torch.cat([torch.zeros(len(boxes_xyxy_net), 1), torch.as_tensor(boxes_xyxy_net, dtype=torch.float32)], 1)
    out = []
    for f, stride in zip(feats, STRIDES):
        r = roi_align(f.float(), rois.to(f.device), output_size=roi_size, spatial_scale=1 / stride,
                      sampling_ratio=2, aligned=True)
        out.append(r.cpu().numpy().astype(np.float16))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--view", type=Path, required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=640, help="network input width (YOLO imgsz)")
    ap.add_argument("--no-flip", action="store_true", help="skip flipped-image features")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    items = pd.read_csv(args.set / "items.csv")
    neck = load_detector(args.weights, args.device)
    keys = ["p3", "p4", "p5"]
    store = {k: [] for k in keys + [k + "_flip" for k in keys] + ["raster", "geometry", "has_mask"]}
    missing = []
    for (split, image), group in items.groupby(["split", "image"], sort=False):
        frame = read_image(args.view / "images" / split / f"{image}.jpg")
        if frame is None:
            raise FileNotFoundError(args.view / "images" / split / f"{image}.jpg")
        mask = read_image(args.view / "road_masks" / split / f"{image}.png", cv2.IMREAD_GRAYSCALE)
        if mask is None:
            missing.append(image)
        img_w, img_h = group["img_w"].iloc[0], group["img_h"].iloc[0]
        x, scale = preprocess(frame, args.width)
        s = scale * frame.shape[1] / img_w                       # network px per original px
        boxes = np.stack([group["x"], group["y"], group["x"] + group["w"], group["y"] + group["h"]], 1) * s
        feats = neck(x.to(args.device))
        for k, arr in zip(keys, roi_features(feats, boxes)):
            store[k].append(arr)
        if not args.no_flip:
            flipped = boxes.copy()
            flipped[:, [0, 2]] = args.width - boxes[:, [2, 0]]
            xf, _ = preprocess(frame[:, ::-1], args.width)
            for k, arr in zip(keys, roi_features(neck(xf.to(args.device)), flipped)):
                store[k + "_flip"].append(arr)
        for r in group.itertuples():
            box = (r.x / img_w, r.y / img_h, (r.x + r.w) / img_w, (r.y + r.h) / img_h)
            foot = (r.foot_x / img_w, r.foot_y / img_h)
            store["raster"].append(make_raster(mask, box, foot).astype(np.float16)[None])
            store["geometry"].append(make_geometry(box, foot)[None])
            store["has_mask"].append(np.array([mask is not None]))

    order = [i for _, g in items.groupby(["split", "image"], sort=False) for i in g.index]
    arrays = {k: np.concatenate(v) for k, v in store.items() if v}
    meta = items.loc[order]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, id=np.array(meta["id"].astype(str).tolist(), dtype=str),
                        session=np.array(meta["clip"].map(session_id).astype(str).tolist(), dtype=str), **arrays)
    print(f"[saved] {args.out}: {len(meta)} pedestrians, {meta['image'].nunique()} images, "
          f"features {[arrays[k].shape[1] for k in keys]}, flip={'p3_flip' in arrays}, no mask: {len(missing)}")


if __name__ == "__main__":
    main()
