"""Road-Guided Attention (RGA) for YOLOv8 detection, plus the comparison variants.

Input: the RGB image and a road mask (pre-computed SegFormer mask, option A) travel
together as one 4-channel array so that every geometric augmentation (mosaic,
affine, flip, letterbox) moves them identically. Channel 4 is road probability:
road=255, everything else 0. Mosaic padding becomes 114 (an "unknown" value).

Modes (all trained identically; only the module differs):
    none    plain YOLOv8 (mask ignored)                      -> baseline
    concat  mask as 4th input channel to the first conv      -> "simple road input"
    attn    residual gate on P3/P4/P5, mask channels zeroed   -> same-capacity generic attention
    rga     residual gate conditioned on road probability    -> proposed
    rgab    rga + road-boundary channel                       -> proposed + boundary

Gate (per detection scale, residual, identity at initialisation):
    a  = tanh(Conv3x3(Conv3x3([Conv1x1(F), road, boundary])))
    F' = F * (1 + gamma * a),   gamma initialised to 0
"""
from __future__ import annotations

from copy import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.data.augment import Albumentations, Compose, Format, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.modules import Conv, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.patches import imread
from ultralytics.utils.torch_utils import de_parallel, initialize_weights

MODES = ("none", "concat", "attn", "rga", "rgab")
ROAD_VALUE = 1                    # value of "road" in make_road_masks.py output


# ----------------------------------------------------------------------------- data
def mask_path(image_path):
    """.../images/<split>/X.jpg -> .../road_masks/<split>/X.png"""
    head, sep, tail = str(image_path).replace("\\", "/").rpartition("/images/")
    if not sep:
        raise ValueError("Image path has no /images/ component: %s" % image_path)
    return head + "/road_masks/" + tail.rsplit(".", 1)[0] + ".png"


def load_road_channel(image_path, size_hw, close_px=0):
    """Road probability channel (uint8 0/255) resized to the image size."""
    mask = imread(mask_path(image_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path(image_path))
    road = (mask[..., 0] == ROAD_VALUE).astype(np.uint8) * 255
    if close_px > 0:
        # Control experiment: fill person/vehicle-shaped holes so the mask cannot act as an object hint.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)
    h, w = size_hw
    return cv2.resize(road, (w, h), interpolation=cv2.INTER_NEAREST)


class FirstThreeChannels:
    """Apply an RGB-only transform (HSV, Albumentations) to channels 0-2 and keep the rest."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, labels):
        img = labels["img"]
        if img.shape[-1] == 3:
            return self.transform(labels)
        extra = img[..., 3:].copy()
        labels["img"] = np.ascontiguousarray(img[..., :3])
        labels = self.transform(labels)
        labels["img"] = np.concatenate([labels["img"], extra], axis=-1)
        return labels


class RoadFormat(Format):
    """Format that flips BGR->RGB on the first three channels only (stock Format skips 4-ch images)."""

    def _format_img(self, img):
        img = img.transpose(2, 0, 1)
        if np.random.uniform(0, 1) > self.bgr:
            img = np.concatenate([img[2::-1], img[3:]], axis=0)
        return torch.from_numpy(np.ascontiguousarray(img))


class RoadYOLODataset(YOLODataset):
    """YOLODataset whose images carry the road mask as channel 4."""

    def __init__(self, *args, mask_close=0, **kwargs):
        self.mask_close = mask_close
        super().__init__(*args, **kwargs)

    def load_image(self, i, rect_mode=True):
        if self.ims[i] is not None:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]
        im, hw0, hw = super().load_image(i, rect_mode)
        if im.shape[-1] == 3:
            road = load_road_channel(self.im_files[i], im.shape[:2], self.mask_close)
            im = np.concatenate([im, road[..., None]], axis=-1)
            if self.augment and self.ims[i] is not None:      # keep the mosaic buffer consistent
                self.ims[i] = im
        return im, hw0, hw

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        items = []
        for t in transforms.transforms:
            if isinstance(t, (RandomHSV, Albumentations)):
                t = FirstThreeChannels(t)
            elif isinstance(t, Format):
                t = copy(t)
                t.__class__ = RoadFormat          # same settings, 4-channel-aware image formatting
            items.append(t)
        return Compose(items)


def build_road_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, mask_close=0):
    """Mirror of ultralytics.data.build.build_yolo_dataset for RoadYOLODataset."""
    return RoadYOLODataset(
        img_path=img_path, imgsz=cfg.imgsz, batch_size=batch, augment=mode == "train", hyp=cfg,
        rect=cfg.rect or rect, cache=cfg.cache or None, single_cls=cfg.single_cls or False,
        stride=int(stride), pad=0.0 if mode == "train" else 0.5, prefix=colorstr(f"{mode}: "),
        task=cfg.task, classes=cfg.classes, data=data,
        fraction=cfg.fraction if mode == "train" else 1.0, mask_close=mask_close)


# ---------------------------------------------------------------------------- model
class RoadGate(nn.Module):
    """Residual spatial gate on one detection scale. Identity while gamma == 0."""

    def __init__(self, channels, hidden=16):
        super().__init__()
        self.reduce = Conv(channels, hidden, 1)
        self.attn = nn.Sequential(Conv(hidden + 2, hidden, 3), nn.Conv2d(hidden, 1, 3, padding=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feature, road, boundary):
        a = torch.tanh(self.attn(torch.cat([self.reduce(feature), road, boundary], 1)))
        return feature * (1 + self.gamma * a)


class RoadDetectionModel(DetectionModel):
    """DetectionModel that accepts [B, 4, H, W] (RGB + road) and applies the chosen road module."""

    road_mode = "none"

    def add_road_module(self, mode):
        if mode not in MODES:
            raise ValueError("mode must be one of %s" % (MODES,))
        self.road_mode = mode
        if mode in ("attn", "rga", "rgab"):
            detect = self.model[-1]
            channels = [branch[0].conv.in_channels for branch in detect.cv2]
            self.road_gates = nn.ModuleList(RoadGate(c) for c in channels)
            initialize_weights(self.road_gates)
        return self

    def expand_first_conv(self):
        """concat mode: 3 -> 4 input channels; the new channel starts with zero weights."""
        conv = self.model[0].conv
        if conv.in_channels == 4:
            return self
        new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = conv.weight
        self.model[0].conv = new.to(conv.weight.device, conv.weight.dtype)
        return self

    def _road_inputs(self, mask, size, dtype):
        if mask is None or self.road_mode == "attn":
            zeros = torch.zeros((1, 1) + tuple(size), device=self.model[0].conv.weight.device, dtype=dtype)
            return zeros, zeros
        road = F.interpolate(mask.to(dtype), size=size, mode="area")
        if self.road_mode == "rgab":
            boundary = (road - F.avg_pool2d(road, 5, 1, 2, count_include_pad=False)).abs() * 4
        else:
            boundary = torch.zeros_like(road)
        return road, boundary

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        mode = self.road_mode
        mask = x[:, 3:4] if x.shape[1] > 3 else None
        if mode == "concat":
            if self.model[0].conv.in_channels == 4 and x.shape[1] == 3:
                x = torch.cat([x, torch.zeros_like(x[:, :1])], 1)
        else:
            x = x[:, :3]
        gates = getattr(self, "road_gates", None)
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if gates is not None and isinstance(m, Detect):
                gated = []
                for feature, gate in zip(x, gates):
                    road, boundary = self._road_inputs(mask, feature.shape[-2:], feature.dtype)
                    if road.shape[0] != feature.shape[0]:
                        road = road.expand(feature.shape[0], -1, -1, -1)
                        boundary = boundary.expand(feature.shape[0], -1, -1, -1)
                    gated.append(gate(feature, road, boundary))
                x = gated
            if profile:
                self._profile_one_layer(m, x, [])
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


# --------------------------------------------------------------------- train / val
class RoadValidator(DetectionValidator):
    mask_close = 0

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_road_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride,
                                  mask_close=self.mask_close)


class RoadTrainer(DetectionTrainer):
    """DetectionTrainer for RoadDetectionModel. Set road_mode / mask_close on the class before use."""

    road_mode = "rga"
    mask_close = 0

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_road_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val",
                                  stride=gs, mask_close=self.mask_close)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = RoadDetectionModel(cfg, nc=self.data["nc"], ch=3, verbose=verbose and RANK == -1)
        saved_mode = getattr(weights, "road_mode", None)
        if saved_mode not in (None, "none") and saved_mode != self.road_mode:
            raise ValueError("Checkpoint mode %s != requested %s" % (saved_mode, self.road_mode))
        model.add_road_module(self.road_mode)
        weights_in = weights.model[0].conv.in_channels if weights is not None else 3
        if self.road_mode == "concat" and weights_in == 4:
            model.expand_first_conv()                      # resuming a concat checkpoint
        if weights:
            model.load(weights)
        if self.road_mode == "concat":
            model.expand_first_conv()                      # from 3-ch pretrained weights
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = RoadValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                                  _callbacks=self.callbacks)
        validator.mask_close = self.mask_close
        return validator
