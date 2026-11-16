"""RCR-Head: Road-Context Risk head for pedestrians found by a (frozen) YOLO detector.

For each pedestrian box it predicts the risk level (safe / caution / risky) and, as auxiliary
outputs, the observable conditions recorded by the labeling criteria v1
(docs/RISK_LABEL_CRITERIA_DRAFT.md): position w.r.t. the ego path, motion direction,
occlusion, sidewalk-road boundary, crosswalk.

Three inputs per pedestrian (each can be switched off for ablations):

    appearance  RoIAlign of the detector's neck features (P3/P4/P5) on the box
                -> body orientation / walking posture, without a separate pose model
    raster      whole-scene road-context raster: road, sidewalk, the pedestrian's box and foot
                -> where the person stands in the road layout, incl. curved roads (no fixed path band)
    geometry    foot position, box size, lateral offset from the image centre

Fusion (`fusion`):
    concat      [appearance, road context] -> MLP                     (comparison)
    film        appearance * (1 + gamma(road)) + beta(road)           (proposed: road-conditioned)
                gamma, beta start at 0, so the head starts as "appearance only" and learns how the
                same posture means different risk in different road contexts.

The detector stays frozen; features are cached once (rcr_features.py) and the head is trained
on the cache (rcr_train.py). Everything here is pure torch / numpy (no ultralytics import).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- labels
RISK_NAMES = ("safe", "caution", "risky")               # class index 0, 1, 2
POSITION_VALUES = ("1", "2", "3", "4")                  # in path / path edge / roadway off path / pedestrian space
DIRECTION_VALUES = ("q", "w", "e", "r")                 # into path / parallel / away / standing-unknown
BINARY_CONDITIONS = ("occluded", "boundary", "crosswalk")


def v1_risk(position, direction, occluded):
    """Risk rule v1 (docs/RISK_LABEL_CRITERIA_DRAFT.md), returned as 0 safe / 1 caution / 2 risky.

    Same rule as pilot_risk_analysis.v1_rule (which returns 1/2/3).
    """
    p, d = str(position), str(direction).lower()
    if p == "1" or (p == "2" and d == "q") or (bool(occluded) and d == "q"):
        return 2
    if p == "2" or (p == "3" and d == "q"):
        return 1
    return 0


# ---------------------------------------------------------------------------- inputs
RASTER_HW = (64, 112)            # ~16:9, like the 640x360 masks
RASTER_CHANNELS = ("road", "sidewalk", "person_box", "foot")
GEOMETRY_NAMES = ("foot_x", "foot_y", "box_h", "box_w", "lateral", "abs_lateral")
ROAD, SIDEWALK = 1, 2            # values in make_road_masks.py output


def make_raster(mask, box_xyxy_norm, foot_xy_norm, hw=RASTER_HW):
    """Road-context raster [4, H, W] float32 in [0, 1].

    mask: uint8 road mask (0 other, 1 road, 2 sidewalk) at any resolution, or None (all zeros).
    box_xyxy_norm, foot_xy_norm: in [0, 1] image coordinates.
    """
    h, w = hw
    out = np.zeros((len(RASTER_CHANNELS), h, w), np.float32)
    if mask is not None:
        # area interpolation of each class = fraction of that class per raster cell
        for k, value in enumerate((ROAD, SIDEWALK)):
            out[k] = cv2.resize((mask == value).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = box_xyxy_norm
    c0, r0 = int(np.floor(np.clip(x0, 0, 1) * w)), int(np.floor(np.clip(y0, 0, 1) * h))
    c1, r1 = int(np.ceil(np.clip(x1, 0, 1) * w)), int(np.ceil(np.clip(y1, 0, 1) * h))
    out[2, r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)] = 1.0
    fx, fy = foot_xy_norm
    yy, xx = np.mgrid[0:h, 0:w]
    sigma = max(1.0, 0.15 * (r1 - r0))      # scales with the person, so near and far feet look alike
    out[3] = np.exp(-((xx + .5 - fx * w) ** 2 + (yy + .5 - fy * h) ** 2) / (2 * sigma ** 2))
    return out


def make_geometry(box_xyxy_norm, foot_xy_norm):
    x0, y0, x1, y1 = box_xyxy_norm
    fx, fy = foot_xy_norm
    lateral = fx - 0.5
    return np.array([fx, fy, y1 - y0, x1 - x0, lateral, abs(lateral)], np.float32)


def flip_inputs(raster, geometry, appearance):
    """Horizontal flip of one sample. Position/direction labels are relative to the ego path,
    so they do not change under a flip."""
    raster = raster[..., ::-1].copy()
    g = geometry.copy()
    g[0] = 1 - g[0]
    g[4] = -g[4]
    appearance = None if appearance is None else [a[..., ::-1].copy() for a in appearance]
    return raster, g, appearance


# ----------------------------------------------------------------------------- model
@dataclass
class RCRConfig:
    use_appearance: bool = True
    use_raster: bool = True
    use_geometry: bool = True
    fusion: str = "film"                 # "film" | "concat"
    condition_heads: bool = True
    appearance_channels: tuple = (128, 256, 512)   # YOLOv8s P3, P4, P5
    roi_size: int = 7
    dim: int = 128
    dropout: float = 0.3
    extra: dict = field(default_factory=dict)

    def name(self):
        parts = [p for p, on in (("A", self.use_appearance), ("R", self.use_raster), ("G", self.use_geometry)) if on]
        tag = "+".join(parts)
        if self.use_appearance and (self.use_raster or self.use_geometry):
            tag += "-" + self.fusion
        if self.condition_heads:
            tag += "-cond"
        return tag


class AppearanceEncoder(nn.Module):
    """RoI features of P3/P4/P5 -> one vector. Each level is projected to the same width first,
    so the head does not depend on which level the box size favours."""

    def __init__(self, channels, dim, roi_size):
        super().__init__()
        self.proj = nn.ModuleList(nn.Sequential(nn.Conv2d(c, 64, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU())
                                  for c in channels)
        self.body = nn.Sequential(
            nn.Conv2d(64 * len(channels), dim, 3, padding=1, bias=False), nn.BatchNorm2d(dim), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, levels):
        return self.body(torch.cat([p(x) for p, x in zip(self.proj, levels)], 1))


class RasterEncoder(nn.Module):
    def __init__(self, dim, in_ch=len(RASTER_CHANNELS)):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.SiLU(),
                                 nn.MaxPool2d(2))
        # 64x112 -> 32x56 -> 16x28 -> 8x14 -> 4x7; the pooled grid keeps left/right/near/far layout
        self.body = nn.Sequential(block(in_ch, 16), block(16, 32), block(32, 32), block(32, 32),
                                  nn.Flatten(), nn.Linear(32 * 4 * 7, dim), nn.SiLU())

    def forward(self, raster):
        return self.body(raster)


class RCRHead(nn.Module):
    def __init__(self, cfg: RCRConfig):
        super().__init__()
        if not (cfg.use_appearance or cfg.use_raster or cfg.use_geometry):
            raise ValueError("at least one input must be on")
        if cfg.fusion not in ("film", "concat"):
            raise ValueError("fusion must be 'film' or 'concat'")
        self.cfg = cfg
        d = cfg.dim
        self.app = AppearanceEncoder(cfg.appearance_channels, d, cfg.roi_size) if cfg.use_appearance else None
        self.raster = RasterEncoder(d) if cfg.use_raster else None
        self.geom = nn.Sequential(nn.Linear(len(GEOMETRY_NAMES), 32), nn.SiLU()) if cfg.use_geometry else None
        ctx_dim = (d if cfg.use_raster else 0) + (32 if cfg.use_geometry else 0)

        self.film = None
        if cfg.use_appearance and ctx_dim and cfg.fusion == "film":
            self.film = nn.Linear(ctx_dim, 2 * d)
            nn.init.zeros_(self.film.weight)          # identity at init: appearance only
            nn.init.zeros_(self.film.bias)
            fused_dim = d + ctx_dim                    # context also kept as its own evidence
        else:
            fused_dim = (d if cfg.use_appearance else 0) + ctx_dim
        self.trunk = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(fused_dim, d), nn.SiLU(), nn.Dropout(cfg.dropout))
        self.risk = nn.Linear(d, len(RISK_NAMES))
        if cfg.condition_heads:
            self.position = nn.Linear(d, len(POSITION_VALUES))
            self.direction = nn.Linear(d, len(DIRECTION_VALUES))
            self.binary = nn.Linear(d, len(BINARY_CONDITIONS))

    def forward(self, appearance=None, raster=None, geometry=None):
        ctx = []
        if self.raster is not None:
            ctx.append(self.raster(raster))
        if self.geom is not None:
            ctx.append(self.geom(geometry))
        ctx = torch.cat(ctx, 1) if ctx else None
        a = self.app(appearance) if self.app is not None else None

        if a is not None and self.film is not None:
            gamma, beta = self.film(ctx).chunk(2, 1)
            fused = torch.cat([a * (1 + gamma) + beta, ctx], 1)
        else:
            fused = torch.cat([t for t in (a, ctx) if t is not None], 1)
        z = self.trunk(fused)
        out = {"risk": self.risk(z)}
        if self.cfg.condition_heads:
            out.update(position=self.position(z), direction=self.direction(z), binary=self.binary(z))
        return out


def rcr_loss(out, target, class_weight=None, cond_weight=0.5):
    """target: dict of tensors. risk [B] long; position/direction [B] long with -1 = missing;
    binary [B, 3] float with nan = missing."""
    loss = F.cross_entropy(out["risk"], target["risk"], weight=class_weight)
    if "position" in out and cond_weight > 0:
        cond = 0.0
        for key in ("position", "direction"):
            ok = target[key] >= 0
            if ok.any():
                cond = cond + F.cross_entropy(out[key][ok], target[key][ok])
        ok = ~torch.isnan(target["binary"])
        if ok.any():
            cond = cond + F.binary_cross_entropy_with_logits(out["binary"][ok], target["binary"][ok])
        loss = loss + cond_weight * cond
    return loss
