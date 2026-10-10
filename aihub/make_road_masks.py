"""prepare_dataset.py 출력 이미지에 SegFormer(Cityscapes) 도로/보도 마스크를 만들고,
보행자 발 위치가 도로 위인지, 도로까지 거리가 얼마인지 objects_road.csv로 기록한다.

마스크 값: 0=기타, 1=road, 2=sidewalk (Cityscapes train id 0, 1)

예시:
    python aihub/make_road_masks.py --data C:/work/data/aihub_f
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
ROAD, SIDEWALK = 0, 1
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img_bgr: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(((img - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))


@torch.inference_mode()
def segment_all(paths: list[Path], data: Path, mask_width: int, batch: int, infer_size: tuple[int, int]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, use_safetensors=True).to(device).eval()
    if device == "cuda":
        model.half()
    todo = [p for p in paths if not mask_path_of(p, data).exists()]
    for i in tqdm(range(0, len(todo), batch), desc="segformer"):
        chunk = todo[i:i + batch]
        imgs = [cv2.imread(str(p)) for p in chunk]
        x = torch.stack([to_tensor(im, infer_size) for im in imgs]).to(device)
        logits = model(pixel_values=x.half() if device == "cuda" else x).logits
        for p, im, lg in zip(chunk, imgs, logits):
            mh = round(im.shape[0] * mask_width / im.shape[1])
            pred = torch.nn.functional.interpolate(lg[None].float(), size=(mh, mask_width), mode="bilinear").argmax(1)[0]
            pred = pred.cpu().numpy()
            mask = np.zeros(pred.shape, dtype=np.uint8)
            mask[pred == ROAD] = 1
            mask[pred == SIDEWALK] = 2
            out = mask_path_of(p, data)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), mask)


def mask_path_of(image_path: Path, data: Path) -> Path:
    return data / "road_masks" / image_path.parent.name / f"{image_path.stem}.png"


def pedestrian_road_relation(data: Path) -> pd.DataFrame:
    objects = pd.read_csv(data / "objects.csv")
    peds = objects[objects["cls"] == "pedestrian"].copy()
    zone, dist, road_ratio = [], [], {}
    for (split, image), group in tqdm(peds.groupby(["split", "image"]), desc="road relation"):
        mask = cv2.imread(str(data / "road_masks" / split / f"{image}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            zone += ["unknown"] * len(group)
            dist += [np.nan] * len(group)
            continue
        road = (mask == 1).astype(np.uint8)
        road_ratio[image] = float(road.mean())
        # 도로가 아닌 픽셀에서 가장 가까운 도로 픽셀까지 거리(마스크 px)
        dt = cv2.distanceTransform(1 - road, cv2.DIST_L2, 3) if road.any() else None
        sx, sy = mask.shape[1] / group["img_w"].iloc[0], mask.shape[0] / group["img_h"].iloc[0]
        for _, o in group.iterrows():
            fx = int(np.clip(o["foot_x"] * sx, 0, mask.shape[1] - 1))
            fy = int(np.clip(o["foot_y"] * sy, 0, mask.shape[0] - 1))
            value = mask[fy, fx]
            zone.append({1: "road", 2: "sidewalk"}.get(int(value), "other"))
            # 거리는 보행자 키(bbox 높이) 대비 비율로 정규화해 원근 차이를 줄인다
            d = float(dt[fy, fx]) / max(o["h"] * sy, 1.0) if dt is not None else np.nan
            dist.append(d)
    peds["foot_zone"] = zone
    peds["road_dist_norm"] = dist
    peds["image_road_ratio"] = peds["image"].map(road_ratio)
    return peds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mask-width", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--infer-size", type=int, nargs=2, default=(1024, 576), help="SegFormer 입력 (W H)")
    args = parser.parse_args()

    paths = sorted((args.data / "images").rglob("*.jpg"))
    print(f"[masks] {len(paths)} images")
    segment_all(paths, args.data, args.mask_width, args.batch, tuple(args.infer_size))
    peds = pedestrian_road_relation(args.data)
    peds.to_csv(args.data / "objects_road.csv", index=False, encoding="utf-8-sig")
    print(peds["foot_zone"].value_counts().to_string())
    print(peds.groupby("split")["foot_zone"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
