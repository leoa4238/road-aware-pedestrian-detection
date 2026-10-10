"""AI-Hub 058 생활도로 객체인식 데이터(전방 카메라)를 YOLO detection 형식으로 변환한다.

- 라벨/이미지 zip을 전부 풀지 않고 필요한 파일만 읽는다.
- train/val/test는 clip(촬영 구간) 단위로 나눠 인접 프레임이 split을 넘나들지 않게 한다.
- 보행자는 instance mask의 가장 아래 픽셀을 발 위치로 기록해 이후 도로 위/밖 판정에 쓴다.

예시:
    python aihub/prepare_dataset.py --label-root "Y:/.../1.데이터/Validation/02.라벨링데이터" \
        --image-root "Y:/.../1.데이터/Validation/01.원천데이터" --out C:/work/data/aihub_f
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

# AI-Hub category_id -> (YOLO class id, name). Adult와 Kid student는 보행자 하나로 합친다.
CLASS_MAP = {98: 0, 97: 0, 3: 1, 8: 2, 2: 3, 99: 4}
CLASS_NAMES = ["pedestrian", "car", "truck_bus", "two_wheeler", "personal_mobility"]
PEDESTRIAN_CLASS = 0


def find_zips(root: Path, camera: str) -> list[Path]:
    return sorted(p for p in root.rglob(f"*image_{camera}*.zip"))


def clip_id_of(file_name: str) -> str:
    # 230914_3/230914_115413_K/230914_115413_K(9)/image_F/xxx.png -> 230914_115413_K(9)
    parts = PurePosixPath(file_name).parts
    return parts[-3] if len(parts) >= 3 else "unknown"


def foot_point(segmentation: dict | None, bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    if segmentation and isinstance(segmentation.get("counts"), str):
        rle = {"size": segmentation["size"], "counts": segmentation["counts"].encode("ascii")}
        mask = mask_utils.decode(rle)
        ys, xs = np.nonzero(mask)
        if len(ys):
            bottom = ys.max()
            row_xs = xs[ys >= bottom - 2]
            return float(row_xs.mean()), float(bottom)
    return x + w / 2, y + h


def read_labels(label_zips: list[Path]) -> list[dict]:
    records = []
    for zpath in label_zips:
        with zipfile.ZipFile(zpath) as zf:
            for name in tqdm(zf.namelist(), desc=f"labels {zpath.name}"):
                if not name.endswith(".json"):
                    continue
                data = json.loads(zf.read(name).decode("utf-8"))
                image = data["images"][0]
                objects, seen = [], set()
                for ann in data.get("annotations", []):
                    cls = CLASS_MAP.get(ann["category_id"])
                    if cls is None:
                        continue
                    bbox = [float(v) for v in ann["bbox"]]
                    key = (cls, *(round(v) for v in bbox))
                    if key in seen or bbox[2] < 2 or bbox[3] < 2:
                        continue
                    seen.add(key)
                    fx, fy = foot_point(ann.get("segmentation"), bbox) if cls == PEDESTRIAN_CLASS else (np.nan, np.nan)
                    objects.append({"cls": cls, "bbox": bbox, "foot_x": fx, "foot_y": fy})
                records.append({
                    "png": data.get("png_filename") or PurePosixPath(image["file_name"]).name,
                    "width": image["width"],
                    "height": image["height"],
                    "clip": clip_id_of(image["file_name"]),
                    "scenario": data.get("scenario", ""),
                    "road": data.get("road", ""),
                    "time": data.get("time", ""),
                    "weather": data.get("weather", ""),
                    "company": data.get("company", ""),
                    "objects": objects,
                })
    return records


def assign_splits(records: list[dict], ratios: tuple[float, float, float], seed: int) -> None:
    clips = sorted({r["clip"] for r in records})
    random.Random(seed).shuffle(clips)
    n_train = int(len(clips) * ratios[0])
    n_val = int(len(clips) * ratios[1])
    split_of = {c: "train" for c in clips[:n_train]}
    split_of.update({c: "val" for c in clips[n_train:n_train + n_val]})
    split_of.update({c: "test" for c in clips[n_train + n_val:]})
    for r in records:
        r["split"] = split_of[r["clip"]]


def write_images(records: list[dict], image_zips: list[Path], out: Path, width: int, quality: int) -> set[str]:
    wanted = {r["png"]: r for r in records}
    written = set()
    for zpath in image_zips:
        with zipfile.ZipFile(zpath) as zf:
            members = [m for m in zf.namelist() if PurePosixPath(m).name in wanted]
            for member in tqdm(members, desc=f"images {zpath.name}"):
                rec = wanted[PurePosixPath(member).name]
                stem = Path(rec["png"]).stem
                target = out / "images" / rec["split"] / f"{stem}.jpg"
                if not target.exists():
                    buf = np.frombuffer(zf.read(member), dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    h = round(img.shape[0] * width / img.shape[1])
                    img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                written.add(rec["png"])
    return written


def write_labels_and_meta(records: list[dict], written: set[str], out: Path) -> None:
    img_rows, obj_rows = [], []
    for rec in records:
        if rec["png"] not in written:
            continue
        stem = Path(rec["png"]).stem
        W, H = rec["width"], rec["height"]
        lines = []
        for i, o in enumerate(rec["objects"]):
            x, y, w, h = o["bbox"]
            cx, cy = min(max((x + w / 2) / W, 0), 1), min(max((y + h / 2) / H, 0), 1)
            lines.append(f"{o['cls']} {cx:.6f} {cy:.6f} {min(w / W, 1):.6f} {min(h / H, 1):.6f}")
            obj_rows.append({
                "image": stem, "obj_idx": i, "split": rec["split"], "cls": CLASS_NAMES[o["cls"]],
                "x": x, "y": y, "w": w, "h": h, "foot_x": o["foot_x"], "foot_y": o["foot_y"],
                "img_w": W, "img_h": H,
            })
        label_path = out / "labels" / rec["split"] / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")
        img_rows.append({k: rec[k] for k in ("split", "clip", "scenario", "road", "time", "weather", "company")}
                        | {"image": stem, "n_objects": len(lines),
                           "n_pedestrian": sum(o["cls"] == PEDESTRIAN_CLASS for o in rec["objects"])})

    for name, rows in (("images.csv", img_rows), ("objects.csv", obj_rows)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    yaml_lines = [f"path: {out.as_posix()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    (out / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    counts = defaultdict(int)
    for r in img_rows:
        counts[r["split"]] += 1
    print(f"[done] images per split: {dict(counts)}, objects: {len(obj_rows)} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera", default="F")
    parser.add_argument("--width", type=int, default=1280, help="저장 이미지 가로 크기(px)")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label_zips = find_zips(args.label_root, args.camera)
    image_zips = find_zips(args.image_root, args.camera)
    print(f"[zips] labels={[p.name for p in label_zips]} images={[p.name for p in image_zips]}")
    if not label_zips or not image_zips:
        raise SystemExit("라벨 또는 이미지 zip을 찾지 못했습니다.")

    records = read_labels(label_zips)
    assign_splits(records, tuple(args.ratios), args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    written = write_images(records, image_zips, args.out, args.width, args.quality)
    write_labels_and_meta(records, written, args.out)


if __name__ == "__main__":
    main()
