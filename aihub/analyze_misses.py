"""학습된 탐지 모델이 놓친 보행자를 조건별로 분석한다.

정답 보행자 하나하나에 대해 conf >= --conf 예측과 IoU 0.5 이상으로 매칭되면 탐지, 아니면 미탐으로 보고
다음 조건별 미탐률을 계산한다.

- 크기: 보행자 bbox 높이 / 이미지 높이 (멀수록 작다)
- 발 위치: make_road_masks.py의 foot_zone (road / sidewalk / other)
- 장면: 시간대, 날씨, 도로 유형, 시나리오 (images.csv)

예시:
    python aihub/analyze_misses.py --data /content/data/aihub_f \
        --weights /content/drive/MyDrive/rapd/runs/yolov8s_e50_s0/weights/best.pt \
        --out /content/drive/MyDrive/rapd/results/misses --examples 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eval_common import load_gt, match, predict, split_images

SIZE_BINS = [0, 0.05, 0.10, 0.20, 1.01]
SIZE_LABELS = ["<5% (원거리)", "5-10%", "10-20%", ">=20% (근거리)"]


def miss_table(peds: pd.DataFrame, by: str) -> pd.DataFrame:
    t = peds.groupby(by, observed=True).agg(n=("missed", "size"), missed=("missed", "sum"))
    t["miss_rate"] = (t["missed"] / t["n"]).round(3)
    t["share_of_all_misses"] = (t["missed"] / max(int(peds["missed"].sum()), 1)).round(3)
    return t.reset_index().rename(columns={by: "value"}).assign(factor=by)


def save_examples(peds: pd.DataFrame, data: Path, split: str, out: Path, n: int, seed: int) -> None:
    missed = peds[peds["missed"]]
    if missed.empty or n <= 0:
        return
    out.mkdir(parents=True, exist_ok=True)
    for image in missed["image"].drop_duplicates().sample(min(n, missed["image"].nunique()), random_state=seed):
        img = cv2.imdecode(np.fromfile(str(data / "images" / split / f"{image}.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        for _, o in peds[peds["image"] == image].iterrows():
            color = (0, 0, 255) if o["missed"] else (0, 200, 0)  # 빨강 = 미탐, 초록 = 탐지
            p1 = (int(o["x1"] * W), int(o["y1"] * H))
            p2 = (int(o["x2"] * W), int(o["y2"] * H))
            cv2.rectangle(img, p1, p2, color, 2)
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            buf.tofile(str(out / f"{image}.jpg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.25, help="이 값 이상 예측만 탐지로 인정")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None, help="결과 폴더 (기본: <data>/eval/misses)")
    parser.add_argument("--examples", type=int, default=20, help="미탐 예시 이미지 저장 개수")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out or args.data / "eval" / "misses"
    out.mkdir(parents=True, exist_ok=True)

    gt = load_gt(args.data, args.split)
    preds = predict(args.weights, split_images(args.data, args.split), imgsz=args.imgsz, conf=args.conf, batch=args.batch)
    _, gt = match(preds, gt, 0.5)
    peds = gt[gt["cls"] == "pedestrian"].copy()
    peds["missed"] = ~peds["matched"]
    peds["size_bin"] = pd.cut(peds["h_norm"], SIZE_BINS, labels=SIZE_LABELS, right=False)

    road_csv = args.data / "objects_road.csv"
    if road_csv.exists():
        road = pd.read_csv(road_csv, usecols=["image", "obj_idx", "foot_zone", "road_dist_norm"])
        road["image"] = road["image"].astype(str)
        peds = peds.merge(road, on=["image", "obj_idx"], how="left")
    images_csv = args.data / "images.csv"
    if images_csv.exists():
        meta = pd.read_csv(images_csv, usecols=["image", "time", "weather", "road", "scenario"])
        meta["image"] = meta["image"].astype(str)
        peds = peds.merge(meta, on="image", how="left")

    factors = [f for f in ("size_bin", "foot_zone", "time", "weather", "road", "scenario") if f in peds.columns]
    table = pd.concat([miss_table(peds, f) for f in factors], ignore_index=True)
    table = table[["factor", "value", "n", "missed", "miss_rate", "share_of_all_misses"]]

    total = len(peds)
    print(f"[pedestrian] test {total}명, 미탐 {int(peds['missed'].sum())}명 "
          f"({peds['missed'].mean():.1%}), conf>={args.conf}")
    print(table.to_string(index=False))
    table.to_csv(out / "miss_rates.csv", index=False, encoding="utf-8-sig")
    peds.to_csv(out / "pedestrians_detected.csv", index=False, encoding="utf-8-sig")
    save_examples(peds, args.data, args.split, out / "examples", args.examples, args.seed)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
