"""위험 보행자 정의 후보를 같은 데이터에 적용해 비율과 예시 이미지로 비교한다.

생활도로에서는 보행자의 68%가 차도 위에 있어 "도로 위 = 위험"으로는 구분력이 없다.
그래서 자차 주행 경로(이미지 하단 중앙의 사다리꼴 영역)와 거리(보행자 크기)를 쓰는 정의를 함께 비교한다.

정의 (기본값은 인자로 조정):
    on_road        발 위치가 도로 마스크 위 (기존 정의, 비교 기준)
    ego_path       발 위치가 자차 주행 경로 사다리꼴 안
    ego_path_near  ego_path 이면서 bbox 높이 / 이미지 높이 >= --near-h (가까움)
    path_or_edge   ego_path_near 이거나, 주행 경로 바로 옆(--edge-margin)에서 도로 위에 있는 근거리 보행자

정의가 타당한지 보는 근거로, 시나리오 코드별 위험 비율을 함께 낸다.
보행자가 도로를 횡단하거나 차도에서 통행하는 시나리오(CROSSING_OR_IN_ROAD)에서 비율이 높고,
보도 통행 시나리오(SIDEWALK)에서 낮을수록 구분력이 있는 정의다.
시나리오는 이미지 단위 정보이므로 보행자별 정답이 아니라 경향 확인용이다.

예시:
    python aihub/compare_risk_definitions.py --data /content/data/aihub_f --out /content/drive/MyDrive/rapd/results/risk
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# AI-Hub 058 시나리오 코드 (데이터 소개 페이지 기준)
CROSSING_OR_IN_ROAD = {"A04", "A13", "B01", "B02", "B08", "B09", "B12"}
SIDEWALK = {"B10"}
EDGE = {"B11"}
DEFINITIONS = ["on_road", "ego_path", "ego_path_near", "path_or_edge"]


def scenario_code(value: object) -> str:
    m = re.search(r"[AB]\d{2}", str(value))
    return m.group(0) if m else "unknown"


def scenario_group(code: str) -> str:
    if code in CROSSING_OR_IN_ROAD:
        return "crossing_or_in_road"
    if code in SIDEWALK:
        return "sidewalk"
    if code in EDGE:
        return "edge"
    return "other"


def path_half_width(v: np.ndarray, horizon: float, hw_bottom: float, hw_horizon: float) -> np.ndarray:
    """정규화 세로 위치 v에서 주행 경로 사다리꼴의 반폭 (가로 정규화 단위)."""
    t = np.clip((v - horizon) / (1 - horizon), 0, 1)
    return hw_horizon + (hw_bottom - hw_horizon) * t


def apply_definitions(peds: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    u = peds["foot_x"] / peds["img_w"]
    v = peds["foot_y"] / peds["img_h"]
    hw = path_half_width(v.to_numpy(), args.horizon, args.hw_bottom, args.hw_horizon)
    lateral = (u - 0.5).abs().to_numpy()
    below_horizon = (v > args.horizon).to_numpy()
    near = (peds["h"] / peds["img_h"] >= args.near_h).to_numpy()
    on_road = (peds["foot_zone"] == "road").to_numpy()

    peds = peds.copy()
    peds["on_road"] = on_road
    peds["ego_path"] = below_horizon & (lateral <= hw)
    peds["ego_path_near"] = peds["ego_path"].to_numpy() & near
    beside_path = below_horizon & (lateral > hw) & (lateral <= hw + args.edge_margin)
    peds["path_or_edge"] = peds["ego_path_near"].to_numpy() | (beside_path & on_road & near)
    return peds


def rate_table(peds: pd.DataFrame, by: str) -> pd.DataFrame:
    t = peds.groupby(by)[DEFINITIONS].mean().round(3)
    t.insert(0, "n", peds.groupby(by).size())
    return t.reset_index()


def draw_examples(peds: pd.DataFrame, data: Path, out: Path, n: int, seed: int, args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    images = peds.drop_duplicates("image").sample(min(n, peds["image"].nunique()), random_state=seed)
    for _, row in images.iterrows():
        path = data / "images" / row["split"] / f"{row['image']}.jpg"
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR) if path.exists() else None
        if img is None:
            continue
        H, W = img.shape[:2]
        # 주행 경로 사다리꼴 (노랑)과 경로 옆 여유 폭 (회색)
        for margin, color in ((args.edge_margin, (160, 160, 160)), (0.0, (0, 220, 255))):
            top = args.hw_horizon + margin
            bottom = args.hw_bottom + margin
            pts = np.array([[(0.5 - top) * W, args.horizon * H], [(0.5 + top) * W, args.horizon * H],
                            [(0.5 + bottom) * W, H - 1], [(0.5 - bottom) * W, H - 1]], dtype=np.int32)
            cv2.polylines(img, [pts], True, color, 2)
        for _, o in peds[peds["image"] == row["image"]].iterrows():
            sx, sy = W / o["img_w"], H / o["img_h"]
            p1 = (int(o["x"] * sx), int(o["y"] * sy))
            p2 = (int((o["x"] + o["w"]) * sx), int((o["y"] + o["h"]) * sy))
            color = (0, 0, 255) if o["path_or_edge"] else (0, 160, 255) if o["ego_path"] else (0, 200, 0)
            cv2.rectangle(img, p1, p2, color, 2)
            cv2.circle(img, (int(o["foot_x"] * sx), int(o["foot_y"] * sy)), 4, color, -1)
            label = "".join(k[0] for k, on in (("R", o["on_road"]), ("P", o["ego_path"]),
                                               ("N", o["ego_path_near"]), ("E", o["path_or_edge"])) if on)
            cv2.putText(img, label, (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, f"{row.get('scenario_code', '')} {row.get('road', '')} {row.get('time', '')}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            buf.tofile(str(out / f"{row['image']}.jpg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="결과 폴더 (기본: <data>/eval/risk)")
    parser.add_argument("--horizon", type=float, default=0.45, help="소실선 높이 (이미지 높이 비율)")
    parser.add_argument("--hw-bottom", type=float, default=0.30, help="이미지 하단에서 주행 경로 반폭 (가로 비율)")
    parser.add_argument("--hw-horizon", type=float, default=0.03, help="소실선에서 주행 경로 반폭 (가로 비율)")
    parser.add_argument("--near-h", type=float, default=0.10, help="가까움 기준: bbox 높이 / 이미지 높이")
    parser.add_argument("--edge-margin", type=float, default=0.08, help="경로 옆 여유 폭 (가로 비율)")
    parser.add_argument("--examples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out or args.data / "eval" / "risk"
    out.mkdir(parents=True, exist_ok=True)

    peds = pd.read_csv(args.data / "objects_road.csv")
    peds["image"] = peds["image"].astype(str)
    peds = peds.dropna(subset=["foot_x", "foot_y"])
    meta = pd.read_csv(args.data / "images.csv", usecols=["image", "scenario", "road", "time", "weather"])
    meta["image"] = meta["image"].astype(str)
    peds = peds.merge(meta, on="image", how="left")
    peds["scenario_code"] = peds["scenario"].map(scenario_code)
    peds["scenario_group"] = peds["scenario_code"].map(scenario_group)

    peds = apply_definitions(peds, args)
    overall = peds[DEFINITIONS].mean().round(3).rename("positive_rate").to_frame()
    by_group = rate_table(peds, "scenario_group")
    by_code = rate_table(peds, "scenario_code")
    by_road = rate_table(peds, "road")

    # 구분력: 횡단/차도 통행 시나리오 비율 - 보도 통행 시나리오 비율 (클수록 좋음)
    g = by_group.set_index("scenario_group")
    if {"crossing_or_in_road", "sidewalk"} <= set(g.index):
        gap = (g.loc["crossing_or_in_road", DEFINITIONS] - g.loc["sidewalk", DEFINITIONS]).round(3)
        overall["crossing_minus_sidewalk"] = gap

    print(f"[pedestrians] {len(peds)}명, 설정: {vars(args)}")
    print("\n== 전체 위험 비율 ==\n" + overall.to_string())
    print("\n== 시나리오 그룹별 ==\n" + by_group.to_string(index=False))
    print("\n== 도로 유형별 ==\n" + by_road.to_string(index=False))

    overall.to_csv(out / "overall.csv", encoding="utf-8-sig")
    by_group.to_csv(out / "by_scenario_group.csv", index=False, encoding="utf-8-sig")
    by_code.to_csv(out / "by_scenario_code.csv", index=False, encoding="utf-8-sig")
    by_road.to_csv(out / "by_road_type.csv", index=False, encoding="utf-8-sig")
    peds.to_csv(out / "pedestrians_risk.csv", index=False, encoding="utf-8-sig")
    draw_examples(peds, args.data, out / "examples", args.examples, args.seed, args)
    print(f"\n[saved] {out}  (예시 이미지: 노랑=주행 경로, 회색=경로 옆 여유 폭, 빨강=path_or_edge, 주황=ego_path)")


if __name__ == "__main__":
    main()
