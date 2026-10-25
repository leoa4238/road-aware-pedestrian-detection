"""평가 스크립트 공용 함수: 정답 읽기, YOLO 예측, IoU 매칭, AP50 계산.

좌표는 모두 이미지 크기로 정규화한 [0, 1] 값으로 다룬다. IoU는 x/y 축별 배율 변화에 불변이므로
원본(objects.csv)과 축소 저장한 jpg 사이의 크기 차이를 따로 맞출 필요가 없다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = ["pedestrian", "car", "truck_bus", "two_wheeler", "personal_mobility"]

# COCO 사전학습 모델의 클래스를 이 연구의 클래스로 대응시킨다. PM은 COCO에 대응 클래스가 없다.
COCO_TO_OURS = {
    "person": "pedestrian",
    "car": "car",
    "bus": "truck_bus",
    "truck": "truck_bus",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
}


def load_gt(data: Path, split: str) -> pd.DataFrame:
    """objects.csv의 split 정답을 정규화 xyxy로 반환한다."""
    obj = pd.read_csv(data / "objects.csv")
    obj = obj[obj["split"] == split].copy()
    obj["x1"] = obj["x"] / obj["img_w"]
    obj["y1"] = obj["y"] / obj["img_h"]
    obj["x2"] = (obj["x"] + obj["w"]) / obj["img_w"]
    obj["y2"] = (obj["y"] + obj["h"]) / obj["img_h"]
    obj["h_norm"] = obj["h"] / obj["img_h"]
    obj["image"] = obj["image"].astype(str)
    return obj.reset_index(drop=True)


def split_images(data: Path, split: str) -> list[Path]:
    return sorted((data / "images" / split).glob("*.jpg"))


def predict(weights: str | Path, images: list[Path], imgsz: int = 640, conf: float = 0.001,
            batch: int = 16, device: str | None = None) -> pd.DataFrame:
    """YOLO 예측 결과를 이 연구의 클래스 이름으로 바꿔 반환한다. 대응 클래스가 없는 예측은 버린다."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    is_coco = "person" in names.values()
    rows = []
    for i in range(0, len(images), batch):
        chunk = [str(p) for p in images[i:i + batch]]
        for path, res in zip(chunk, model.predict(chunk, imgsz=imgsz, conf=conf, device=device, verbose=False)):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxyn = res.boxes.xyxyn.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            stem = Path(path).stem
            for (x1, y1, x2, y2), c, k in zip(xyxyn, confs, clss):
                name = names[int(k)]
                ours = COCO_TO_OURS.get(name) if is_coco else name
                if ours is None:
                    continue
                rows.append({"image": stem, "cls": ours, "conf": float(c),
                             "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
    return pd.DataFrame(rows, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N, 4), b: (M, 4) xyxy -> (N, M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def match(preds: pd.DataFrame, gt: pd.DataFrame, iou_thr: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """클래스·이미지별로 신뢰도 높은 예측부터 아직 매칭되지 않은 정답에 greedy 매칭한다.

    반환: (preds + tp 열, gt + matched/matched_conf 열)
    """
    preds = preds.sort_values("conf", ascending=False).reset_index(drop=True)
    preds["tp"] = False
    gt = gt.copy()
    gt["matched"] = False
    gt["matched_conf"] = np.nan
    gt_groups = {key: idx for key, idx in gt.groupby(["image", "cls"]).groups.items()}
    for (image, cls), p_idx in preds.groupby(["image", "cls"]).groups.items():
        g_idx = gt_groups.get((image, cls))
        if g_idx is None:
            continue
        g_idx = np.asarray(g_idx)
        p_idx = np.sort(np.asarray(p_idx))  # preds가 conf 내림차순이므로 인덱스 순서 = conf 순서
        ious = iou_matrix(preds.loc[p_idx, ["x1", "y1", "x2", "y2"]].to_numpy(),
                          gt.loc[g_idx, ["x1", "y1", "x2", "y2"]].to_numpy())
        taken = np.zeros(len(g_idx), dtype=bool)
        for row, pi in enumerate(p_idx):
            cand = np.where(~taken & (ious[row] >= iou_thr))[0]
            if len(cand) == 0:
                continue
            best = cand[np.argmax(ious[row, cand])]
            taken[best] = True
            preds.at[pi, "tp"] = True
            gt.at[g_idx[best], "matched"] = True
            gt.at[g_idx[best], "matched_conf"] = preds.at[pi, "conf"]
    return preds, gt


def average_precision(tp: np.ndarray, n_gt: int) -> float:
    """신뢰도 내림차순으로 정렬된 tp 배열의 AP (COCO 101점 보간)."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    # precision envelope: 오른쪽에서 왼쪽으로 누적 최대
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def ap50_per_class(preds: pd.DataFrame, gt: pd.DataFrame, classes: list[str]) -> dict[str, float]:
    matched_preds, _ = match(preds, gt, 0.5)
    out = {}
    for cls in classes:
        p = matched_preds[matched_preds["cls"] == cls]  # 이미 conf 내림차순
        out[cls] = average_precision(p["tp"].to_numpy(), int((gt["cls"] == cls).sum()))
    return out
