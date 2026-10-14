"""Read-only YOLO error analysis with a mandatory baseline-reproduction gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from analysis_sanity import run_sanity, sha256
from analysis_dataset_audit import audit_dataset, load_tables
from analysis_statistics import hypothesis_analysis, logistic_failure_analysis
from analysis_visuals import make_review_outputs, make_figures
from error_analysis_core import (read_image, xywh_to_xyxy, rescale_xyxy, match_detections,
                                 signed_distance, boundary_features, bin_value, HEIGHT_EDGES,
                                 BOUNDARY_EDGES, IMAGE_DISTANCE_EDGES, rate_metrics, group_table)
from error_analysis_core import pairwise_iou, session_id

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def validate_sanity(saved, data, weights, expected, version):
    if not saved.get("passed"):
        raise RuntimeError("Baseline reproduction failed; error analysis is blocked")
    required = {"weights_sha256": sha256(weights), "yaml_sha256": sha256(Path(data) / "dataset.yaml"),
                "expected_sha256": sha256(expected), "data": str(Path(data).resolve())}
    for name, value in required.items():
        if saved["identity"].get(name) != value:
            raise RuntimeError("Sanity identity mismatch: " + name)
    if saved["ultralytics_version"] != version:
        raise RuntimeError("Inference package version differs from the passing sanity run")


def infer_cache(args, test, identity):
    from ultralytics import YOLO
    cache = args.output / "prediction_cache"
    cache.mkdir(parents=True, exist_ok=True)
    results = {}
    todo = []
    for row in test.itertuples(index=False):
        image = args.data / "images" / "test" / (row.image + ".jpg")
        source_hash = sha256(image)
        path = cache / (row.image + ".json")
        if args.resume and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["run_identity"] != identity or saved["image_sha256"] != source_hash:
                raise RuntimeError("Stale prediction cache: " + row.image)
            results[row.image] = saved
        else:
            if path.exists():
                raise FileExistsError("Prediction cache exists; use --resume")
            todo.append((row.image, image, source_hash))
    if todo:
        model = YOLO(str(args.weights))
        if [model.names[k] for k in sorted(model.names)] != identity["classes"]:
            raise ValueError("Checkpoint class names/order do not match dataset.yaml")
        for offset in range(0, len(todo), args.batch):
            chunk = todo[offset:offset + args.batch]
            decoded = [read_image(path) for _, path, _ in chunk]
            if any(im is None for im in decoded):
                raise ValueError("Undecodable test image; refusing to silently drop it")
            prediction = model.predict(source=decoded, imgsz=identity["imgsz"], conf=args.conf,
                                       iou=args.nms_iou, max_det=args.max_det, batch=args.batch,
                                       device=args.device, half=args.half, rect=True,
                                       verbose=False, save=False)
            for (stem, _, source_hash), result in zip(chunk, prediction):
                boxes = result.boxes.xyxy.cpu().numpy().astype(float)
                scores = result.boxes.conf.cpu().numpy().astype(float)
                if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                    raise RuntimeError("Nonfinite detector predictions")
                saved = {"run_identity": identity, "image_sha256": source_hash,
                         "original_shape": list(result.orig_shape), "boxes_xyxy_resized": boxes.tolist(),
                         "scores": scores.tolist(), "classes": result.boxes.cls.cpu().numpy().astype(int).tolist()}
                temporary = cache / (stem + ".tmp")
                write_json(temporary, saved)
                temporary.replace(cache / (stem + ".json"))
                results[stem] = saved
            if offset % 50 < args.batch:
                print("[predict] %d/%d (cached %d)" % (offset + len(chunk), len(todo), len(test) - len(todo)), flush=True)
    return results


def match_all(args, images, objects, road, predictions, names):
    road_index = road.set_index(["split", "image", "obj_idx"])
    grouped_objects = {image: frame.sort_values("obj_idx") for image, frame in
                       objects[objects["split"] == "test"].groupby("image")}
    rows = []
    for meta in images[images["split"] == "test"].to_dict("records"):
        image = meta["image"]
        group = grouped_objects.get(image, objects.iloc[:0])
        if group.empty:
            # Empty annotations still have FP evaluation; image dimensions come from the saved image.
            decoded = read_image(args.data / "images" / "test" / (image + ".jpg"))
            ih, iw = decoded.shape[:2]
            coordinate_status = "stored_image_no_original_metadata"
            if args.empty_original_size:
                iw, ih = args.empty_original_size
                coordinate_status = "explicit_original_size_for_empty_annotation"
            else:
                unique_sizes = objects[["img_w", "img_h"]].drop_duplicates()
                if len(unique_sizes) == 1:
                    iw, ih = unique_sizes.iloc[0].to_numpy(float)
                    coordinate_status = "inferred_unique_original_dataset_size_for_empty_annotation"
        else:
            if group[["img_w", "img_h"]].drop_duplicates().shape[0] != 1:
                raise ValueError("Inconsistent original dimensions in object metadata")
            iw, ih = float(group["img_w"].iloc[0]), float(group["img_h"].iloc[0])
            coordinate_status = "original_object_metadata"
        prediction = predictions[image]
        rh, rw = prediction["original_shape"]
        gt_boxes = xywh_to_xyxy(group[["x", "y", "w", "h"]].to_numpy())
        gt_classes = [names.index(c) for c in group["cls"]]
        pred_boxes = rescale_xyxy(prediction["boxes_xyxy_resized"], (rw, rh), (iw, ih))
        matched, fn, fp, confused = match_detections(gt_boxes, gt_classes, pred_boxes,
                                                   prediction["classes"], prediction["scores"], args.iou)
        tp_by_gt = {g: (p, overlap) for g, p, overlap in matched}
        mask = read_image(args.data / "road_masks" / "test" / (image + ".png"), cv2.IMREAD_GRAYSCALE)
        if mask is not None and not set(np.unique(mask)).issubset({0, 1, 2}):
            raise ValueError("Unexpected mask class values: " + image)
        distances = signed_distance(mask) if mask is not None else None
        meta = {key: meta[key] for key in ["image", "split", "clip", "road", "scenario", "time", "weather"]}
        meta["day_night"] = "night" if meta["time"] == "night" else ("daytime" if meta["time"] in
                             ["mrh", "day", "lunch", "afterschool", "erh"] else "unknown")
        meta["coordinate_status"] = coordinate_status
        for g, obj in enumerate(group.to_dict("records")):
            record = dict(meta)
            record.update(obj)
            p, overlap = tp_by_gt.get(g, (None, None))
            status = "TP" if p is not None else "FN"
            diagnostic = confused.get(g)
            record.update(record_type="gt", status=status, is_tp=int(status == "TP"), gt_class=obj["cls"],
                          pred_class=names[prediction["classes"][p]] if p is not None else "",
                          prediction_index=p, prediction_confidence=prediction["scores"][p] if p is not None else None,
                          matched_iou=overlap, fn_reason="" if status == "TP" else
                          ("wrong_class_overlap" if diagnostic else "no_correct_class_match"),
                          confused_prediction_class=names[prediction["classes"][diagnostic[0]]] if diagnostic else "")
            for field, coordinate in zip(["prediction_x1_original", "prediction_y1_original",
                                           "prediction_x2_original", "prediction_y2_original"],
                                          pred_boxes[p] if p is not None else [None] * 4):
                record[field] = coordinate
            record["confused_prediction_index"] = diagnostic[0] if diagnostic else None
            record["confused_prediction_iou"] = diagnostic[1] if diagnostic else None
            overlaps = pairwise_iou([gt_boxes[g]], pred_boxes)[0]
            record["max_prediction_iou_any_class"] = float(overlaps.max()) if len(overlaps) else 0.
            record["max_prediction_iou_correct_class"] = max([overlaps[k] for k, c in enumerate(prediction["classes"])
                                                               if c == gt_classes[g]], default=0.)
            key = ("test", image, obj["obj_idx"])
            estimate = road_index.loc[key] if key in road_index.index else None
            record["foot_zone"] = estimate["foot_zone"] if estimate is not None else "unknown"
            record["road_dist_norm_legacy"] = estimate["road_dist_norm"] if estimate is not None else np.nan
            if obj["cls"] == "pedestrian":
                record.update(boundary_features(mask, (obj["foot_x"], obj["foot_y"]),
                                               (obj["w"], obj["h"]), (iw, ih), distances))
            rows.append(record)
        for p in fp:
            box = pred_boxes[p]
            record = dict(meta)
            record.update(record_type="prediction", status="FP", is_tp=0, gt_class="",
                          pred_class=names[prediction["classes"][p]], obj_idx=None, prediction_index=p,
                          prediction_confidence=prediction["scores"][p], matched_iou=None,
                          img_w=iw, img_h=ih, x=box[0], y=box[1], w=box[2] - box[0], h=box[3] - box[1],
                          foot_x=(box[0] + box[2]) / 2, foot_y=box[3], foot_zone="unknown",
                          foot_point_source="predicted_bbox_bottom_center", fn_reason="")
            record.update(boundary_features(mask, (record["foot_x"], record["foot_y"]),
                                           (record["w"], record["h"]), (iw, ih), distances))
            record["foot_zone"] = record["support_zone"] if record["support_zone"] != "unknown" else record["point_zone"]
            for field, coordinate in zip(["prediction_x1_original", "prediction_y1_original",
                                           "prediction_x2_original", "prediction_y2_original"], box):
                record[field] = coordinate
            rows.append(record)
    frame = pd.DataFrame(rows)
    frame["height_bin"] = frame["h"].map(lambda value: bin_value(value, HEIGHT_EDGES))
    frame["boundary_bin"] = frame["boundary_abs_bbox_norm"].map(lambda value: bin_value(value, BOUNDARY_EDGES))
    frame["boundary_image_bin"] = frame["boundary_abs_image_norm"].map(lambda value: bin_value(value, IMAGE_DISTANCE_EDGES))
    frame["size_boundary_stratum"] = frame["height_bin"] + " | " + frame["boundary_bin"]
    return frame


def class_metrics(detections, names):
    results = {}
    for name in names:
        tp = int(((detections["gt_class"] == name) & (detections["status"] == "TP")).sum())
        fn = int(((detections["gt_class"] == name) & (detections["status"] == "FN")).sum())
        fp = int(((detections["pred_class"] == name) & (detections["status"] == "FP")).sum())
        results[name] = rate_metrics(tp, fp, fn)
    results["all_classes_micro"] = rate_metrics(sum(r["tp"] for r in results.values()),
                                                sum(r["fp"] for r in results.values()),
                                                sum(r["fn"] for r in results.values()))
    return results


def write_report(output, summary, audit, groups):
    ped = summary["fixed_threshold_metrics"]["pedestrian"]
    contrast = summary["boundary_hypothesis"]["primary_unadjusted"]
    lines = ["# 보행자 탐지 실패 분석", "", "## 재현 확인과 평가 정의",
             "- 최종 best.pt, 기존 test split 사용. 새로운 탐지 모델을 학습하지 않았다.",
             "- Ultralytics 재현 평가 AP와 고정 임계값 P/R/F1은 서로 다른 지표다.",
             "- confidence=%g, 매칭 IoU=%g. 클래스별 confidence 내림차순 일대일 매칭." %
             (summary["settings"]["conf"], summary["settings"]["iou"]),
             "- 추론 좌표를 실제 저장 이미지 크기에서 원본 메타데이터 크기로 변환했다.",
             "", "## 보행자 결과",
             "- GT %d개 중 TP %d개, FN %d개; 보행자 예측 FP %d개." % (ped["gt"], ped["tp"], ped["fn"], ped["fp"]),
             "- Precision %.4f, Recall %.4f, F1 %.4f, Miss Rate %.4f." %
             (ped["precision"], ped["recall"], ped["f1"], ped["miss_rate"]),
             "- FN 분류: " + json.dumps(summary["pedestrian_fn_reasons"], ensure_ascii=False),
             "", "## 도로 경계 가설",
             "- 기존 road_dist_norm은 발 주변 창에서 도로 영역까지의 거리다. 경계 거리로 사용하지 않았다.",
             "- 새 부호 거리는 도로 안 음수, 밖 양수이며 최근접 반대 영역 픽셀 중심까지 거리다.",
             "- bbox 높이 및 마스크/이미지 높이 정규화를 각각 제공했다.",
             "- 640px 마스크는 원본 폭 1920px에서 약 3px 간격이다. 실제 연석/미터 거리나 GT가 아니다.",
             "- 경계 근처: |부호 거리|/bbox 높이 ≤ 0.25. 구간은 상세 예측 분석 전에 코드에 고정했다.",
             "- 도로/비도로 경계는 person·차량의 분할 구멍도 포함한다. 경계 근처 결과를 연석 위험으로 해석할 수 없다.",
             "- 가까운 바닥 지지 픽셀로 옮긴 보조 분석도 제공하지만 이것 역시 추정이며 실제 경계 복원이 아니다."]
    if "near_minus_far_miss_rate" in contrast:
        lines.extend(["- 근처 FN %d/%d, 먼 곳 FN %d/%d." %
                      (contrast["near_fn"], contrast["near_gt"], contrast["far_fn"], contrast["far_gt"]),
                      "- 근처−먼 곳 미탐률 차이 %.4f, clip-bootstrap 95%% 구간 %s." %
                      (contrast["near_minus_far_miss_rate"], contrast["ci95"]),
                      "- 이 값만으로 가설을 입증하지 않는다. 크기별 층화와 정규화 민감도, 수동 마스크 검수를 함께 봐야 한다."])
        if contrast["near_minus_far_miss_rate"] <= 0:
            lines.append("- 현재 정의와 고정 분할에서는 ‘경계 부근 보행자를 더 자주 놓친다’는 가설이 지지되지 않았다. 경계 어텐션의 필요성을 입증한 결과로 쓰지 않는다.")
        else:
            lines.append("- 근처 미탐률이 높은 관찰은 탐색적 근거일 뿐이다. 구간별 크기 혼동과 마스크 검수를 통과하기 전에는 경계 어텐션 효과를 주장하지 않는다.")
        lines.append("- 세션 단위 bootstrap 민감도: " + json.dumps(summary["boundary_hypothesis"]["session_cluster_sensitivity"], ensure_ascii=False))
    else:
        lines.append("- 근처/먼 곳 표본 부족으로 차이를 계산할 수 없었다.")
    lines.extend(["", "## 조건별 결과", "", "|조건|그룹|GT|TP|FN|Recall|95% CI|표본 상태|",
             "|---|---|---:|---:|---:|---:|---|---|"])
    for row in groups.itertuples(index=False):
        if row.dimension in ("height_bin", "foot_zone", "road", "day_night", "weather"):
            ci = "n/a" if not np.isfinite([row.recall_ci_low, row.recall_ci_high]).all() else "[%.3f, %.3f]" % (row.recall_ci_low, row.recall_ci_high)
            recall_text = "%.3f" % row.recall if np.isfinite(row.recall) else "n/a"
            lines.append("|%s|%s|%d|%d|%d|%s|%s|%s|" %
                         (row.dimension, row.group, row.gt, row.tp, row.fn, recall_text, ci, row.sample_status))
    lines.extend(["", "## 데이터 점검과 한계",
                  "- clip split 중복 %d개. 괄호 전 주행 세션이 여러 split에 걸친 경우 %d개, 관련 이미지 %d행." %
                  (audit["clip_split_overlap_count"], audit["session_split_overlap_count"], audit["session_overlap_image_rows"]),
                  "- 원본 CSV와 YOLO 라벨 역변환의 0.01px 초과 좌표 차이 %d개. 자동 수정하지 않았다." % audit["label_coordinate_error_gt_0_01_px"],
                  "- 객체 수/형식 불일치로 좌표 비교 불가능한 이미지 %d개는 별도 보고했다. 위 좌표 차이 0개는 이 이미지를 포함한 전체 정합성 증명이 아니다." % len(audit["label_structure_mismatches"]),
                  "- 메타데이터 이미지 %d행, 디스크 이미지 %d개. 누락 파일 %d개." %
                  (audit["metadata_images"], sum(audit["disk_images"].values()), len(audit["missing_images"])),
                  "- 메타데이터에서 확인된 중복 파일명 %d행. 이미 덮어써진 원본은 이 백업만으로 검증 불가." % audit["duplicate_filename_rows"],
                  "- 가림 정보는 제공되지 않아 가림별 성능 분석 불가. 작은 box를 가림이라고 간주하지 않았다.",
                  "- 마스크 검수 시트는 정답 칸을 비워 뒀다. 검수 전 SegFormer 분할 정확도를 주장하지 않는다.",
                  "- 모든 조건 비교는 탐색적이며 다중 비교·작은 그룹·같은 주행 세션의 상관이 남는다.",
                  "- 조건별 CSV에 clip CI와 세션 CI를 나란히 제공한다. 같은 세션의 여러 split 배치는 독립 주행 일반화 검증을 약화한다.",
                  "- 로지스틱 회귀는 test FN의 정규화된 설명적 분석이며 인과 추론/새 탐지기 학습이 아니다.",
                  "- 로지스틱 회귀의 공선성/설계 행렬 rank는 JSON에 보고한다. 밤·안개처럼 완전히 겹친 변수의 개별 효과를 분리 해석하지 않는다.",
                  "", "## RGA 개발 판단",
                  "- 경계 prior의 채택은 평균 결과만으로 정하지 않는다. bbox 크기별 차이와 마스크 검수 후 결정한다.",
                  "- 고려할 입력: 부호/절대 거리 두 정규화, 작은 객체 특징, road/sidewalk/unknown 및 prior 신뢰도.",
                  "- 마스크가 틀릴 때도 탐지 특징을 지우지 않는 residual attention 경로를 우선한다.",
                  "- 다음 비교: 기존 YOLO, 단순 road 입력, 동일 용량 일반 attention, RGA, RGA+경계. 동일 split/학습/seed로 비교한다.",
                  "- 이번 test 진단에서 본 조건은 새 연구 가설 생성용이다. 제안 모델의 최종 독립 검증셋을 따로 확보해야 한다."])
    (Path(output) / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected", type=Path, default=REPO / "outputs/aihub/results/yolov8s_e50_s0.json")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/aihub/error_analysis")
    parser.add_argument("--sanity-dir", type=Path)
    parser.add_argument("--stage", choices=["audit", "sanity", "analyze", "all"], default="all")
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sanity-batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--nms-iou", type=float, default=.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--logistic-bootstrap", type=int, default=200)
    parser.add_argument("--review-per-zone", type=int, default=50)
    parser.add_argument("--examples-per-type", type=int, default=12)
    parser.add_argument("--empty-original-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Original dimensions for images with empty annotations; otherwise unique dataset metadata size is inferred and flagged")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (0 < args.iou <= 1 and 0 <= args.conf <= 1 and 0 < args.nms_iou <= 1):
        parser.error("Invalid matching, confidence or NMS threshold")
    if min(args.batch, args.sanity_batch, args.bootstrap, args.logistic_bootstrap, args.max_det) < 1:
        parser.error("Batch/repetitions/max-det must be positive")
    if min(args.review_per_zone, args.examples_per_type) < 0 or (args.empty_original_size and min(args.empty_original_size) <= 0):
        parser.error("Invalid review count or image dimensions")
    args.data, args.weights, args.output = args.data.resolve(), args.weights.resolve(), args.output.resolve()
    if args.output == args.data or args.data in args.output.parents:
        parser.error("Output must not modify the dataset directory")
    args.output.mkdir(parents=True, exist_ok=True)
    settings_dir = args.output / "ultralytics_settings"
    settings_dir.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    if args.package_dir:
        sys.path.insert(0, str(args.package_dir.resolve()))
    config = yaml.safe_load((args.data / "dataset.yaml").read_text(encoding="utf-8"))
    mapping = config["names"]
    names = [mapping[i] for i in sorted(mapping)] if isinstance(mapping, dict) else list(mapping)
    if "pedestrian" not in names:
        parser.error("Pedestrian class not found")
    if args.stage == "audit":
        print(json.dumps(audit_dataset(args.data, args.output, names), indent=2))
        return
    import torch
    import ultralytics
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    sanity_dir = args.sanity_dir or args.output / "sanity"
    if args.stage in ("sanity", "all"):
        saved = run_sanity(args.data, args.weights, args.expected, sanity_dir,
                           args.sanity_batch, args.device, args.half, resume=args.resume)
        if args.stage == "sanity":
            return
    else:
        saved = json.loads((sanity_dir / "sanity_check.json").read_text(encoding="utf-8"))
    validate_sanity(saved, args.data, args.weights, args.expected, ultralytics.__version__)
    audit = audit_dataset(args.data, args.output, names)
    if audit["test_analysis_blocking"]:
        raise RuntimeError("Blocking test data audit issue; see dataset_audit.json")
    identity = {"weights_sha256": sha256(args.weights), "metadata_sha256":
                {n: sha256(args.data / n) for n in ["images.csv", "objects.csv", "objects_road.csv", "dataset.yaml"]},
                "ultralytics_version": ultralytics.__version__, "imgsz": int(saved["identity"]["imgsz"]),
                "conf": args.conf, "iou": args.iou, "nms_iou": args.nms_iou,
                "half": args.half, "max_det": args.max_det, "classes": names, "batch": args.batch,
                "device": args.device}
    manifest_path = args.output / "analysis_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume or prior["identity"] != identity:
            raise RuntimeError("Existing analysis or changed settings; use --resume with identical settings or new output")
    source_files = ["analyze_detection_errors.py", "analysis_sanity.py", "analysis_dataset_audit.py",
                    "analysis_statistics.py", "analysis_visuals.py", "error_analysis_core.py"]
    source_hashes = {n: sha256(REPO / "aihub" / n) for n in source_files}
    manifest = {"identity": identity, "source_sha256": source_hashes, "created_utc": datetime.now(timezone.utc).isoformat(),
                "sanity_result": str((sanity_dir / "sanity_check.json").resolve()),
                "sanity_sha256": sha256(sanity_dir / "sanity_check.json"), "new_detector_trained": False,
                "group_cutoffs_prespecified_in_code": True}
    write_json(manifest_path, manifest)
    images, objects, road = load_tables(args.data)
    test = images[images["split"] == "test"].sort_values("image")
    predictions = infer_cache(args, test, identity)
    print("[match] one-to-one GT/prediction matching", flush=True)
    detections = match_all(args, images, objects, road, predictions, names)
    detections.to_csv(args.output / "detections_matched.csv", index=False, encoding="utf-8-sig")
    pedestrians = detections[(detections["record_type"] == "gt") & (detections["gt_class"] == "pedestrian")].copy()
    groups = group_table(pedestrians, ["height_bin", "foot_zone", "road", "day_night", "time", "weather",
                                     "scenario", "boundary_bin", "boundary_image_bin", "size_boundary_stratum"], args.bootstrap)
    groups.to_csv(args.output / "pedestrian_failure_analysis.csv", index=False, encoding="utf-8-sig")
    groups[groups["dimension"].isin(["boundary_bin", "boundary_image_bin", "size_boundary_stratum"])].to_csv(
        args.output / "road_boundary_analysis.csv", index=False, encoding="utf-8-sig")
    hypothesis, strata = hypothesis_analysis(pedestrians, args.bootstrap)
    strata.to_csv(args.output / "boundary_size_stratified_contrasts.csv", index=False, encoding="utf-8-sig")
    logistic = logistic_failure_analysis(pedestrians, args.logistic_bootstrap)
    session_peds = pedestrians.copy()
    session_peds["clip"] = session_peds["clip"].map(session_id)
    logistic["session_cluster_sensitivity"] = logistic_failure_analysis(session_peds, args.logistic_bootstrap)
    write_json(args.output / "logistic_failure_analysis.json", logistic)
    print("[visuals] seeded review overlays and figures", flush=True)
    visual = make_review_outputs(args.data, detections, args.output, args.review_per_zone, args.examples_per_type)
    figures = make_figures(groups, args.output, args.conf, args.iou)
    summary = {"status": "completed", "test_images": len(test), "settings": identity,
               "fixed_threshold_metrics": class_metrics(detections, names),
               "ultralytics_sanity_metrics_separate": saved["observed"],
               "pedestrian_fn_reasons": pedestrians[pedestrians["status"] == "FN"]["fn_reason"].value_counts().to_dict(),
               "wrong_class_overlap_classes": pedestrians[pedestrians["status"] == "FN"]["confused_prediction_class"].value_counts().to_dict(),
               "boundary_hypothesis": hypothesis, "mask_review": visual, "figure_count": figures,
               "estimated_mask_is_ground_truth": False, "occlusion_evaluated": False,
               "new_detector_trained": False, "coordinate_transform": "Resized prediction -> original img_w/img_h; raw CSV GT unchanged"}
    write_json(args.output / "summary.json", summary)
    write_report(args.output, summary, audit, groups)
    if identity["metadata_sha256"] != {n: sha256(args.data / n) for n in identity["metadata_sha256"]}:
        raise RuntimeError("Source metadata unexpectedly changed")
    print(json.dumps({"output": str(args.output), "pedestrian": summary["fixed_threshold_metrics"]["pedestrian"],
                      "hypothesis": hypothesis["primary_unadjusted"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
