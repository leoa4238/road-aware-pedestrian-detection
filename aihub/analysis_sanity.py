"""Evaluate the supplied final checkpoint before any detection-error analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def run_sanity(data, weights, expected, output, batch=2, device="0", half=False,
               tolerance=0.00005, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(data).resolve() / "dataset.yaml"
    reference = json.loads(Path(expected).read_text(encoding="utf-8"))
    identity = {"weights_sha256": sha256(weights), "yaml_sha256": sha256(source),
                "expected_sha256": sha256(expected), "data": str(Path(data).resolve()),
                "imgsz": int(reference["imgsz"]), "batch": batch, "device": device,
                "half": half, "absolute_tolerance": tolerance}
    target = output / "sanity_check.json"
    if resume and target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Sanity cache settings changed; use a new output directory")
        if not saved["passed"]:
            raise RuntimeError("Cached baseline reproduction failed; analysis is blocked")
        return saved
    if target.exists():
        raise FileExistsError("Sanity results exist; use --resume or a new output")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["path"] = Path(data).resolve().as_posix()
    evaluation_yaml = output / "evaluation_dataset.yaml"
    evaluation_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    settings = output / "ultralytics_settings"
    settings.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings)
    import ultralytics
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    model = YOLO(str(weights))
    metrics = model.val(data=str(evaluation_yaml), split="test", imgsz=reference["imgsz"],
                        batch=batch, device=device, half=half, workers=0, plots=False,
                        project=str(output), name="sanity_val", exist_ok=True, verbose=False)
    observed = {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
                "per_class_mAP50": {metrics.names[int(c)]: float(ap)
                                    for c, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)}}
    differences = {"mAP50": observed["mAP50"] - reference["mAP50"],
                   "pedestrian_mAP50": observed["per_class_mAP50"]["pedestrian"]
                   - reference["per_class_mAP50"]["pedestrian"]}
    result = {"identity": identity, "expected": reference, "observed": observed,
              "differences": differences, "passed": all(abs(d) <= tolerance for d in differences.values()),
              "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "note": "Ultralytics AP is distinct from fixed-threshold one-to-one P/R/F1."}
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if sha256(source) != identity["yaml_sha256"]:
        raise RuntimeError("Original dataset.yaml was unexpectedly changed")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("Baseline does not reproduce within predeclared tolerance; detailed analysis blocked")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.00005)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--package-dir", type=Path, help="Optional isolated original Ultralytics installation")
    args = parser.parse_args()
    if args.batch < 1 or args.tolerance < 0:
        parser.error("batch must be positive and tolerance nonnegative")
    if args.package_dir:
        sys.path.insert(0, str(args.package_dir.resolve()))
    run_sanity(args.data, args.weights, args.expected, args.output,
               args.batch, args.device, args.half, args.tolerance, args.resume)


if __name__ == "__main__":
    main()
