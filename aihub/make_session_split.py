"""Build a new session-level train/val/test split as a separate data version.

A session is a recording start (clip name without its "(n)" suffix); clips of the
same session are consecutive segments of one drive, so they must stay in one split.
Nothing in the existing dataset is moved or overwritten: the script writes image
lists and a dataset yaml next to the existing `images/` folder, using relative
"./images/..." entries that Ultralytics resolves against the list's directory.

Example:
    python aihub/make_session_split.py --data data/analysis_backup_20261013/dataset/aihub_f \
        --version session_v1 --report-dir outputs/aihub/splits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from error_analysis_core import session_id

SPLITS = ("train", "val", "test")


def assign(sessions, ratios, seed):
    """Within each (time, weather) stratum, give each session to the split furthest below target."""
    rng = np.random.default_rng(seed)
    result = {}
    for _, stratum in sessions.groupby(["time", "weather"], sort=True):
        order = stratum.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        # Large sessions first so small ones can fill the remaining gaps.
        order = order.sort_values("images", ascending=False, kind="stable")
        total = order["images"].sum()
        filled = dict.fromkeys(SPLITS, 0)
        for name, row in order.iterrows():
            deficit = {s: ratios[s] * total - filled[s] for s in SPLITS}
            split = max(SPLITS, key=lambda s: (deficit[s], -SPLITS.index(s)))
            result[name] = split
            filled[split] += row["images"]
    return pd.Series(result)


def score(sessions, split, ratios):
    """Mean absolute deviation from target shares (images, pedestrians, rainy/foggy/night images)."""
    frame = sessions.assign(split=split)
    columns = ["images", "pedestrians", "rainy", "foggy", "night"]
    error = 0.
    for column in columns:
        total = frame[column].sum()
        if total:
            shares = frame.groupby("split")[column].sum() / total
            error += sum(abs(shares.get(s, 0) - ratios[s]) for s in SPLITS)
    return error / len(columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="dataset root containing images.csv")
    parser.add_argument("--version", default="session_v1")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--ratios", type=float, nargs=3, default=(.7, .15, .15))
    parser.add_argument("--search-seeds", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ratios = dict(zip(SPLITS, args.ratios))
    if abs(sum(ratios.values()) - 1) > 1e-6:
        parser.error("ratios must sum to 1")

    images = pd.read_csv(args.data / "images.csv")
    images["session"] = images["clip"].map(session_id)
    # Same filename in different clips: only one file survived on disk, so the pair is unusable.
    duplicated = images["image"].duplicated(keep=False)
    excluded = images[duplicated]
    images = images[~duplicated].copy()
    images["source_split"] = images["split"]
    images["path"] = ["./images/%s/%s.jpg" % (s, n) for s, n in zip(images["split"], images["image"])]
    missing = [p for p in images["path"] if not (args.data / p).exists()]
    if missing:
        raise FileNotFoundError("%d listed images are missing, e.g. %s" % (len(missing), missing[0]))

    sessions = images.groupby("session").agg(
        images=("image", "size"), pedestrians=("n_pedestrian", "sum"),
        time=("time", "first"), weather=("weather", "first"),
        times=("time", "nunique"), weathers=("weather", "nunique"))
    if (sessions[["times", "weathers"]] > 1).any(axis=None):
        raise ValueError("A session mixes time/weather values; stratification assumption broken")
    for column, value in (("rainy", "rainy"), ("foggy", "foggy")):
        sessions[column] = np.where(sessions["weather"] == value, sessions["images"], 0)
    sessions["night"] = np.where(sessions["time"] == "night", sessions["images"], 0)

    best = None
    for seed in range(args.search_seeds):
        split = assign(sessions, ratios, seed)
        value = score(sessions, split, ratios)
        if best is None or value < best[0]:
            best = (value, seed, split)
    error, seed, split = best
    images["split"] = images["session"].map(split)

    list_names = {s: "split_%s_%s.txt" % (args.version, s) for s in SPLITS}
    yaml_path = args.data / ("dataset_%s.yaml" % args.version)
    targets = [args.data / n for n in list_names.values()] + [yaml_path]
    if any(t.exists() for t in targets) and not args.overwrite:
        raise FileExistsError("Split version already exists; choose a new --version or pass --overwrite")
    for s in SPLITS:
        paths = images.loc[images["split"] == s, "path"].sort_values()
        (args.data / list_names[s]).write_text("\n".join(paths) + "\n", encoding="utf-8")
    base = (args.data / "dataset.yaml").read_text(encoding="utf-8").splitlines()
    names = base[[i for i, line in enumerate(base) if line.startswith("names:")][0]:]
    yaml_path.write_text("\n".join(["path: " + args.data.resolve().as_posix()]
                                   + ["%s: %s" % (s, list_names[s]) for s in SPLITS] + names) + "\n",
                         encoding="utf-8")

    report = args.report_dir / args.version
    report.mkdir(parents=True, exist_ok=True)
    images.drop(columns=["path"]).to_csv(report / "split_manifest.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(report / "excluded_images.csv", index=False, encoding="utf-8-sig")
    table = lambda column: images.groupby(["split", column]).size().unstack(fill_value=0).to_dict(orient="index")
    by_split = images.groupby("split")
    summary = {
        "version": args.version, "seed": seed, "ratios": ratios, "balance_error": round(error, 4),
        "dataset_yaml": str(yaml_path), "excluded_images": sorted(excluded["image"].unique()),
        "sessions": by_split["session"].nunique().to_dict(),
        "images": by_split.size().to_dict(),
        "pedestrians": by_split["n_pedestrian"].sum().astype(int).to_dict(),
        "session_split_overlap": int((images.groupby("session")["split"].nunique() > 1).sum()),
        "weather_images": table("weather"), "time_images": table("time"), "road_images": table("road"),
        "moved_from_source_split": pd.crosstab(images["source_split"], images["split"]).to_dict(orient="index"),
        "limitation": "Night and fog co-occur in every night session of this data; their effects cannot be separated.",
    }
    (report / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=int),
                                               encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("seed", "balance_error", "sessions", "images", "pedestrians",
                                              "session_split_overlap", "weather_images", "time_images")},
                     indent=1, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
