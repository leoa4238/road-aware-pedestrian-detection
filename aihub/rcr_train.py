"""Train and compare RCR-Head variants on cached features with session-grouped cross-validation.

Labels
    --labels attributes_pilot_v0.csv   condition labels (criteria v1); risk = rule v1 (rcr_head.v1_risk)
    --labels labels_partial_*.tsv      researcher v0 labels (3 risky / 2 caution / 1 safe / 0 excluded)

Every variant sees exactly the same folds (StratifiedGroupKFold by session, repeated over seeds),
so differences between variants are paired. A rule baseline ("foot on road = risky") is reported
on the same people.

Example:
    python aihub/rcr_train.py --features data/rcr/pilot_v0_features.npz \
        --labels data/labeling/pilot_v0/attributes_pilot_v0.csv --out outputs/aihub/rcr/pilot_v0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcr_head import (BINARY_CONDITIONS, DIRECTION_VALUES, POSITION_VALUES, RASTER_HW, RCRConfig,  # noqa: E402
                      RCRHead, rcr_loss, v1_risk)

RISKY = 2

# name -> config. Order = the table in docs/RCR_HEAD_DESIGN.md.
VARIANTS = {
    "G (geometry)":              dict(use_appearance=False, use_raster=False, condition_heads=False),
    "R (road raster)":           dict(use_appearance=False, use_geometry=False, condition_heads=False),
    "R+G":                       dict(use_appearance=False, condition_heads=False),
    "A (appearance)":            dict(use_raster=False, use_geometry=False, condition_heads=False),
    "A+R+G concat":              dict(fusion="concat", condition_heads=False),
    "A+R+G film":                dict(fusion="film", condition_heads=False),
    "A+R+G film + conditions":   dict(fusion="film", condition_heads=True),          # full RCR-Head
}


# ------------------------------------------------------------------------- data
def load_labels(path):
    path = Path(path)
    if path.suffix.lower() == ".tsv":                       # v0: one overall judgement
        df = pd.read_csv(path, sep="\t")
        df = df[df["label"] > 0].copy()
        df["risk"] = df["label"].astype(int) - 1
        for k in ("position", "direction"):
            df[k] = -1
        for k in BINARY_CONDITIONS:
            df[k] = np.nan
        return df[["id", "risk", "position", "direction", *BINARY_CONDITIONS]]
    df = pd.read_csv(path, dtype={"position": "string", "direction": "string"})
    df = df[(df.get("excluded", 0) == 0) & df["position"].notna() & df["direction"].notna()].copy()
    df["direction"] = df["direction"].str.lower()
    df["risk"] = [v1_risk(p, d, o) for p, d, o in zip(df["position"], df["direction"], df["occluded"])]
    df["position"] = df["position"].map({v: i for i, v in enumerate(POSITION_VALUES)}).fillna(-1).astype(int)
    df["direction"] = df["direction"].map({v: i for i, v in enumerate(DIRECTION_VALUES)}).fillna(-1).astype(int)
    for k in BINARY_CONDITIONS:
        df[k] = df[k].astype(float) if k in df else np.nan
    return df[["id", "risk", "position", "direction", *BINARY_CONDITIONS]]


class Data:
    def __init__(self, features_path, labels):
        with np.load(features_path, allow_pickle=False) as f:      # closes the file (Windows)
            z = {k: f[k] for k in f.files}
        ids = pd.Series(np.arange(len(z["id"])), index=z["id"].astype(str))
        labels = labels[labels["id"].astype(str).isin(ids.index)].reset_index(drop=True)
        idx = ids.loc[labels["id"].astype(str)].to_numpy()
        self.labels = labels
        self.session = z["session"][idx]
        self.app = [torch.from_numpy(z[k][idx].astype(np.float32)) for k in ("p3", "p4", "p5")]
        self.app_flip = ([torch.from_numpy(z[k + "_flip"][idx].astype(np.float32)) for k in ("p3", "p4", "p5")]
                         if "p3_flip" in z else None)
        self.raster = torch.from_numpy(z["raster"][idx].astype(np.float32))
        self.geometry = torch.from_numpy(z["geometry"][idx].astype(np.float32))
        self.y = torch.as_tensor(labels["risk"].to_numpy(), dtype=torch.long)
        self.position = torch.as_tensor(labels["position"].to_numpy(), dtype=torch.long)
        self.direction = torch.as_tensor(labels["direction"].to_numpy(), dtype=torch.long)
        self.binary = torch.as_tensor(labels[list(BINARY_CONDITIONS)].to_numpy(dtype=float), dtype=torch.float32)
        self.channels = tuple(a.shape[1] for a in self.app)

    def __len__(self):
        return len(self.y)

    def batch(self, idx, flip=None):
        """Inputs for rows idx; flip is a bool tensor (per row) or None."""
        app = [a[idx] for a in self.app]
        raster, geom = self.raster[idx], self.geometry[idx].clone()
        if flip is not None and flip.any():
            raster = raster.clone()
            raster[flip] = raster[flip].flip(-1)
            geom[flip, 0] = 1 - geom[flip, 0]
            geom[flip, 4] = -geom[flip, 4]
            if self.app_flip is not None:
                app = [torch.where(flip[:, None, None, None], af[idx], a) for a, af in zip(app, self.app_flip)]
        return dict(appearance=app, raster=raster, geometry=geom)

    def targets(self, idx):
        return dict(risk=self.y[idx], position=self.position[idx], direction=self.direction[idx],
                    binary=self.binary[idx])

    def foot_on_road(self):
        """Road share of the raster cell under the foot (same idea as the pilot rule)."""
        h, w = RASTER_HW
        c = (self.geometry[:, 0] * w).long().clamp(0, w - 1)
        r = (self.geometry[:, 1] * h).long().clamp(0, h - 1)
        return self.raster[torch.arange(len(self)), 0, r, c] >= 0.5


# ---------------------------------------------------------------------- training
def train_one(data, train_idx, cfg, epochs, lr, weight_decay, batch_size, seed, device, cond_weight):
    torch.manual_seed(seed)
    model = RCRHead(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    counts = torch.bincount(data.y[train_idx], minlength=3).float()
    weight = (counts.sum() / (3 * counts.clamp(min=1))).to(device)      # balanced classes
    gen = torch.Generator().manual_seed(seed)
    train_idx = torch.as_tensor(train_idx)
    for _ in range(epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=gen)]
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:                   # BatchNorm needs > 1 sample
                continue
            flip = torch.rand(len(idx), generator=gen) < 0.5
            inputs = to_device(data.batch(idx, flip), device)
            loss = rcr_loss(model(**inputs), to_device(data.targets(idx), device), weight, cond_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model, data, idx, device):
    model.eval()
    out = model(**to_device(data.batch(torch.as_tensor(idx)), device))
    pred = {"risk": out["risk"].argmax(1).cpu().numpy()}
    if "position" in out:
        pos = out["position"].argmax(1).cpu().numpy()
        dirn = out["direction"].argmax(1).cpu().numpy()
        occ = (out["binary"][:, BINARY_CONDITIONS.index("occluded")] > 0).cpu().numpy()
        pred.update(position=pos, direction=dirn,
                    risk_via_conditions=np.array([v1_risk(POSITION_VALUES[p], DIRECTION_VALUES[d], o)
                                                  for p, d, o in zip(pos, dirn, occ)]))
    return pred


def to_device(d, device):
    return {k: ([t.to(device) for t in v] if isinstance(v, list) else v.to(device)) for k, v in d.items()}


def scores(y, p):
    return {"macro_f1": f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0),
            "risky_recall": recall_score(y == RISKY, p == RISKY, zero_division=0),
            "risky_precision": precision_score(y == RISKY, p == RISKY, zero_division=0)}


def summarize(per_seed):
    """per_seed: list of metric dicts (one per seed, computed on that seed's pooled out-of-fold predictions)."""
    keys = per_seed[0].keys()
    return {k: {"mean": round(float(np.mean([s[k] for s in per_seed])), 4),
                "sd": round(float(np.std([s[k] for s in per_seed])), 4)} for k in keys}


def run(args):
    labels = load_labels(args.labels)
    data = Data(args.features, labels)
    y = data.y.numpy()
    print(f"[data] {len(data)} labeled pedestrians, {len(set(data.session))} sessions, "
          f"classes safe/caution/risky = {np.bincount(y, minlength=3).tolist()}, appearance channels {data.channels}")
    if len(data) < 30:
        raise SystemExit("Too few labeled pedestrians matched the feature cache.")

    folds = []
    for seed in range(args.seeds):
        cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        folds.append(list(cv.split(np.zeros(len(y)), y, data.session)))

    rule = np.where(data.foot_on_road().numpy(), RISKY, 0)
    result = {"n": len(data), "sessions": int(len(set(data.session))),
              "class_counts": dict(zip(["safe", "caution", "risky"], np.bincount(y, minlength=3).tolist())),
              "settings": {k: str(v) for k, v in vars(args).items()},
              "rule: foot on road = risky": summarize([scores(y, rule)]),
              "variants": {}}
    names = [n for n in VARIANTS if not args.only or n in args.only]
    for name in names:
        cfg = RCRConfig(appearance_channels=data.channels, dim=args.dim, dropout=args.dropout, **VARIANTS[name])
        per_seed, per_seed_cond, cond_acc = [], [], []
        for seed, seed_folds in enumerate(folds):
            oof = np.empty(len(y), dtype=int)
            oof_cond = np.empty(len(y), dtype=int) if cfg.condition_heads else None
            pos_ok, dir_ok, n_pos, n_dir = 0, 0, 0, 0
            for k, (tr, te) in enumerate(seed_folds):
                model = train_one(data, tr, cfg, args.epochs, args.lr, args.weight_decay, args.batch_size,
                                  seed * 100 + k, args.device, args.cond_weight)
                pred = predict(model, data, te, args.device)
                oof[te] = pred["risk"]
                if oof_cond is not None:
                    oof_cond[te] = pred["risk_via_conditions"]
                    pt, dt = data.position[te].numpy(), data.direction[te].numpy()
                    pos_ok += int((pred["position"] == pt)[pt >= 0].sum()); n_pos += int((pt >= 0).sum())
                    dir_ok += int((pred["direction"] == dt)[dt >= 0].sum()); n_dir += int((dt >= 0).sum())
            per_seed.append(scores(y, oof))
            if oof_cond is not None:
                per_seed_cond.append(scores(y, oof_cond))
                cond_acc.append({"position_acc": pos_ok / max(n_pos, 1), "direction_acc": dir_ok / max(n_dir, 1)})
        entry = {"config": cfg.name(), "params": sum(p.numel() for p in RCRHead(cfg).parameters()),
                 "risk": summarize(per_seed), "_per_seed_macro_f1": [round(s["macro_f1"], 4) for s in per_seed]}
        if per_seed_cond:
            entry["risk_via_conditions"] = summarize(per_seed_cond)
            entry["conditions"] = summarize(cond_acc)
        result["variants"][name] = entry
        r = entry["risk"]
        print(f"{name:28s} macro-F1 {r['macro_f1']['mean']:.3f}±{r['macro_f1']['sd']:.3f}  "
              f"risky recall {r['risky_recall']['mean']:.3f}  precision {r['risky_precision']['mean']:.3f}")

    # Paired comparisons on the same folds (difference of per-seed macro-F1).
    pairs = [("A+R+G film", "A+R+G concat"), ("A+R+G film", "A (appearance)"), ("A+R+G film", "R+G"),
             ("A+R+G film + conditions", "A+R+G film")]
    result["paired_macro_f1_difference"] = {}
    for a, b in pairs:
        if a in result["variants"] and b in result["variants"]:
            d = np.array(result["variants"][a]["_per_seed_macro_f1"]) - np.array(result["variants"][b]["_per_seed_macro_f1"])
            result["paired_macro_f1_difference"][f"{a} - {b}"] = {
                "mean": round(float(d.mean()), 4), "sd": round(float(d.std()), 4), "seeds_positive": int((d > 0).sum())}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="variant names to run (default: all)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--cond-weight", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[saved] {args.out}")
    print(json.dumps(result["paired_macro_f1_difference"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
