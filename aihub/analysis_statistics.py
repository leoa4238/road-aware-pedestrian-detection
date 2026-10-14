"""Descriptive test-set statistics, not detector training or causal inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from error_analysis_core import clustered_boundary_contrast, session_id


def hypothesis_analysis(pedestrians, repetitions):
    overall = clustered_boundary_contrast(pedestrians, repetitions)
    strata = []
    for height, group in pedestrians.groupby("height_bin"):
        result = clustered_boundary_contrast(group, repetitions)
        result["height_bin"] = height
        strata.append(result)
    pd_strata = pd.DataFrame(strata)
    support = pedestrians.copy()
    support["boundary_abs_bbox_norm"] = support["support_signed_boundary_bbox_norm"].abs()
    sensitivity = clustered_boundary_contrast(support, repetitions)
    sessions = pedestrians.copy()
    sessions["clip"] = sessions["clip"].map(session_id)
    session_contrast = clustered_boundary_contrast(sessions, repetitions)
    result = {"primary_unadjusted": overall, "height_strata": strata,
              "support_point_sensitivity": sensitivity,
              "session_cluster_sensitivity": session_contrast,
              "interpretation": "Exploratory association using estimated masks; not physical-curb or causal evidence."}
    return result, pd_strata


def logistic_failure_analysis(pedestrians, repetitions=200, seed=42):
    required = ["boundary_abs_bbox_norm", "h", "is_tp", "time", "weather", "clip"]
    frame = pedestrians[required].dropna().copy()
    frame = frame[(frame["h"] > 0) & np.isfinite(frame["boundary_abs_bbox_norm"])]
    target = 1 - frame["is_tp"].to_numpy(int)
    if len(frame) < 30 or len(np.unique(target)) < 2 or frame["clip"].nunique() < 2:
        return {"status": "insufficient_data"}
    features = pd.DataFrame({"log1p_abs_boundary_bbox_norm": np.log1p(frame["boundary_abs_bbox_norm"]),
                             "log_bbox_height_px": np.log(frame["h"]),
                             "night": (frame["time"] == "night").astype(float)})
    weather = pd.get_dummies(frame["weather"], prefix="weather", dtype=float)
    # clear is the prespecified reference; unseen codes are not silently treated as clear.
    weather = weather.drop(columns=["weather_clear"], errors="ignore")
    features = pd.concat([features, weather], axis=1)
    variable = features.std(ddof=0) > 0
    features = features.loc[:, variable]
    means, scales = features.mean(), features.std(ddof=0)
    x = ((features - means) / scales).to_numpy()
    augmented_rank = int(np.linalg.matrix_rank(np.column_stack([np.ones(len(x)), x])))
    collinear_pairs = [[features.columns[i], features.columns[j]]
                       for i in range(x.shape[1]) for j in range(i + 1, x.shape[1])
                       if np.allclose(x[:, i], x[:, j]) or np.allclose(x[:, i], -x[:, j])]
    model = LogisticRegression(C=1., penalty="l2", solver="lbfgs", max_iter=1000)
    model.fit(x, target)
    clips = frame["clip"].to_numpy()
    unique, inverse = np.unique(clips, return_inverse=True)
    rng = np.random.default_rng(seed)
    coefficients = []
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[inverse]
        if len(np.unique(target[weights > 0])) < 2:
            continue
        fit = LogisticRegression(C=1., penalty="l2", solver="lbfgs", max_iter=1000)
        fit.fit(x, target, sample_weight=weights)
        coefficients.append(fit.coef_[0])
    intervals = np.quantile(coefficients, [.025, .975], axis=0) if coefficients else None
    rows = []
    for i, name in enumerate(features.columns):
        row = {"feature": name, "coefficient_per_std": float(model.coef_[0, i]),
               "odds_ratio_per_std": float(np.exp(model.coef_[0, i])),
               "input_mean": float(means[name]), "input_std": float(scales[name])}
        if intervals is not None:
            row.update(coefficient_ci_low=float(intervals[0, i]), coefficient_ci_high=float(intervals[1, i]))
        rows.append(row)
    return {"status": "exploratory_regularized_association", "gt_used": len(frame), "fn": int(target.sum()),
            "clips": len(unique), "bootstrap_valid_replicates": len(coefficients), "C": 1.,
            "design_rank": augmented_rank, "design_columns_including_intercept": x.shape[1] + 1,
            "rank_deficient": augmented_rank < x.shape[1] + 1, "perfectly_collinear_pairs": collinear_pairs,
            "terms": rows, "limitations": "Test-set explanatory fit only. Estimated mask, residual confounding, within-session dependence and L2 shrinkage limit causal claims."}
