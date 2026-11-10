#!/usr/bin/env python3
"""Fit HDBSCAN on science scores and predict injection rows without refitting."""

import argparse
import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd


VOTE = "beam_track_vote_score"
CONT = "beam_track_continuity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science", required=True, type=Path)
    parser.add_argument("--injection", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=20)
    return parser.parse_args()


def score_array(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame[[VOTE, CONT]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values >= 0.0).all(axis=1)
    return values, valid


def point_rule(frame: pd.DataFrame) -> np.ndarray:
    vote = pd.to_numeric(frame[VOTE], errors="coerce").fillna(0.0).to_numpy(float)
    cont = pd.to_numeric(frame[CONT], errors="coerce").fillna(0.0).to_numpy(float)
    return (cont >= 0.35) | ((cont >= 0.20) & (vote >= 0.05))


def cluster_summary(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["score_cluster"] >= -1].copy()
    summary = (
        valid.groupby("score_cluster", sort=True)
        .agg(
            cluster_size=("score_cluster", "size"),
            median_vote=(VOTE, "median"),
            median_continuity=(CONT, "median"),
        )
        .reset_index()
    )
    summary["cluster_is_noise"] = summary["score_cluster"].eq(-1)
    summary["cluster_retained"] = (~summary["cluster_is_noise"]) & (
        (summary["median_continuity"] >= 0.75)
        | (
            (summary["median_continuity"] >= 0.60)
            & (summary["median_vote"] >= 0.10)
        )
    )
    return summary


def annotate_science(
    frame: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    valid: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    result["score_cluster"] = -99
    result.loc[valid, "score_cluster"] = labels
    result["score_cluster_prob"] = np.nan
    result.loc[valid, "score_cluster_prob"] = probabilities
    summary = cluster_summary(result)
    keep_map = dict(zip(summary["score_cluster"], summary["cluster_retained"]))
    point = point_rule(result)
    cluster_keep = result["score_cluster"].map(keep_map).fillna(False).to_numpy(bool)
    noise = result["score_cluster"].eq(-1).to_numpy()
    result["point_score_retained"] = point
    result["frozen_hierarchy_retained"] = cluster_keep | (noise & point)
    result["retention_path"] = "rejected"
    result.loc[cluster_keep, "retention_path"] = "science_cluster_rule"
    result.loc[noise & point, "retention_path"] = "science_noise_point_rule"
    return result, summary


def annotate_injection(
    frame: pd.DataFrame,
    clusterer: hdbscan.HDBSCAN,
    median: np.ndarray,
    scale: np.ndarray,
    keep_map: dict[int, bool],
) -> pd.DataFrame:
    result = frame.copy()
    values, valid = score_array(result)
    labels = np.full(len(result), -99, dtype=int)
    strengths = np.full(len(result), np.nan, dtype=float)
    predicted_labels, predicted_strengths = hdbscan.approximate_predict(
        clusterer, (values[valid] - median) / scale
    )
    labels[valid] = predicted_labels.astype(int)
    strengths[valid] = predicted_strengths
    result["score_cluster"] = labels
    result["score_cluster_prob"] = strengths
    point = point_rule(result)
    cluster_keep = np.asarray([keep_map.get(int(label), False) for label in labels])
    noise = labels == -1
    result["point_score_retained"] = point
    result["frozen_hierarchy_retained"] = cluster_keep | (noise & point)
    result["retention_path"] = "rejected"
    result.loc[cluster_keep, "retention_path"] = "frozen_science_cluster_rule"
    result.loc[noise & point, "retention_path"] = "predicted_noise_point_rule"
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    science = pd.read_csv(args.science)
    injection = pd.read_csv(args.injection)
    science_values, science_valid = score_array(science)
    science_fit = science_values[science_valid]
    median = np.nanmedian(science_fit, axis=0)
    q25 = np.nanquantile(science_fit, 0.25, axis=0)
    q75 = np.nanquantile(science_fit, 0.75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict((science_fit - median) / scale)
    science_out, summary = annotate_science(
        science, labels, clusterer.probabilities_, science_valid
    )
    keep_map = dict(zip(summary["score_cluster"], summary["cluster_retained"]))
    injection_out = annotate_injection(injection, clusterer, median, scale, keep_map)

    science_out.to_csv(args.out_dir / "science_frozen_clustered.csv", index=False)
    science_out[science_out["frozen_hierarchy_retained"]].to_csv(
        args.out_dir / "science_frozen_retained.csv", index=False
    )
    injection_out.to_csv(args.out_dir / "injection_frozen_clustered.csv", index=False)
    injection_out[injection_out["frozen_hierarchy_retained"]].to_csv(
        args.out_dir / "injection_frozen_retained.csv", index=False
    )
    summary.to_csv(args.out_dir / "science_cluster_summary.csv", index=False)

    audit = {
        "science_input": int(len(science_out)),
        "science_valid_scores": int(science_valid.sum()),
        "science_clusters": int((summary["score_cluster"] >= 0).sum()),
        "science_nonnoise_cluster_rows": int((science_out["score_cluster"] >= 0).sum()),
        "science_point_score_retained": int(science_out["point_score_retained"].sum()),
        "science_frozen_hierarchy_retained": int(
            science_out["frozen_hierarchy_retained"].sum()
        ),
        "injection_input": int(len(injection_out)),
        "injection_predicted_nonnoise_rows": int((injection_out["score_cluster"] >= 0).sum()),
        "injection_point_score_retained": int(injection_out["point_score_retained"].sum()),
        "injection_frozen_hierarchy_retained": int(
            injection_out["frozen_hierarchy_retained"].sum()
        ),
        "scale_median": median.tolist(),
        "scale_iqr": scale.tolist(),
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
    }
    (args.out_dir / "frozen_model_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
