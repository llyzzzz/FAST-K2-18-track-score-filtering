#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local parameter-sensitivity audit for the FAST K2-18b frozen-HDBSCAN pipeline.

Purpose
-------
Test whether the adopted candidate-reduction / injection-recovery trade-off
depends critically on a narrowly tuned parameter value.

This is NOT a hyperparameter optimization script.

Each non-baseline configuration changes only ONE parameter relative to the
adopted configuration.

Science data:
    used to fit robust scaling + HDBSCAN + cluster-level retention map.

Injection data:
    mapped into the frozen science-derived hierarchy with approximate_predict;
    never used to fit scaling or HDBSCAN.

Recovery matching:
    same sequential greedy one-to-one logic as audit_recovery.py.
"""

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import hdbscan


# ============================================================
# Adopted manuscript parameters
# ============================================================

BASE = {
    "min_cluster_size": 50,
    "min_samples": 20,

    "cluster_strong_cont": 0.75,
    "cluster_mid_cont": 0.60,
    "cluster_mid_vote": 0.10,

    "point_strong_cont": 0.35,
    "point_mid_cont": 0.20,
    "point_mid_vote": 0.05,
}

DEFAULT_FREQ_TOL_HZ = 22.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--science",
        required=True,
        type=Path,
        help="Science track-score CSV."
    )

    p.add_argument(
        "--injection",
        required=True,
        type=Path,
        help="Injection track-score CSV."
    )

    p.add_argument(
        "--injections",
        required=True,
        type=Path,
        help="Injection ground-truth catalogue."
    )

    p.add_argument(
        "--drift-tolerance",
        required=True,
        type=float,
        help="Recovery drift tolerance in Hz/s."
    )

    p.add_argument(
        "--freq-tolerance-hz",
        type=float,
        default=DEFAULT_FREQ_TOL_HZ,
        help="Recovery frequency tolerance in Hz."
    )

    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV."
    )

    return p.parse_args()


def as_bool_array(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "t"})
        .to_numpy()
    )


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return 0.0, 0.0

    p = successes / total

    denominator = 1.0 + z * z / total

    center = (
        p + z * z / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )

    return center - half_width, center + half_width


def load_injections(path):
    df = pd.read_csv(path)

    required = [
        "f_start_MHz",
        "drift_rate_Hz_per_s",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Injection catalogue missing {col}. "
                f"Columns={list(df.columns)}"
            )

    if "beam_id" in df.columns:
        df = df[
            df["beam_id"].astype(str).eq("M01")
        ]

    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No M01 injections found.")

    return df


def load_scores(path):
    df = pd.read_csv(path)

    required = [
        "start_freq_MHz",
        "drift_rate_Hz_per_sec",
        "beam_track_vote_score",
        "beam_track_continuity",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{path} missing {col}. "
                f"Columns={list(df.columns)}"
            )

    return df


def robust_fit(values):
    median = np.nanmedian(values, axis=0)

    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)

    scale = q75 - q25

    # Same safety logic: avoid division by zero.
    scale = np.where(
        np.isfinite(scale) & (scale > 0),
        scale,
        1.0
    )

    return median, scale


def point_rule(vote, cont, cfg):
    return (
        (cont >= cfg["point_strong_cont"])
        |
        (
            (cont >= cfg["point_mid_cont"])
            &
            (vote >= cfg["point_mid_vote"])
        )
    )


def cluster_rule(summary, cfg):
    return (
        (~summary["cluster_is_noise"])
        &
        (
            (summary["median_continuity"]
             >= cfg["cluster_strong_cont"])
            |
            (
                (summary["median_continuity"]
                 >= cfg["cluster_mid_cont"])
                &
                (summary["median_vote"]
                 >= cfg["cluster_mid_vote"])
            )
        )
    )


def fit_frozen_hierarchy(science, injection, cfg):
    """
    Fit scaling + HDBSCAN only on valid science rows.
    Map injection rows into frozen science-derived hierarchy.
    """

    s_vote = pd.to_numeric(
        science["beam_track_vote_score"],
        errors="coerce"
    ).to_numpy(float)

    s_cont = pd.to_numeric(
        science["beam_track_continuity"],
        errors="coerce"
    ).to_numpy(float)

    i_vote = pd.to_numeric(
        injection["beam_track_vote_score"],
        errors="coerce"
    ).to_numpy(float)

    i_cont = pd.to_numeric(
        injection["beam_track_continuity"],
        errors="coerce"
    ).to_numpy(float)

    s_valid = np.isfinite(s_vote) & np.isfinite(s_cont)
    i_valid = np.isfinite(i_vote) & np.isfinite(i_cont)

    science_values = np.column_stack([
        s_vote[s_valid],
        s_cont[s_valid],
    ])

    median, scale = robust_fit(science_values)

    science_scaled = (
        science_values - median
    ) / scale

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        min_samples=cfg["min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    science_labels_valid = clusterer.fit_predict(
        science_scaled
    )

    # --------------------------------------------------------
    # Build full science labels
    # --------------------------------------------------------

    science_labels = np.full(
        len(science),
        -999,
        dtype=int
    )

    science_labels[s_valid] = science_labels_valid

    # --------------------------------------------------------
    # Science cluster summary
    # --------------------------------------------------------

    rows = []

    for label in np.unique(science_labels_valid):
        mask_valid = (
            science_labels_valid == label
        )

        rows.append({
            "score_cluster": int(label),
            "cluster_size": int(mask_valid.sum()),
            "median_vote": float(
                np.median(s_vote[s_valid][mask_valid])
            ),
            "median_continuity": float(
                np.median(s_cont[s_valid][mask_valid])
            ),
            "cluster_is_noise": bool(label == -1),
        })

    summary = pd.DataFrame(rows)

    summary["cluster_retained"] = cluster_rule(
        summary,
        cfg
    )

    retained_cluster_map = {
        int(row["score_cluster"]):
            bool(row["cluster_retained"])
        for _, row in summary.iterrows()
    }

    # --------------------------------------------------------
    # Science point rule
    # --------------------------------------------------------

    science_point = np.zeros(
        len(science),
        dtype=bool
    )

    science_point[s_valid] = point_rule(
        s_vote[s_valid],
        s_cont[s_valid],
        cfg
    )

    science_cluster_keep = np.zeros(
        len(science),
        dtype=bool
    )

    science_noise = np.zeros(
        len(science),
        dtype=bool
    )

    for idx in np.flatnonzero(s_valid):
        label = int(science_labels[idx])

        if label == -1:
            science_noise[idx] = True
        else:
            science_cluster_keep[idx] = (
                retained_cluster_map.get(
                    label,
                    False
                )
            )

    science_hierarchy = (
        science_cluster_keep
        |
        (science_noise & science_point)
    )

    # --------------------------------------------------------
    # Frozen injection prediction
    # --------------------------------------------------------

    injection_labels = np.full(
        len(injection),
        -999,
        dtype=int
    )

    injection_strengths = np.full(
        len(injection),
        np.nan,
        dtype=float
    )

    if i_valid.any():
        injection_values = np.column_stack([
            i_vote[i_valid],
            i_cont[i_valid],
        ])

        injection_scaled = (
            injection_values - median
        ) / scale

        predicted_labels, strengths = (
            hdbscan.approximate_predict(
                clusterer,
                injection_scaled
            )
        )

        injection_labels[i_valid] = (
            predicted_labels
        )

        injection_strengths[i_valid] = (
            strengths
        )

    injection_point = np.zeros(
        len(injection),
        dtype=bool
    )

    injection_point[i_valid] = point_rule(
        i_vote[i_valid],
        i_cont[i_valid],
        cfg
    )

    injection_cluster_keep = np.zeros(
        len(injection),
        dtype=bool
    )

    injection_noise = np.zeros(
        len(injection),
        dtype=bool
    )

    for idx in np.flatnonzero(i_valid):
        label = int(
            injection_labels[idx]
        )

        if label == -1:
            injection_noise[idx] = True
        else:
            injection_cluster_keep[idx] = (
                retained_cluster_map.get(
                    label,
                    False
                )
            )

    injection_hierarchy = (
        injection_cluster_keep
        |
        (injection_noise & injection_point)
    )

    return {
        "science_point": science_point,
        "science_hierarchy": science_hierarchy,
        "injection_point": injection_point,
        "injection_hierarchy": injection_hierarchy,
        "science_clusters": len(
            summary[summary["score_cluster"] != -1]
        ),
        "science_noise_rows": int(
            (science_labels_valid == -1).sum()
        ),
        "scale_median_vote": median[0],
        "scale_median_cont": median[1],
        "scale_iqr_vote": scale[0],
        "scale_iqr_cont": scale[1],
    }


def greedy_matches(
    injections,
    frequency,
    drift,
    retained,
    drift_tolerance,
    frequency_tolerance_hz,
):
    """
    Same matching logic as audit_recovery.py.
    """

    positions = np.flatnonzero(
        retained
        & np.isfinite(frequency)
        & np.isfinite(drift)
    )

    used = set()
    matches = []

    for injection_index, injection_row in injections.iterrows():

        injection_frequency = float(
            injection_row["f_start_MHz"]
        )

        injection_drift = float(
            injection_row["drift_rate_Hz_per_s"]
        )

        compatible = positions[
            (
                np.abs(
                    frequency[positions]
                    - injection_frequency
                )
                * 1.0e6
                <= frequency_tolerance_hz
            )
            &
            (
                np.abs(
                    drift[positions]
                    - injection_drift
                )
                <= drift_tolerance
            )
        ]

        compatible = np.asarray([
            p
            for p in compatible
            if int(p) not in used
        ], dtype=int)

        if len(compatible) == 0:
            continue

        ranking = (
            np.abs(
                frequency[compatible]
                - injection_frequency
            )
            * 1.0e6
            +
            100.0
            * np.abs(
                drift[compatible]
                - injection_drift
            )
        )

        candidate_position = int(
            compatible[
                int(np.argmin(ranking))
            ]
        )

        used.add(candidate_position)

        matches.append(
            (
                int(injection_index),
                candidate_position,
            )
        )

    return matches


def build_configs():
    """
    One-factor-at-a-time local perturbations.
    No Cartesian grid search.
    """

    configs = []

    # Baseline
    configs.append(
        ("baseline", deepcopy(BASE))
    )

    def add(name, key, value):
        cfg = deepcopy(BASE)
        cfg[key] = value
        configs.append(
            (name, cfg)
        )

    # HDBSCAN
    add(
        "mcs_30",
        "min_cluster_size",
        30
    )

    add(
        "mcs_70",
        "min_cluster_size",
        70
    )

    add(
        "minsamples_10",
        "min_samples",
        10
    )

    add(
        "minsamples_30",
        "min_samples",
        30
    )

    # Cluster thresholds
    add(
        "cluster_strong_cont_070",
        "cluster_strong_cont",
        0.70
    )

    add(
        "cluster_strong_cont_080",
        "cluster_strong_cont",
        0.80
    )

    add(
        "cluster_mid_cont_055",
        "cluster_mid_cont",
        0.55
    )

    add(
        "cluster_mid_cont_065",
        "cluster_mid_cont",
        0.65
    )

    add(
        "cluster_mid_vote_008",
        "cluster_mid_vote",
        0.08
    )

    add(
        "cluster_mid_vote_012",
        "cluster_mid_vote",
        0.12
    )

    # Point thresholds
    add(
        "point_strong_cont_030",
        "point_strong_cont",
        0.30
    )

    add(
        "point_strong_cont_040",
        "point_strong_cont",
        0.40
    )

    add(
        "point_mid_cont_015",
        "point_mid_cont",
        0.15
    )

    add(
        "point_mid_cont_025",
        "point_mid_cont",
        0.25
    )

    add(
        "point_mid_vote_003",
        "point_mid_vote",
        0.03
    )

    add(
        "point_mid_vote_007",
        "point_mid_vote",
        0.07
    )

    return configs


def main():
    args = parse_args()

    science = load_scores(
        args.science
    )

    injection = load_scores(
        args.injection
    )

    injections = load_injections(
        args.injections
    )

    injection_frequency = pd.to_numeric(
        injection["start_freq_MHz"],
        errors="coerce"
    ).to_numpy(float)

    injection_drift = pd.to_numeric(
        injection["drift_rate_Hz_per_sec"],
        errors="coerce"
    ).to_numpy(float)

    rows = []

    configs = build_configs()

    print(
        f"Running {len(configs)} "
        f"one-factor-at-a-time configurations..."
    )

    for index, (name, cfg) in enumerate(
        configs,
        start=1
    ):
        print(
            f"[{index}/{len(configs)}] {name}"
        )

        result = fit_frozen_hierarchy(
            science,
            injection,
            cfg
        )

        hierarchy_matches = greedy_matches(
            injections,
            injection_frequency,
            injection_drift,
            result["injection_hierarchy"],
            args.drift_tolerance,
            args.freq_tolerance_hz,
        )

        point_matches = greedy_matches(
            injections,
            injection_frequency,
            injection_drift,
            result["injection_point"],
            args.drift_tolerance,
            args.freq_tolerance_hz,
        )

        hierarchy_recovered = len(
            hierarchy_matches
        )

        point_recovered = len(
            point_matches
        )

        low, high = wilson(
            hierarchy_recovered,
            len(injections)
        )

        rows.append({
            "configuration": name,

            "min_cluster_size":
                cfg["min_cluster_size"],

            "min_samples":
                cfg["min_samples"],

            "cluster_strong_cont":
                cfg["cluster_strong_cont"],

            "cluster_mid_cont":
                cfg["cluster_mid_cont"],

            "cluster_mid_vote":
                cfg["cluster_mid_vote"],

            "point_strong_cont":
                cfg["point_strong_cont"],

            "point_mid_cont":
                cfg["point_mid_cont"],

            "point_mid_vote":
                cfg["point_mid_vote"],

            "science_input":
                len(science),

            "science_point_retained":
                int(
                    result[
                        "science_point"
                    ].sum()
                ),

            "science_hierarchy_retained":
                int(
                    result[
                        "science_hierarchy"
                    ].sum()
                ),

            "injection_input":
                len(injection),

            "injection_point_retained":
                int(
                    result[
                        "injection_point"
                    ].sum()
                ),

            "injection_hierarchy_retained":
                int(
                    result[
                        "injection_hierarchy"
                    ].sum()
                ),

            "point_recovered":
                point_recovered,

            "hierarchy_recovered":
                hierarchy_recovered,

            "point_recovery_percent":
                100.0
                * point_recovered
                / len(injections),

            "hierarchy_recovery_percent":
                100.0
                * hierarchy_recovered
                / len(injections),

            "hierarchy_wilson_low_percent":
                100.0 * low,

            "hierarchy_wilson_high_percent":
                100.0 * high,

            "science_clusters":
                result["science_clusters"],
        })

    output = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Add differences relative to baseline
    # --------------------------------------------------------

    baseline = output[
        output["configuration"]
        == "baseline"
    ].iloc[0]

    output[
        "delta_science_hierarchy_rows"
    ] = (
        output["science_hierarchy_retained"]
        - baseline["science_hierarchy_retained"]
    )

    output[
        "delta_injection_hierarchy_rows"
    ] = (
        output["injection_hierarchy_retained"]
        - baseline["injection_hierarchy_retained"]
    )

    output[
        "delta_hierarchy_recovered"
    ] = (
        output["hierarchy_recovered"]
        - baseline["hierarchy_recovered"]
    )

    output[
        "delta_hierarchy_recovery_pp"
    ] = (
        output["hierarchy_recovery_percent"]
        - baseline["hierarchy_recovery_percent"]
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        args.output,
        index=False
    )

    json_path = args.output.with_suffix(
        ".json"
    )

    with open(
        json_path,
        "w"
    ) as f:
        json.dump(
            output.to_dict(
                orient="records"
            ),
            f,
            indent=2
        )

    print()
    print(
        output[
            [
                "configuration",
                "science_hierarchy_retained",
                "injection_hierarchy_retained",
                "hierarchy_recovered",
                "hierarchy_recovery_percent",
                "delta_hierarchy_recovered",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"[OK] CSV  : {args.output}")
    print(f"[OK] JSON : {json_path}")


if __name__ == "__main__":
    main()
