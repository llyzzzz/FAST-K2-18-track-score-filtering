#!/usr/bin/env python3
"""Audit one-to-one injection recovery for adopted, point, and safe rules."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustered", required=True, type=Path)
    parser.add_argument("--injections", required=True, type=Path)
    parser.add_argument("--drift-tolerance", required=True, type=float)
    parser.add_argument("--retained-column", default="score_cluster_retained")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def as_bool(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).to_numpy()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - half_width, center + half_width


def greedy_matches(
    injections: pd.DataFrame,
    frequency: np.ndarray,
    drift: np.ndarray,
    retained: np.ndarray,
    drift_tolerance: float,
) -> list[tuple[int, int]]:
    positions = np.flatnonzero(retained & np.isfinite(frequency) & np.isfinite(drift))
    used: set[int] = set()
    matches: list[tuple[int, int]] = []
    for injection_index, injection in injections.iterrows():
        injection_frequency = float(injection["f_start_MHz"])
        injection_drift = float(injection["drift_rate_Hz_per_s"])
        compatible = positions[
            (np.abs(frequency[positions] - injection_frequency) * 1.0e6 <= 22.5)
            & (np.abs(drift[positions] - injection_drift) <= drift_tolerance)
        ]
        compatible = np.asarray([p for p in compatible if int(p) not in used], dtype=int)
        if len(compatible) == 0:
            continue
        ranking = (
            np.abs(frequency[compatible] - injection_frequency) * 1.0e6
            + 100.0 * np.abs(drift[compatible] - injection_drift)
        )
        candidate_position = int(compatible[int(np.argmin(ranking))])
        used.add(candidate_position)
        matches.append((int(injection_index), candidate_position))
    return matches


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.clustered)
    injections = pd.read_csv(args.injections)
    if "beam_id" in injections.columns:
        injections = injections[injections["beam_id"].astype(str).eq("M01")]
    injections = injections.reset_index(drop=True)

    frequency = pd.to_numeric(candidates["start_freq_MHz"], errors="coerce").to_numpy(float)
    drift = pd.to_numeric(candidates["drift_rate_Hz_per_sec"], errors="coerce").to_numpy(float)
    vote = pd.to_numeric(candidates["beam_track_vote_score"], errors="coerce").fillna(0.0)
    continuity = pd.to_numeric(candidates["beam_track_continuity"], errors="coerce").fillna(0.0)
    if args.retained_column not in candidates.columns:
        raise ValueError(f"Missing retained column: {args.retained_column}")
    adopted = as_bool(candidates[args.retained_column])
    point = ((continuity >= 0.35) | ((continuity >= 0.20) & (vote >= 0.05))).to_numpy()
    safe = adopted | point

    rows = []
    matched_tables = []
    for rule_name, mask in (("adopted_hierarchy", adopted), ("point_only", point), ("safe_union", safe)):
        matches = greedy_matches(
            injections, frequency, drift, mask, args.drift_tolerance
        )
        lower, upper = wilson(len(matches), len(injections))
        rows.append(
            {
                "rule": rule_name,
                "input_candidates": int(len(candidates)),
                "retained_candidates": int(mask.sum()),
                "injections": int(len(injections)),
                "recovered": int(len(matches)),
                "recovery_percent": 100.0 * len(matches) / len(injections),
                "wilson_low_percent": 100.0 * lower,
                "wilson_high_percent": 100.0 * upper,
                "drift_tolerance_hz_s": args.drift_tolerance,
                "retained_column": args.retained_column,
            }
        )
        matched_tables.extend(
            {"rule": rule_name, "injection_row": injection, "candidate_row": candidate}
            for injection, candidate in matches
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    pd.DataFrame(matched_tables).to_csv(
        args.output.with_name(args.output.stem + "_matches.csv"), index=False
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
