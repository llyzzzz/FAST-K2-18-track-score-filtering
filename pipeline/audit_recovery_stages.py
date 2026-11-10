#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic stage-by-stage injection recovery audit for the K2-18b pipeline.

Supported candidate files
-------------------------
1. Early / intermediate stages:
   bliss_hits_1000_1500.csv
   bliss_hits_1050_1450.csv
   science_candidates_preselected.csv
   injection_track_scores_one-bin.csv

   All eligible M01 rows in the file are treated as having passed that stage.

2. Frozen/scored stages:
   injection_frozen_clustered.csv
   injection_frozen_retained.csv

   If beam-track scores and an adopted hierarchy-retention column are present,
   the following rules are audited:
       adopted_hierarchy
       point_only
       safe_union

Recovery matcher
----------------
Sequential greedy one-to-one matching in injection-catalog order, following
the original audit_recovery.py logic:

    |Delta f| <= frequency tolerance
    |Delta drift| <= drift tolerance

If multiple unmatched candidate rows satisfy the tolerances, choose the one
minimizing

    |Delta f_Hz| + 100 * |Delta drift_Hz_s|

Each candidate row can be used at most once.

The script never modifies source CSV files.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


POINT_CONT_STRONG = 0.35
POINT_CONT_MID = 0.20
POINT_VOTE_MID = 0.05

DEFAULT_FREQ_TOL_HZ = 22.5

REQUIRED_CANDIDATE_COLUMNS = (
    "start_freq_MHz",
    "drift_rate_Hz_per_sec",
)

REQUIRED_INJECTION_COLUMNS = (
    "f_start_MHz",
    "drift_rate_Hz_per_s",
)

HIERARCHY_COLUMNS = (
    "frozen_hierarchy_retained",
    "score_cluster_retained",
    "adopted_retained",
)

BEAM_COLUMNS = (
    "beam_id",
    "beam",
    "Beam",
    "beam_name",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Candidate CSV from any pipeline stage.",
    )

    p.add_argument(
        "--injections",
        required=True,
        type=Path,
        help="Ground-truth injection catalogue.",
    )

    p.add_argument(
        "--drift-tolerance",
        required=True,
        type=float,
        help="Injection-recovery drift tolerance in Hz/s.",
    )

    p.add_argument(
        "--freq-tolerance-hz",
        type=float,
        default=DEFAULT_FREQ_TOL_HZ,
        help="Injection-recovery frequency tolerance in Hz (default 22.5).",
    )

    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output summary CSV.",
    )

    p.add_argument(
        "--stage-name",
        default=None,
        help="Optional human-readable stage name.",
    )

    p.add_argument(
        "--retained-column",
        default=None,
        help="Explicit adopted hierarchy retained column.",
    )

    p.add_argument(
        "--target-beam",
        default="M01",
        help="Injected target beam (default M01).",
    )

    p.add_argument(
        "--beam-column",
        default=None,
        help="Explicit candidate beam column name if present.",
    )

    p.add_argument(
        "--assume-m01",
        action="store_true",
        help=(
            "Fallback only: treat candidate file as M01-only if no beam "
            "identity can be detected from a beam column or origin."
        ),
    )

    p.add_argument(
        "--snr-min",
        type=float,
        default=None,
        help="Optional additional SNR cut for audit only.",
    )

    return p.parse_args()


def require_columns(
    df: pd.DataFrame,
    required,
    label: str,
) -> None:
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{label} missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def as_bool(series: pd.Series) -> np.ndarray:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "t"})
        .to_numpy()
    )


def wilson(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0

    proportion = successes / total
    denominator = 1.0 + z * z / total

    center = (
        proportion + z * z / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )

    return center - half_width, center + half_width


def detect_beam_column(
    df: pd.DataFrame,
    explicit: str | None,
) -> str | None:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"--beam-column={explicit!r} not found.\n"
                f"Available columns: {list(df.columns)}"
            )
        return explicit

    for col in BEAM_COLUMNS:
        if col in df.columns:
            return col

    return None


def extract_beam_from_origin(value: str) -> str | None:
    """
    Extract FAST beam ID such as M01, M08, M10 from origin strings like:

        A:K2-18b_M01-0
        K2-18b_M08-0
        .../M14/...
        source:M18_xxx

    This intentionally does not rely on \\b because '_' is a regex word
    character and can break the expected boundary behavior.
    """

    if value is None:
        return None

    text = str(value)

    pattern = re.compile(
        r"(?:^|[^A-Za-z0-9])"
        r"(M(?:0[1-9]|1[0-9]))"
        r"(?=$|[^A-Za-z0-9])",
        flags=re.IGNORECASE,
    )

    match = pattern.search(text)

    if match is None:
        return None

    return match.group(1).upper()


def detect_origin_beams(
    df: pd.DataFrame,
) -> pd.Series | None:
    if "origin" not in df.columns:
        return None

    beams = df["origin"].apply(extract_beam_from_origin)

    if beams.notna().any():
        return beams

    return None


def filter_candidate_to_target_beam(
    df: pd.DataFrame,
    target_beam: str,
    beam_column: str | None,
    assume_m01: bool,
) -> tuple[pd.DataFrame, str]:
    """
    Beam provenance priority:

    1. explicit/detected beam column
    2. beam parsed from origin
    3. --assume-m01 fallback

    IMPORTANT:
    --assume-m01 never overrides real beam information.
    """

    target_beam = target_beam.upper()

    # 1. Dedicated beam column has highest priority.
    if beam_column is not None:
        beam_values = (
            df[beam_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        mask = beam_values.eq(target_beam)

        out = df.loc[mask].copy()

        if len(out) == 0:
            raise ValueError(
                f"No {target_beam} rows found in beam column "
                f"{beam_column!r}."
            )

        return out, f"beam_column:{beam_column}"

    # 2. Try parsing beam ID from origin.
    origin_beams = detect_origin_beams(df)

    if origin_beams is not None:
        mask = origin_beams.eq(target_beam)

        out = df.loc[mask].copy()

        if len(out) == 0:
            available = sorted(
                origin_beams.dropna().unique().tolist()
            )

            raise ValueError(
                f"Beam IDs were detected in 'origin', but no "
                f"{target_beam} rows were found.\n"
                f"Detected beams: {available}"
            )

        return out, "origin"

    # 3. Only now allow assume-m01 fallback.
    if assume_m01:
        return df.copy(), "assume_m01"

    raise ValueError(
        "Cannot establish M01 provenance for candidate file.\n"
        "No recognized beam column was found and no FAST beam ID could be "
        "parsed from the 'origin' column.\n"
        "If this file is known from provenance to contain only M01 rows, "
        "rerun with --assume-m01."
    )


def load_injections(
    path: Path,
    target_beam: str,
) -> pd.DataFrame:
    injections = pd.read_csv(path)

    require_columns(
        injections,
        REQUIRED_INJECTION_COLUMNS,
        "Injection catalogue",
    )

    injections = injections.copy()

    injections["_injection_original_row"] = np.arange(
        len(injections)
    )

    # If injection catalogue has beam information, keep M01 only.
    beam_col = detect_beam_column(
        injections,
        explicit=None,
    )

    if beam_col is not None:
        mask = (
            injections[beam_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq(target_beam.upper())
        )

        injections = injections.loc[mask].copy()

    elif "origin" in injections.columns:
        origin_beams = detect_origin_beams(injections)

        if origin_beams is not None:
            mask = origin_beams.eq(target_beam.upper())
            injections = injections.loc[mask].copy()

    if len(injections) == 0:
        raise ValueError(
            f"No injections remain for target beam {target_beam}."
        )

    # Preserve original audit_recovery.py behavior:
    # injection catalogue is processed sequentially after reset_index().
    injections = injections.reset_index(drop=True)

    return injections


def choose_hierarchy_column(
    df: pd.DataFrame,
    explicit: str | None,
) -> str | None:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Requested retained column {explicit!r} not found.\n"
                f"Available columns: {list(df.columns)}"
            )

        return explicit

    # Deliberately do NOT auto-select point_score_retained.
    for col in HIERARCHY_COLUMNS:
        if col in df.columns:
            return col

    return None


def determine_mode(
    df: pd.DataFrame,
    retained_column: str | None,
) -> tuple[str, str | None]:
    has_vote = "beam_track_vote_score" in df.columns
    has_cont = "beam_track_continuity" in df.columns

    hierarchy_col = choose_hierarchy_column(
        df,
        retained_column,
    )

    if has_vote and has_cont and hierarchy_col is not None:
        return "scored", hierarchy_col

    return "stage", None


def detect_snr_column(
    df: pd.DataFrame,
) -> str | None:
    for col in (
        "SNR",
        "snr",
        "signal_to_noise",
    ):
        if col in df.columns:
            return col

    return None


def greedy_matches(
    injections: pd.DataFrame,
    candidate_frequency: np.ndarray,
    candidate_drift: np.ndarray,
    retained: np.ndarray,
    drift_tolerance: float,
    frequency_tolerance_hz: float,
) -> list[tuple[int, int]]:
    """
    Same matching order and tie behavior as original audit_recovery.py.

    injections:
        processed sequentially in catalogue order

    candidates:
        remain in original CSV row order

    candidate reuse:
        forbidden

    ranking:
        |Delta f_Hz| + 100 * |Delta drift|
    """

    positions = np.flatnonzero(
        retained
        & np.isfinite(candidate_frequency)
        & np.isfinite(candidate_drift)
    )

    used: set[int] = set()

    matches: list[tuple[int, int]] = []

    for injection_index, injection in injections.iterrows():
        injection_frequency = float(
            injection["f_start_MHz"]
        )

        injection_drift = float(
            injection["drift_rate_Hz_per_s"]
        )

        compatible = positions[
            (
                np.abs(
                    candidate_frequency[positions]
                    - injection_frequency
                )
                * 1.0e6
                <= frequency_tolerance_hz
            )
            &
            (
                np.abs(
                    candidate_drift[positions]
                    - injection_drift
                )
                <= drift_tolerance
            )
        ]

        compatible = np.asarray(
            [
                p
                for p in compatible
                if int(p) not in used
            ],
            dtype=int,
        )

        if len(compatible) == 0:
            continue

        ranking = (
            np.abs(
                candidate_frequency[compatible]
                - injection_frequency
            )
            * 1.0e6
            +
            100.0
            * np.abs(
                candidate_drift[compatible]
                - injection_drift
            )
        )

        # np.argmin preserves first occurrence, and compatible is still in
        # original candidate-row order, matching audit_recovery.py behavior.
        candidate_position = int(
            compatible[int(np.argmin(ranking))]
        )

        used.add(candidate_position)

        matches.append(
            (
                int(injection_index),
                candidate_position,
            )
        )

    return matches


def make_match_table(
    injections: pd.DataFrame,
    candidates: pd.DataFrame,
    matches: list[tuple[int, int]],
    stage: str,
    rule: str,
) -> pd.DataFrame:
    rows = []

    for inj_row, cand_row in matches:
        inj = injections.iloc[inj_row]
        cand = candidates.iloc[cand_row]

        delta_freq_hz = (
            float(cand["start_freq_MHz"])
            - float(inj["f_start_MHz"])
        ) * 1.0e6

        delta_drift = (
            float(cand["drift_rate_Hz_per_sec"])
            - float(inj["drift_rate_Hz_per_s"])
        )

        row = {
            "stage": stage,
            "rule": rule,
            "injection_row": inj_row,
            "injection_original_row": inj.get(
                "_injection_original_row",
                np.nan,
            ),
            "candidate_row": cand_row,
            "injection_freq_MHz": float(
                inj["f_start_MHz"]
            ),
            "candidate_freq_MHz": float(
                cand["start_freq_MHz"]
            ),
            "delta_freq_Hz": delta_freq_hz,
            "injection_drift_Hz_s": float(
                inj["drift_rate_Hz_per_s"]
            ),
            "candidate_drift_Hz_s": float(
                cand["drift_rate_Hz_per_sec"]
            ),
            "delta_drift_Hz_s": delta_drift,
            "match_rank_metric": (
                abs(delta_freq_hz)
                + 100.0 * abs(delta_drift)
            ),
        }

        for col in (
            "SNR",
            "snr",
            "width",
            "width_channels",
            "beam_id",
        ):
            if col in injections.columns:
                row[f"injection_{col}"] = inj[col]

        rows.append(row)

    return pd.DataFrame(rows)


def make_missed_table(
    injections: pd.DataFrame,
    matches: list[tuple[int, int]],
    stage: str,
    rule: str,
) -> pd.DataFrame:
    matched_rows = {
        int(inj_row)
        for inj_row, _ in matches
    }

    missed_indices = [
        i
        for i in range(len(injections))
        if i not in matched_rows
    ]

    missed = injections.iloc[
        missed_indices
    ].copy()

    missed.insert(
        0,
        "injection_row",
        missed_indices,
    )

    missed.insert(
        0,
        "rule",
        rule,
    )

    missed.insert(
        0,
        "stage",
        stage,
    )

    return missed


def main() -> None:
    args = parse_args()

    candidates_all = pd.read_csv(
        args.candidate
    )

    require_columns(
        candidates_all,
        REQUIRED_CANDIDATE_COLUMNS,
        "Candidate file",
    )

    total_file_rows = len(
        candidates_all
    )

    beam_column = detect_beam_column(
        candidates_all,
        args.beam_column,
    )

    candidates, beam_provenance = (
        filter_candidate_to_target_beam(
            candidates_all,
            target_beam=args.target_beam,
            beam_column=beam_column,
            assume_m01=args.assume_m01,
        )
    )

    candidates = candidates.reset_index(
        drop=True
    )

    candidate_rows_after_m01_filter = len(
        candidates
    )

    injections = load_injections(
        args.injections,
        args.target_beam,
    )

    frequency = pd.to_numeric(
        candidates["start_freq_MHz"],
        errors="coerce",
    ).to_numpy(float)

    drift = pd.to_numeric(
        candidates["drift_rate_Hz_per_sec"],
        errors="coerce",
    ).to_numpy(float)

    finite = (
        np.isfinite(frequency)
        & np.isfinite(drift)
    )

    mode, retained_column = determine_mode(
        candidates,
        args.retained_column,
    )

    snr_mask = np.ones(
        len(candidates),
        dtype=bool,
    )

    if args.snr_min is not None:
        snr_column = detect_snr_column(
            candidates
        )

        if snr_column is None:
            raise ValueError(
                "--snr-min was requested, but no recognized "
                "SNR column exists."
            )

        snr = pd.to_numeric(
            candidates[snr_column],
            errors="coerce",
        ).to_numpy(float)

        snr_mask &= (
            np.isfinite(snr)
            & (snr >= args.snr_min)
        )

    if mode == "scored":
        vote = pd.to_numeric(
            candidates[
                "beam_track_vote_score"
            ],
            errors="coerce",
        ).fillna(0.0).to_numpy(float)

        continuity = pd.to_numeric(
            candidates[
                "beam_track_continuity"
            ],
            errors="coerce",
        ).fillna(0.0).to_numpy(float)

        adopted = as_bool(
            candidates[
                retained_column
            ]
        )

        point = (
            (continuity >= POINT_CONT_STRONG)
            |
            (
                (continuity >= POINT_CONT_MID)
                & (vote >= POINT_VOTE_MID)
            )
        )

        safe = adopted | point

        rules = [
            (
                "adopted_hierarchy",
                adopted
                & finite
                & snr_mask,
            ),
            (
                "point_only",
                point
                & finite
                & snr_mask,
            ),
            (
                "safe_union",
                safe
                & finite
                & snr_mask,
            ),
        ]

    else:
        rules = [
            (
                "stage_pass",
                finite & snr_mask,
            )
        ]

    stage = (
        args.stage_name
        or args.candidate.stem
    )

    summary_rows = []
    all_matches = []
    all_missed = []

    for rule_name, mask in rules:
        matches = greedy_matches(
            injections=injections,
            candidate_frequency=frequency,
            candidate_drift=drift,
            retained=mask,
            drift_tolerance=args.drift_tolerance,
            frequency_tolerance_hz=args.freq_tolerance_hz,
        )

        recovered = len(matches)
        total_injections = len(injections)
        missed = (
            total_injections - recovered
        )

        lower, upper = wilson(
            recovered,
            total_injections,
        )

        summary_rows.append(
            {
                "stage": stage,
                "candidate_file": args.candidate.name,
                "mode": mode,
                "rule": rule_name,
                "total_file_rows": total_file_rows,
                "candidate_rows_after_M01_filter":
                    candidate_rows_after_m01_filter,
                "retained_candidates": int(
                    mask.sum()
                ),
                "injections": total_injections,
                "recovered": recovered,
                "missed": missed,
                "recovery_percent":
                    100.0
                    * recovered
                    / total_injections,
                "wilson_low_percent":
                    100.0 * lower,
                "wilson_high_percent":
                    100.0 * upper,
                "frequency_tolerance_hz":
                    args.freq_tolerance_hz,
                "drift_tolerance_hz_s":
                    args.drift_tolerance,
                "retained_column":
                    retained_column
                    if rule_name
                    == "adopted_hierarchy"
                    else "",
                "beam_provenance":
                    beam_provenance,
                "target_beam":
                    args.target_beam,
                "snr_min":
                    args.snr_min
                    if args.snr_min is not None
                    else "",
            }
        )

        match_table = make_match_table(
            injections,
            candidates,
            matches,
            stage,
            rule_name,
        )

        missed_table = make_missed_table(
            injections,
            matches,
            stage,
            rule_name,
        )

        all_matches.append(
            match_table
        )

        all_missed.append(
            missed_table
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        args.output,
        index=False,
    )

    matches_path = (
        args.output.with_name(
            args.output.stem
            + "_matches.csv"
        )
    )

    missed_path = (
        args.output.with_name(
            args.output.stem
            + "_missed.csv"
        )
    )

    if all_matches:
        pd.concat(
            all_matches,
            ignore_index=True,
        ).to_csv(
            matches_path,
            index=False,
        )
    else:
        pd.DataFrame().to_csv(
            matches_path,
            index=False,
        )

    if all_missed:
        pd.concat(
            all_missed,
            ignore_index=True,
        ).to_csv(
            missed_path,
            index=False,
        )
    else:
        pd.DataFrame().to_csv(
            missed_path,
            index=False,
        )

    print(
        json.dumps(
            summary_rows,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    print(
        f"[OK] summary : "
        f"{args.output}"
    )

    print(
        f"[OK] matches : "
        f"{matches_path}"
    )

    print(
        f"[OK] missed  : "
        f"{missed_path}"
    )


if __name__ == "__main__":
    main()
