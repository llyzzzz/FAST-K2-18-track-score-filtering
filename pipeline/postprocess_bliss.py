#!/usr/bin/env python3
"""Band-limit and multibeam-vet a BLISS hit table with audited tolerances."""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


CENTER_BEAM = "M01"
OFF_BEAMS = ("M08", "M10", "M12", "M14", "M16", "M18")
FREQ_RESOLUTION_HZ = 7.450580596923828125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hits", required=True, type=Path)
    parser.add_argument("--m01", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--drift-tolerance", default="one-bin")
    return parser.parse_args()


def extract_beam(origin: object) -> str | None:
    match = re.search(r"(M\d{2})", str(origin))
    return match.group(1) if match else None


def infer_drift_step(frame: pd.DataFrame) -> float:
    values = np.unique(pd.to_numeric(frame["drift_rate_Hz_per_sec"], errors="coerce"))
    values = values[np.isfinite(values)]
    differences = np.diff(np.sort(values))
    differences = differences[differences > 1.0e-10]
    if len(differences) == 0:
        raise ValueError("Could not infer a BLISS drift-grid step")
    return float(np.median(differences))


def resolve_drift_tolerance(spec: str, grid_step: float) -> float:
    if spec == "one-bin":
        return grid_step * (1.0 + 1.0e-6)
    return float(spec)


def multibeam_preselection(frame: pd.DataFrame, drift_tolerance: float) -> pd.DataFrame:
    work = frame.copy()
    work["beam"] = work["origin"].map(extract_beam)
    work = work.dropna(subset=["beam"])
    work["f_start_Hz"] = work["start_freq_MHz"] * 1.0e6

    beam_arrays: dict[str, np.ndarray] = {}
    for beam, group in work.groupby("beam"):
        array = group[["f_start_Hz", "drift_rate_Hz_per_sec"]].to_numpy(float)
        beam_arrays[beam] = array[np.argsort(array[:, 0])]

    if CENTER_BEAM not in beam_arrays:
        raise ValueError("M01 is absent from the BLISS output")
    missing = sorted(set(OFF_BEAMS) - set(beam_arrays))
    if missing:
        raise ValueError(f"Missing OFF beams in BLISS output: {missing}")

    center = work[work["beam"] == CENTER_BEAM].copy()
    order = np.argsort(center["f_start_Hz"].to_numpy())
    sorted_center = center[["f_start_Hz", "drift_rate_Hz_per_sec"]].to_numpy(float)[order]
    keep = np.ones(len(center), dtype=bool)
    frequency_tolerance = 3.0 * FREQ_RESOLUTION_HZ

    for sorted_position, (frequency, drift) in enumerate(sorted_center):
        matched = False
        for beam in OFF_BEAMS:
            reference = beam_arrays[beam]
            left = np.searchsorted(reference[:, 0], frequency - frequency_tolerance, side="left")
            right = np.searchsorted(reference[:, 0], frequency + frequency_tolerance, side="right")
            if left < right and np.any(
                np.abs(reference[left:right, 1] - drift) <= drift_tolerance
            ):
                matched = True
                break
        if matched:
            keep[order[sorted_position]] = False

    return center.loc[keep].drop(columns=["beam", "f_start_Hz"], errors="ignore")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hits = pd.read_csv(args.hits)
    band_limited = hits[
        hits["start_freq_MHz"].between(1050.0, 1450.0, inclusive="both")
    ].copy()
    grid_step = infer_drift_step(band_limited)
    drift_tolerance = resolve_drift_tolerance(args.drift_tolerance, grid_step)
    candidates = multibeam_preselection(band_limited, drift_tolerance)

    with h5py.File(args.m01, "r") as handle:
        data = handle["data"]
        n_time = int(data.shape[0])
        tsamp = float(data.attrs["tsamp"])
    elapsed_span = (n_time - 1) * tsamp
    candidates["end_freq_MHz"] = (
        candidates["start_freq_MHz"]
        + candidates["drift_rate_Hz_per_sec"] * elapsed_span / 1.0e6
    )

    band_limited.to_csv(args.out_dir / "bliss_hits_1050_1450.csv", index=False)
    candidates.to_csv(args.out_dir / "science_candidates_preselected.csv", index=False)
    audit = {
        "initial_bliss_hits_all_seven_beams": int(len(hits)),
        "initial_bliss_hits_m01": int(
            hits["origin"].map(extract_beam).eq(CENTER_BEAM).sum()
        ),
        "after_band_edge_exclusion_all_seven_beams": int(len(band_limited)),
        "m01_hits_after_band_edge_exclusion": int(
            band_limited["origin"].map(extract_beam).eq(CENTER_BEAM).sum()
        ),
        "after_multibeam_preselection": int(len(candidates)),
        "frequency_tolerance_hz": 3.0 * FREQ_RESOLUTION_HZ,
        "drift_grid_step_hz_s": grid_step,
        "drift_tolerance_hz_s": drift_tolerance,
        "n_time_bins": n_time,
        "time_sampling_s": tsamp,
        "first_to_last_bin_span_s": elapsed_span,
    }
    (args.out_dir / "stagewise_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
