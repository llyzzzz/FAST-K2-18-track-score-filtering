#!/usr/bin/env python3
"""Run the seven-beam BLISS search on one uninjected K2-18 epoch."""

import argparse
import os
from pathlib import Path

import blissdedrift as bliss
import pandas as pd


BEAMS = ("M01", "M08", "M10", "M12", "M14", "M16", "M18")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--snr-threshold", type=float, default=10.0)
    parser.add_argument("--max-drift", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size > 0 and not args.force:
        print(f"Skipping completed output: {args.output}", flush=True)
        return
    lock_path = args.output.with_suffix(args.output.suffix + ".running")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"Skipping output already in progress: {args.output}", flush=True)
        return
    os.close(lock_fd)

    try:
        run_search(args)
    finally:
        lock_path.unlink(missing_ok=True)


def run_search(args: argparse.Namespace) -> None:
    paths = [args.data_dir / f"K2-18b_{beam}.h5" for beam in BEAMS]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    cadence = bliss.cadence([[str(path)] for path in paths])
    cadence.set_device(args.device)

    noise_options = bliss.estimators.noise_power_estimate_options()
    noise_options.masked_estimate = True
    noise_options.estimator_method = bliss.estimators.noise_power_estimator.stddev
    working = bliss.estimators.estimate_noise_power(cadence, noise_options)

    drift_options = bliss.integrate_drifts_options()
    drift_options.desmear = True
    drift_options.low_rate_Hz_per_sec = -args.max_drift
    drift_options.high_rate_Hz_per_sec = args.max_drift
    working = bliss.drift_search.integrate_drifts(working, drift_options)
    working.set_device(args.device)

    hit_options = bliss.drift_search.hit_search_options()
    hit_options.snr_threshold = args.snr_threshold
    hits = bliss.drift_search.hit_search(working, hit_options)

    result = pd.DataFrame(bliss.plot_utils.get_hits_list(hits))
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} hits to {args.output}", flush=True)


if __name__ == "__main__":
    main()
