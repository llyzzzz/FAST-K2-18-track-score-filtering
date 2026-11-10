#!/usr/bin/env python3
"""Run the fixed BLISS search for one beam and write an auditable hit table."""

import argparse
from pathlib import Path

import blissdedrift as bliss
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--beam", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-drift", type=float, default=5.0)
    parser.add_argument("--snr-threshold", type=float, default=10.0)
    parser.add_argument("--no-desmear", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size > 0:
        print(f"Skipping completed output: {args.output}", flush=True)
        return

    cadence = bliss.cadence([[str(args.input)]])
    cadence.set_device(args.device)
    noise = bliss.estimators.noise_power_estimate_options()
    noise.masked_estimate = True
    noise.estimator_method = bliss.estimators.noise_power_estimator.stddev
    working = bliss.estimators.estimate_noise_power(cadence, noise)
    drift = bliss.integrate_drifts_options()
    drift.desmear = not args.no_desmear
    drift.low_rate_Hz_per_sec = -args.max_drift
    drift.high_rate_Hz_per_sec = args.max_drift
    working = bliss.drift_search.integrate_drifts(working, drift)
    working.set_device(args.device)
    options = bliss.drift_search.hit_search_options()
    options.snr_threshold = args.snr_threshold
    hits = bliss.drift_search.hit_search(working, options)
    frame = pd.DataFrame(bliss.plot_utils.get_hits_list(hits))
    frame["beam_id"] = args.beam
    frame["input_file"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} {args.beam} hits to {args.output}", flush=True)


if __name__ == "__main__":
    main()
