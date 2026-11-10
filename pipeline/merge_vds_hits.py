#!/usr/bin/env python3
"""Merge overlapping frequency-slice hit tables using disjoint core channels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--hits-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    pieces = []
    for row in manifest.itertuples(index=False):
        frame = pd.read_csv(args.hits_dir / f"part_{int(row.part):02d}.csv")
        frequency = pd.to_numeric(frame["start_freq_MHz"], errors="coerce").to_numpy(float)
        channel = np.rint(
            (frequency - float(row.original_fch1_mhz)) / float(row.original_foff_mhz)
        ).astype(np.int64)
        keep = (channel >= int(row.core_start_channel)) & (channel < int(row.core_stop_channel))
        core = frame.loc[keep].copy()
        core["origin"] = f"K2-18b_M14_vds_part_{int(row.part):02d}.h5"
        core["beam_id"] = "M14"
        pieces.append(core)
    merged = pd.concat(pieces, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged):,} core-region hits to {args.output}", flush=True)


if __name__ == "__main__":
    main()
