#!/usr/bin/env python3
"""Create frequency-disjoint HDF5 virtual datasets without copying spectra."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--overlap-channels", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as source_file:
        source_data = source_file["data"]
        shape = source_data.shape
        source = h5py.VirtualSource(str(args.input), "data", shape=shape)
        boundaries = np.linspace(0, shape[-1], args.parts + 1, dtype=int)
        manifest = []

        for index, (core_start, core_stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            start = max(0, core_start - args.overlap_channels)
            stop = min(shape[-1], core_stop + args.overlap_channels)
            output = args.out_dir / f"part_{index:02d}.h5"
            layout = h5py.VirtualLayout(shape=shape[:-1] + (stop - start,), dtype=source_data.dtype)
            layout[...] = source[..., start:stop]
            with h5py.File(output, "w", libver="latest") as target:
                for key, value in source_file.attrs.items():
                    target.attrs[key] = value
                dataset = target.create_virtual_dataset("data", layout, fillvalue=0)
                for key, value in source_data.attrs.items():
                    dataset.attrs[key] = value
                dataset.attrs["fch1"] = float(source_data.attrs["fch1"]) + start * float(
                    source_data.attrs["foff"]
                )
                dataset.attrs["nchans"] = stop - start
            print(f"{output}: channels {start}:{stop}", flush=True)
            manifest.append(
                {
                    "part": index,
                    "path": str(output),
                    "source_start_channel": start,
                    "source_stop_channel": stop,
                    "core_start_channel": int(core_start),
                    "core_stop_channel": int(core_stop),
                    "original_fch1_mhz": float(source_data.attrs["fch1"]),
                    "original_foff_mhz": float(source_data.attrs["foff"]),
                }
            )
        pd.DataFrame(manifest).to_csv(args.out_dir / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
