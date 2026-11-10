#!/usr/bin/env python3
"""R2 normalization comparison, fitted exclusively on uninjected science data.

The only representation change is a fixed, science-derived affine transform
for each beam. OFF arithmetic, track scores and retention rules remain those
of the adopted primary pipeline. No BLISS search or pre-selection is run.
HDF5 files are opened read-only; transformed frequency blocks exist in memory.

Configuration JSON (one epoch):
  epoch, science_beams {M01,M08,M10,M12,M14,M16,M18: HDF5 path},
  injection_m01, science_candidates, injection_candidates, injections,
  raw_baseline_dir (published/independently verified frozen model products),
  input_review {science_inputs_are_uninjected: true,
                upstream_selection_status: "verified", evidence: [path,...]}.
The candidate inputs must contain the original raw score columns as well as
start/end frequency and drift. The review gate deliberately prevents an
unverified July18 M14 selection from silently entering a new comparison.

Sampling reproduces the archived normalization builder's channel selection,
chunk traversal, float32 input conversion, sample truncation, and finite-only
median/MAD/std fallback. Its Welford OFF arithmetic is NOT reused: normalized
OFF uses the primary float32 nanmean/nanstd(ddof=0), with no maximum-OFF floor.
Thus this new experiment is not a replay of the old injection self-fit table.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd


BEAMS = ("M01", "M08", "M10", "M12", "M14", "M16", "M18")
PIPELINE = Path(__file__).resolve().parents[1]
VOTE, CONT = "beam_track_vote_score", "beam_track_continuity"
HISTORICAL_STATS_SOURCE_SHA256 = "29bfe8b853f7c00b091cb2ecdfb2d41db22b86e95df25ce4bd162a15132c06dc"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path, hash_contents=False):
    p = Path(path).resolve(strict=True)
    st = p.stat()
    result = {"path": str(p), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if hash_contents:
        result["sha256"] = digest_file(p)
    return result


def parameters(args):
    return {"statistics_frequency_ranges_mhz": [[1050.0, 1450.0]],
            "max_stat_samples": args.max_stat_samples,
            "stats_freq_chunk": args.stats_freq_chunk,
            "coarse_mhz": args.coarse_mhz, "window_pad_hz": 100.0,
            "sigma_vote_hz": 15.0, "cutoff_sigma": 3.0,
            "continuity_core_hz": 15.0, "k": 2.0, "ddof": 0,
            "floor_off_max": False, "adaptive_sigma": False,
            "off_arithmetic": "float32 numpy.nanmean/nanstd(axis=0,ddof=0)",
            "min_cluster_size": args.min_cluster_size, "min_samples": args.min_samples,
            "hdbscan_version_required": "0.8.40"}


def channel_range(low_mhz, high_mhz, fch1, foff, nchans):
    a, b = (min(low_mhz, high_mhz) - fch1) / foff, (max(low_mhz, high_mhz) - fch1) / foff
    start = max(0, int(np.floor(min(a, b))))
    end = min(nchans, int(np.ceil(max(a, b))) + 1)
    if start >= end:
        raise ValueError("Statistics interval does not overlap the data")
    return start, end


def robust_location_scale(samples):
    """Exact archived arithmetic, with additional explicit fallback accounting."""
    samples = np.asarray(samples, dtype=np.float64)
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        raise ValueError("No finite statistics samples")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    method = "1.4826*MAD"
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(finite))
        method = "population_std_fallback"
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(np.float32(scale)) or np.float32(scale) <= 0:
        raise ValueError("Invalid intensity scale after MAD and standard-deviation fallback")
    return {"median": median, "mad": mad, "robust_sigma": scale,
            "scale_method": method, "sample_count": int(len(samples)),
            "finite_sample_count": int(len(finite)),
            "nonfinite_sample_count": int(len(samples) - len(finite)),
            "selected_float32_values_sha256": hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()}


def estimate_statistics(readers, frequency_ranges, max_samples=1_000_000, chunk_channels=100_000):
    """Same selected values/order as the builder; use an HDF5 hyperslab to avoid
    allocating all unselected channels. Counts include pre-filter nonfinites.
    """
    ref = readers["M01"]
    ranges = [channel_range(lo, hi, ref.fch1, ref.foff, ref.nfreq) for lo, hi in frequency_ranges]
    total_channels = sum(end - start for start, end in ranges)
    nonfrequency = int(np.prod(ref.shape[:-1]))
    target_channels = max(1, int(max_samples // max(nonfrequency, 1)))
    stride = max(1, int(np.ceil(total_channels / target_channels)))
    samples = {beam: [] for beam in BEAMS}
    cumulative = 0
    selected_indices = []
    for start, end in ranges:
        for left in range(start, end, chunk_channels):
            right = min(end, left + chunk_channels)
            offset = (-cumulative) % stride
            if left + offset < right:
                selected_indices.extend(range(left + offset, right, stride))
                slc = [slice(None)] * ref.ndim
                slc[-1] = slice(left + offset, right, stride)
                for beam in BEAMS:
                    block = np.asarray(readers[beam].data[tuple(slc)], dtype=np.float32)
                    samples[beam].append(block.reshape(-1))
            cumulative += right - left
    result = {}
    for beam in BEAMS:
        if not samples[beam]:
            raise ValueError(f"No selected samples for {beam}")
        values = np.concatenate(samples[beam])[:max_samples]
        result[beam] = robust_location_scale(values)
    return {"beams": result, "frequency_ranges_mhz": frequency_ranges,
            "channel_ranges_exclusive_end": ranges, "frequency_stride": stride,
            "samples_per_channel": nonfrequency, "selected_channels": len(selected_indices),
            "selected_channel_indices_sha256": hashlib.sha256(np.asarray(selected_indices, dtype="<i8").tobytes()).hexdigest(),
            "max_stat_samples": max_samples, "stats_freq_chunk": chunk_channels,
            "reference_algorithm_sha256": HISTORICAL_STATS_SOURCE_SHA256,
            "fit_population": "uninjected science beams only"}


def normalize_block(array, statistics):
    """Keep float32 subtract/divide and the original nonfinite propagation."""
    array = np.asarray(array, dtype=np.float32)
    median = np.float32(statistics["median"])
    sigma = np.float32(statistics["robust_sigma"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Invalid frozen float32 intensity scale")
    return (array - median) / sigma


def primary_upper(normalized_off):
    """The adopted primary envelope, including its nonfinite handling."""
    stacked = np.stack(normalized_off, axis=0).astype(np.float32, copy=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0, ddof=0)
        upper = mean + 2.0 * std
    counts = {"nonfinite_off_values": int((~np.isfinite(stacked)).sum()),
              "all_nan_off_pixels": int(np.isnan(stacked).all(axis=0).sum()),
              "nonfinite_upper_pixels": int((~np.isfinite(upper)).sum()),
              "runtime_warnings": sorted(set(str(w.message) for w in caught))}
    return upper, counts


def reader_metadata(reader):
    meta = {"shape": list(reader.shape), "dtype": str(reader.data.dtype),
            "chunks": list(reader.data.chunks) if reader.data.chunks else None,
            "fch1_mhz": reader.fch1, "foff_mhz": reader.foff}
    for key in ("tsamp", "tstart"):
        value = reader.data.attrs.get(key, reader.h5.attrs.get(key))
        if value is None:
            raise ValueError(f"Missing required time coordinate {key}: {reader.path}")
        meta[key] = float(np.asarray(value).reshape(-1)[0])
        if not np.isfinite(meta[key]):
            raise ValueError(f"Nonfinite {key}: {reader.path}")
    return meta


@contextlib.contextmanager
def open_readers(config, reference):
    readers = {}
    try:
        for beam in BEAMS:
            readers[beam] = reference.FastH5Reader(config["science_beams"][beam])
        readers["injection_M01"] = reference.FastH5Reader(config["injection_m01"])
        # A time slab spans many 4-MiB chunks; cache them across adjacent windows.
        # This affects I/O only, and never changes the underlying arrays.
        for reader in readers.values():
            reader.h5.close()
            reader.h5 = h5py.File(reader.path, "r", rdcc_nbytes=int(config.get("hdf5_cache_mib", 512)) * 1024**2,
                                  rdcc_nslots=10007)
            reader.data = reader.h5[reader.dataset_name]
        base = reader_metadata(readers["M01"])
        for name, reader in readers.items():
            meta = reader_metadata(reader)
            for key in ("shape", "fch1_mhz", "foff_mhz", "tsamp", "tstart"):
                if meta[key] != base[key]:
                    raise ValueError(f"Grid/time mismatch for {name}, {key}: {meta[key]} vs {base[key]}")
        if os.path.samefile(config["science_beams"]["M01"], config["injection_m01"]):
            raise ValueError("Science and injected M01 must be distinct files")
        yield readers
    finally:
        for reader in readers.values():
            reader.close()


def score_candidate_rows(reference, frame, rows, on, upper, frequencies):
    results = np.empty((len(rows), 4), dtype=float)
    for j, i in enumerate(rows):
        start, end = float(frame.iloc[i]["start_freq_MHz"]), float(frame.iloc[i]["end_freq_MHz"])
        lo, hi = min(start, end) - 100.0 / 1e6, max(start, end) + 100.0 / 1e6
        cols = np.flatnonzero((frequencies >= lo) & (frequencies <= hi))
        if not len(cols):
            raise ValueError(f"Candidate row {i} has no local channels")
        c0, c1 = cols[0], cols[-1] + 1
        vote, cont = reference.score_one_candidate(on[:, c0:c1], upper[:, c0:c1],
            frequencies[c0:c1], start, end, 15.0, cutoff_sigma=3.0, continuity_core_hz=15.0)
        if not np.isfinite([vote, cont]).all():
            raise ValueError(f"Invalid score at original row {i}: {vote}, {cont}")
        results[j] = (lo, hi, vote, cont)
    return results


def score_populations(readers, frames, stats, reference, out, coarse_mhz=1.0):
    groups = {}
    for pop, frame in frames.items():
        numeric = frame[["start_freq_MHz", "end_freq_MHz"]].apply(pd.to_numeric, errors="raise").to_numpy(float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Nonfinite candidate coordinates in {pop}")
        midpoint = numeric.mean(axis=1)
        keys = np.floor(midpoint / coarse_mhz) * coarse_mhz
        groups[pop] = {float(key): np.flatnonzero(keys == key) for key in np.unique(keys)}
    keys = sorted(set(groups["science"]) | set(groups["injection"]))
    parts = out / "score_groups"
    parts.mkdir(exist_ok=True)
    values = {pop: np.full((len(frame), 4), np.nan) for pop, frame in frames.items()}
    group_audit = []
    for gi, key in enumerate(keys):
        part = parts / f"group_{gi:06d}.npz"
        sr = groups["science"].get(key, np.array([], dtype=int))
        ir = groups["injection"].get(key, np.array([], dtype=int))
        if part.exists():
            with np.load(part, allow_pickle=False) as cached:
                if not np.array_equal(cached["science_rows"], sr) or not np.array_equal(cached["injection_rows"], ir):
                    raise RuntimeError("Cached group does not match the candidate ordering")
                values["science"][sr] = cached["science_scores"]
                values["injection"][ir] = cached["injection_scores"]
                group_audit.append(json.loads(str(cached["audit"].item())))
            continue
        ranges = [frames[pop].iloc[rows][["start_freq_MHz", "end_freq_MHz"]].to_numpy(float)
                  for pop, rows in (("science", sr), ("injection", ir)) if len(rows)]
        lo = min(a.min() for a in ranges) - 100.0 / 1e6
        hi = max(a.max() for a in ranges) + 100.0 / 1e6
        off = []
        frequencies = None
        for beam in BEAMS[1:]:
            block, axis, _, _ = readers[beam].read_freq_chunk(lo, hi)
            if frequencies is None:
                frequencies = axis
            elif not np.array_equal(axis, frequencies):
                raise ValueError("OFF frequency axis mismatch")
            off.append(normalize_block(block, stats["beams"][beam]))
        upper, counters = primary_upper(off)
        del off
        scores = {}
        for pop, rows, beam in (("science", sr, "M01"), ("injection", ir, "injection_M01")):
            if not len(rows):
                scores[pop] = np.empty((0, 4))
                continue
            block, axis, _, _ = readers[beam].read_freq_chunk(lo, hi)
            if not np.array_equal(axis, frequencies) or block.shape != upper.shape:
                raise ValueError("M01/OFF coordinate mismatch")
            on = normalize_block(block, stats["beams"]["M01"])
            counters[pop + "_nonfinite_m01_pixels"] = int((~np.isfinite(on)).sum())
            scores[pop] = score_candidate_rows(reference, frames[pop], rows, on, upper, frequencies)
            del block, on
            values[pop][rows] = scores[pop]
        audit = {"group": gi, "coarse_group_mhz": key, "low_mhz": lo, "high_mhz": hi,
                 "science_rows": len(sr), "injection_rows": len(ir), **counters}
        tmp = part.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, science_rows=sr, injection_rows=ir,
                     science_scores=scores["science"], injection_scores=scores["injection"],
                     audit=np.asarray(json.dumps(audit)))
        os.replace(tmp, part)
        group_audit.append(audit)
        write_json(out / "progress.json", {"stage": "scoring", "completed_groups": gi + 1,
                   "total_groups": len(keys), "latest_group": audit, "updated_unix": time.time()})
        print(f"Scored group {gi+1}/{len(keys)} at {key:.3f} MHz", flush=True)
    for pop, frame in frames.items():
        if not np.isfinite(values[pop]).all():
            raise RuntimeError(f"Uncomputed/invalid scores remain for {pop}; refusing to fit")
        result = frame.copy()
        for j, column in enumerate(("beam_freq_low_MHz", "beam_freq_high_MHz", VOTE, CONT)):
            result[column] = values[pop][:, j]
        result["beam_status"], result["beam_error"] = "ok", ""
        result.to_csv(out / f"{pop}_normalized_scores.csv", index=False)
    write_json(out / "scoring_audit.json", {"all_rows_scored": True,
        "science_input_rows": len(frames["science"]), "injection_input_rows": len(frames["injection"]),
        "invalid_score_rows": 0, "groups": group_audit,
        "intensity_stats_source": "science_intensity_statistics.json",
        "intensity_stats_sha256": digest_file(out / "science_intensity_statistics.json"),
        "normalized_score_files": {pop: identity(out / f"{pop}_normalized_scores.csv", True) for pop in frames},
        "injection_uses_exact_science_M01_statistics": True,
        "nonfinite_policy": "Stats discard nonfinites; transformed pixels retain them; primary nanmean/nanstd and comparison semantics preserved. No candidate rows silently removed."})


def fit_and_predict(science, injection, out, reference, min_cluster_size=50, min_samples=20):
    """Same fit/annotation calls as frozen_hdbscan_validation, with saved model."""
    import joblib
    out.mkdir(exist_ok=True)
    science_values, valid = reference.score_array(science)
    _, injection_valid = reference.score_array(injection)
    if not valid.all() or not injection_valid.all():
        raise ValueError("Invalid candidate scores; refused rather than silently changing the evaluated population")
    fit = science_values[valid]
    median = np.nanmedian(fit, axis=0)
    q25, q75 = np.nanquantile(fit, 0.25, axis=0), np.nanquantile(fit, 0.75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    model = reference.hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
        metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    labels = model.fit_predict((fit - median) / scale)
    science_out, summary = reference.annotate_science(science, labels, model.probabilities_, valid)
    keep = dict(zip(summary["score_cluster"], summary["cluster_retained"]))
    bundle = {"clusterer": model, "median": median.copy(), "scale": scale.copy(),
              "retention_map": keep, "fit_population": "science only"}
    joblib.dump(bundle, out / "science_model.joblib")
    saved_labels = model.labels_.copy()
    injection_out = reference.annotate_injection(injection, model, median, scale, keep)
    if not np.array_equal(saved_labels, model.labels_):
        raise RuntimeError("Science fit changed during injection prediction")
    for pop, frame in (("science", science_out), ("injection", injection_out)):
        frame.to_csv(out / f"{pop}_frozen_clustered.csv", index=False)
        frame[frame["frozen_hierarchy_retained"]].to_csv(out / f"{pop}_frozen_retained.csv", index=False)
    summary.to_csv(out / "science_cluster_summary.csv", index=False)
    audit = {"science_input": len(science), "injection_input": len(injection),
             "science_valid_scores": int(valid.sum()), "injection_valid_scores": int(injection_valid.sum()),
             "science_clusters": int((summary["score_cluster"] >= 0).sum()),
             "science_frozen_hierarchy_retained": int(science_out["frozen_hierarchy_retained"].sum()),
             "injection_frozen_hierarchy_retained": int(injection_out["frozen_hierarchy_retained"].sum()),
             "scale_median": median.tolist(), "scale_iqr": scale.tolist(),
             "min_cluster_size": min_cluster_size, "min_samples": min_samples,
             "fit_population": "science only", "prediction_membership_strength_used_for_retention": False,
             "science_labels_unchanged_after_prediction": True,
             "model_sha256": digest_file(out / "science_model.joblib")}
    write_json(out / "frozen_model_audit.json", audit)
    return science_out, injection_out, audit


def recovery(candidates, injections, drift_tolerance, reference, out):
    if "beam_id" in injections:
        injections = injections[injections["beam_id"].astype(str).eq("M01")]
    injections = injections.reset_index(drop=True)
    if not len(injections):
        raise ValueError("Empty M01 injection catalogue")
    frequency = pd.to_numeric(candidates["start_freq_MHz"], errors="raise").to_numpy(float)
    drift = pd.to_numeric(candidates["drift_rate_Hz_per_sec"], errors="raise").to_numpy(float)
    masks = {"adopted_hierarchy": reference.as_bool(candidates["frozen_hierarchy_retained"]),
             "point_only": reference.as_bool(candidates["point_score_retained"])}
    rows, matched, ids = [], [], {}
    for rule, mask in masks.items():
        pairs = reference.greedy_matches(injections, frequency, drift, mask, drift_tolerance)
        ids[rule] = set(i for i, _ in pairs)
        lo, hi = reference.wilson(len(pairs), len(injections))
        rows.append({"rule": rule, "input_candidates": len(candidates), "retained_candidates": int(mask.sum()),
                     "injections": len(injections), "recovered": len(pairs),
                     "recovery_percent": 100 * len(pairs) / len(injections),
                     "wilson_low_percent": 100 * lo, "wilson_high_percent": 100 * hi,
                     "frequency_tolerance_hz": 22.5, "drift_tolerance_hz_s": drift_tolerance})
        matched.extend({"rule": rule, "injection_row": i, "candidate_row": j} for i, j in pairs)
    pd.DataFrame(rows).to_csv(out / "injection_recovery.csv", index=False)
    pd.DataFrame(matched, columns=["rule", "injection_row", "candidate_row"]).to_csv(out / "injection_recovery_matches.csv", index=False)
    return rows, ids


def validate_config(config):
    if set(config["science_beams"]) != set(BEAMS):
        raise ValueError("Exactly M01/M08/M10/M12/M14/M16/M18 science beams are required")
    review = config.get("input_review", {})
    if review.get("science_inputs_are_uninjected") is not True:
        raise ValueError("Science input roles must be verified before normalization")
    if review.get("upstream_selection_status") != "verified":
        raise ValueError("Upstream/M14 selection review not verified; computation is gated")
    evidence = review.get("evidence", [])
    if not evidence or any(not Path(p).is_file() for p in evidence):
        raise ValueError("Provide existing input/upstream review evidence files")
    for key in ("science_candidates", "injection_candidates", "injections", "injection_m01", "raw_baseline_dir"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(config[key])


def verify_baseline(result, baseline_dir):
    checks = {}
    for pop in ("science", "injection"):
        current = result[pop]
        original = pd.read_csv(Path(baseline_dir) / f"{pop}_frozen_clustered.csv")
        if len(original) != len(current):
            raise RuntimeError(f"Raw baseline row-count mismatch: {pop}")
        for col in ("start_freq_MHz", "end_freq_MHz", "drift_rate_Hz_per_sec", VOTE, CONT):
            if not np.allclose(current[col].to_numpy(float), original[col].to_numpy(float), rtol=0,
                               atol=1e-12 if "freq" in col else 1e-14, equal_nan=True):
                raise RuntimeError(f"Raw baseline coordinate/score mismatch: {pop}/{col}")
        for col in ("frozen_hierarchy_retained", "point_score_retained"):
            if not np.array_equal(current[col].to_numpy(bool), original[col].to_numpy(bool)):
                raise RuntimeError(f"Raw baseline retention membership mismatch: {pop}/{col}")
        checks[pop] = {"rows": len(current), "exact_retention_membership": True}
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--stage", choices=("stats", "score", "fit", "all"), default="all")
    p.add_argument("--max-stat-samples", type=int, default=1_000_000)
    p.add_argument("--stats-freq-chunk", type=int, default=100_000)
    p.add_argument("--coarse-mhz", type=float, default=1.0)
    p.add_argument("--min-cluster-size", type=int, default=50)
    p.add_argument("--min-samples", type=int, default=20)
    args = p.parse_args()
    if min(args.max_stat_samples, args.stats_freq_chunk) < 1 or args.coarse_mhz <= 0:
        p.error("Positive sampling/chunk parameters required")
    config = json.loads(args.config.read_text())
    validate_config(config)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scorer_path = PIPELINE / "add_beam_track_vote_score_fast.py"
    model_path = PIPELINE / "frozen_hdbscan_validation.py"
    recovery_path = PIPELINE / "audit_recovery.py"
    reference = load_module(scorer_path, "r2_historical_scorer")
    params = parameters(args)
    frames = {pop: pd.read_csv(config[pop + "_candidates"]) for pop in ("science", "injection")}
    for pop, frame in frames.items():
        required = {"start_freq_MHz", "end_freq_MHz", "drift_rate_Hz_per_sec", VOTE, CONT}
        if not required.issubset(frame.columns):
            raise ValueError(f"{pop} input must preserve the verified primary raw scores and coordinates")
    versions = {}
    for package in ("numpy", "pandas", "h5py", "hdbscan", "scikit-learn", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    run = {"epoch": config["epoch"], "configuration": config, "parameters": params,
           "input_review_evidence": [identity(path, True) for path in config["input_review"]["evidence"]],
           "input_tables": {k: identity(config[k], True) for k in ("science_candidates", "injection_candidates", "injections")},
           "implementation": {p.name: identity(p, True) for p in (Path(__file__), scorer_path, model_path, recovery_path)},
           "versions": versions, "python": sys.version,
           "large_HDF5_identity_scope": "path, size, mtime_ns, header and selected statistic sample checksum; not a whole-file byte hash"}
    with open_readers(config, reference) as readers:
        run["hdf5_inputs"] = {name: {**identity(reader.path), **reader_metadata(reader)} for name, reader in readers.items()}
        key = out / "run_identity.json"
        if key.exists() and json.loads(key.read_text()) != run:
            raise RuntimeError("Existing output belongs to different inputs/code/parameters; choose a new output directory")
        write_json(key, run)
        try:
            stats_file = out / "science_intensity_statistics.json"
            if args.stage in ("stats", "all") and not stats_file.exists():
                stats = estimate_statistics(readers, params["statistics_frequency_ranges_mhz"], args.max_stat_samples, args.stats_freq_chunk)
                stats["run_identity_sha256"] = digest_file(key)
                write_json(stats_file, stats)
            if args.stage in ("score", "all"):
                stats = json.loads(stats_file.read_text())
                if stats["run_identity_sha256"] != digest_file(key):
                    raise RuntimeError("Frozen intensity statistics do not match this run")
                score_populations(readers, frames, stats, reference, out, args.coarse_mhz)
        except Exception as exc:
            write_json(out / "failure.json", {"stage": args.stage, "error": repr(exc), "time_unix": time.time()})
            raise
    if args.stage in ("fit", "all"):
        if versions["hdbscan"] != "0.8.40":
            raise RuntimeError("HDBSCAN 0.8.40 is required for the adopted baseline replay")
        model_reference = load_module(model_path, "r2_frozen_model_reference")
        match_reference = load_module(recovery_path, "r2_recovery_reference")
        score_audit = json.loads((out / "scoring_audit.json").read_text())
        if score_audit["intensity_stats_sha256"] != digest_file(out / "science_intensity_statistics.json"):
            raise RuntimeError("Scored representation no longer matches frozen intensity statistics")
        for pop in ("science", "injection"):
            if score_audit["normalized_score_files"][pop]["sha256"] != digest_file(out / f"{pop}_normalized_scores.csv"):
                raise RuntimeError("Normalized score checksum changed after scoring")
        normalized = {pop: pd.read_csv(out / f"{pop}_normalized_scores.csv") for pop in ("science", "injection")}
        injections = pd.read_csv(config["injections"])
        tolerance = 0.01905417 if config["epoch"] == "20250509" else 0.03844263
        results, recovery_ids, counts = {}, {}, {}
        for name, tables in (("raw_frozen", frames), ("normalized_frozen", normalized)):
            branch = out / name
            sci, inj, audit = fit_and_predict(tables["science"], tables["injection"], branch,
                model_reference, args.min_cluster_size, args.min_samples)
            results[name] = {"science": sci, "injection": inj, "audit": audit}
            counts[name], recovery_ids[name] = recovery(inj, injections, tolerance, match_reference, branch)
            if name == "raw_frozen":
                checks = verify_baseline(results[name], config["raw_baseline_dir"])
                write_json(out / "raw_baseline_replay.json", checks)
        raw = recovery_ids["raw_frozen"]["adopted_hierarchy"]
        norm = recovery_ids["normalized_frozen"]["adopted_hierarchy"]
        changes = [{"injection_row": i, "change": label} for label, ids in
                   (("gained_after_normalization", norm - raw), ("lost_after_normalization", raw - norm)) for i in sorted(ids)]
        pd.DataFrame(changes, columns=["injection_row", "change"]).to_csv(out / "normalization_recovered_id_changes.csv", index=False)
        write_json(out / "comparison.json", {"epoch": config["epoch"], "status": "complete",
            "protocol": "science-fixed intensity statistics; separately fitted raw and normalized science models frozen for corresponding injected data",
            "raw_baseline_reproduced": True, "injection_strengths_used_as_threshold": False,
            "new_search_or_preselection": False, "new_injections": False,
            "frozen_intensity_statistics_sha256": digest_file(out / "science_intensity_statistics.json"),
            "recovery": counts, "normalized_only_recovered_ids": sorted(norm - raw),
            "raw_only_recovered_ids": sorted(raw - norm),
            "models": {name: value["audit"] for name, value in results.items()}})
    write_json(out / "progress.json", {"stage": args.stage, "status": "complete", "updated_unix": time.time()})


if __name__ == "__main__":
    main()
