#!/usr/bin/env python3
"""Complete a scored R2 representation in the isolated historical HDBSCAN runtime.

This separates the read-only intensity/scoring stage from the model runtime.
An earlier R1 manifest described HDBSCAN 0.8.40, but the live scoring environment
now has 0.8.41. Scoring never imports HDBSCAN. This launcher requires 0.8.40,
binds every existing score/statistics/input artifact by hash, records a distinct
fit identity, and requires exact raw-baseline retention replay before reporting
the new normalized-model result. It does not rescore or change existing data.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import frozen_normalization as core

DRIFT_TOLERANCES = {"20250509": 0.0190541709, "20250710": 0.03844263,
                    "20250718": 0.0384426255}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.run_dir.resolve()
    run = json.loads((out / "run_identity.json").read_text())
    config = run["configuration"]
    core.validate_config(config)
    stats = json.loads((out / "science_intensity_statistics.json").read_text())
    if stats["run_identity_sha256"] != core.digest_file(out / "run_identity.json"):
        raise RuntimeError("Statistics are not bound to this scoring identity")
    scored = json.loads((out / "scoring_audit.json").read_text())
    if not scored["all_rows_scored"] or scored["invalid_score_rows"]:
        raise RuntimeError("Incomplete or invalid normalized scores")
    if scored["intensity_stats_sha256"] != core.digest_file(out / "science_intensity_statistics.json"):
        raise RuntimeError("Frozen statistics changed after scoring")
    versions = {pkg: importlib.metadata.version(pkg) for pkg in run["versions"]}
    if versions["hdbscan"] != "0.8.40":
        raise RuntimeError("Fit launcher requires the isolated HDBSCAN 0.8.40 runtime")
    for pkg, value in run["versions"].items():
        if pkg != "hdbscan" and versions[pkg] != value:
            raise RuntimeError(f"Unexpected runtime change in {pkg}: {value} -> {versions[pkg]}")
    for artifact in [*run["input_tables"].values(), *run["implementation"].values(), *run["input_review_evidence"]]:
        if core.digest_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"Scoring input or implementation changed: {artifact['path']}")
    raw, normalized = {}, {}
    for pop in ("science", "injection"):
        path = out / f"{pop}_normalized_scores.csv"
        if core.digest_file(path) != scored["normalized_score_files"][pop]["sha256"]:
            raise RuntimeError(f"Normalized {pop} score file changed")
        raw[pop] = pd.read_csv(config[pop + "_candidates"])
        normalized[pop] = pd.read_csv(path)
        if len(raw[pop]) != len(normalized[pop]):
            raise RuntimeError("Representation changed the candidate population size")
        columns = ["start_freq_MHz", "end_freq_MHz", "drift_rate_Hz_per_sec"]
        for column in columns:
            np.testing.assert_allclose(raw[pop][column].to_numpy(), normalized[pop][column].to_numpy(),
                rtol=0, atol=1e-12 if "freq" in column else 1e-14)
    model = core.load_module(core.PIPELINE / "frozen_hdbscan_validation.py", "r2_final_frozen_reference")
    match = core.load_module(core.PIPELINE / "audit_recovery.py", "r2_final_matching_reference")
    fit_identity = {"scoring_run_identity_sha256": core.digest_file(out / "run_identity.json"),
        "scoring_audit_sha256": core.digest_file(out / "scoring_audit.json"),
        "frozen_statistics_sha256": core.digest_file(out / "science_intensity_statistics.json"),
        "fit_launcher": core.identity(__file__, True), "fit_versions": versions,
        "hdbscan_module_path": model.hdbscan.__file__, "python":sys.version,
        "model_parameters": {key:run["parameters"][key] for key in ("min_cluster_size", "min_samples")},
        "runtime_note":"HDBSCAN was not imported or used in intensity normalization/scoring; the model fit uses isolated 0.8.40. All other package versions match scoring.",
        "drift_tolerance_hz_s": DRIFT_TOLERANCES[config["epoch"]],
        "frequency_tolerance_hz":22.5}
    idpath = out / "fitting_identity.json"
    if idpath.exists() and json.loads(idpath.read_text()) != fit_identity:
        raise RuntimeError("Different fitting identity already present")
    core.write_json(idpath, fit_identity)
    injections = pd.read_csv(config["injections"])
    results, counts, recovery_ids = {}, {}, {}
    for name, tables in (("raw_frozen", raw), ("normalized_frozen", normalized)):
        branch = out / name
        sci, inj, audit = core.fit_and_predict(tables["science"], tables["injection"], branch, model,
            run["parameters"]["min_cluster_size"], run["parameters"]["min_samples"])
        results[name] = {"science":sci, "injection":inj, "audit":audit}
        counts[name], recovery_ids[name] = core.recovery(inj, injections,
            DRIFT_TOLERANCES[config["epoch"]], match, branch)
        if name == "raw_frozen":
            checks = core.verify_baseline(results[name], config["raw_baseline_dir"])
            core.write_json(out / "raw_baseline_replay.json", checks)
        print(name, audit["science_frozen_hierarchy_retained"], counts[name], flush=True)
    old, new = (recovery_ids[name]["adopted_hierarchy"] for name in ("raw_frozen", "normalized_frozen"))
    changes = [{"injection_row":i,"change":label} for label, ids in
        (("gained_after_normalization",new-old),("lost_after_normalization",old-new)) for i in sorted(ids)]
    pd.DataFrame(changes, columns=["injection_row","change"]).to_csv(out / "normalization_recovered_id_changes.csv", index=False)
    core.write_json(out / "comparison.json", {"epoch":config["epoch"], "status":"complete",
        "protocol":"Science-fixed intensity statistics; separately fitted raw and normalized science models frozen for corresponding injected data",
        "raw_baseline_reproduced":True, "injection_strengths_used_as_threshold":False,
        "new_search_or_preselection":False, "new_injections":False,
        "fitting_identity_sha256":core.digest_file(idpath),
        "frozen_intensity_statistics_sha256":core.digest_file(out / "science_intensity_statistics.json"),
        "recovery":counts, "normalized_only_recovered_ids":sorted(new-old), "raw_only_recovered_ids":sorted(old-new),
        "models":{name:values["audit"] for name,values in results.items()}})
    core.write_json(out / "progress.json", {"stage":"fit", "status":"complete", "fit_runtime":"isolated HDBSCAN 0.8.40"})


if __name__ == "__main__":
    main()
