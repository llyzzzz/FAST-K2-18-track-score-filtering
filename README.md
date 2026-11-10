# FAST K2-18 multibeam track-score filtering

Analysis code for *A Multibeam Track-Score Filtering Framework for FAST Narrowband Technosignature Searches toward K2-18*. The main workflow filters narrowband candidates using seven FAST beams, computes two scores along each candidate's frequency track, and fits a model on the uninjected science data for each epoch. The model is held fixed when evaluating injections from the same epoch.

## Setup

Use Python 3.10 to match the recorded normalization environment. The archived runs used Python 3.10.19 on Linux, with HDBSCAN 0.8.40 for model fitting. The requirements files record the Python package versions used for fitting and recovery. The full server environment also included system libraries and, for raw searches, BLISS. Install the HDF5 dependencies if you will compute track scores from intensity products or run the normalization comparison.

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-score.txt
# For HDF5 input and intensity-level scoring:
python -m pip install -r requirements-hdf5.txt
```

The search wrappers also require a working BLISS Python installation (`blissdedrift`, v0.0.1rc8 in the archived analysis records) and an appropriate compute device. Depending on the observation products, the HDF5 installation may require additional compression filters. Use a new output directory for each run.

## Frozen validation from existing scores

Use the science score table, injection-run score table and injection log for the same epoch. Keep the original table row order. The score tables contain `beam_track_vote_score` and `beam_track_continuity`. Recovery matching also reads `start_freq_MHz` and `drift_rate_Hz_per_sec` from the candidate table and `f_start_MHz` and `drift_rate_Hz_per_s` from the injection log. If the log contains `beam_id`, the matcher selects M01 entries.

The commands below illustrate the May 9 run. Replace the input paths with your local copies of the corresponding files.

```sh
python pipeline/frozen_hdbscan_validation.py \
  --science inputs/20250509/science_track_scores.csv \
  --injection inputs/20250509/injection_track_scores_one-bin.csv \
  --out-dir runs/20250509/frozen \
  --min-cluster-size 50 --min-samples 20

python pipeline/audit_recovery.py \
  --clustered runs/20250509/frozen/injection_frozen_clustered.csv \
  --injections inputs/20250509/injection_log.csv \
  --drift-tolerance 0.0190541709 \
  --retained-column frozen_hierarchy_retained \
  --output runs/20250509/injection_recovery.csv
```

The recovery frequency tolerance is 22.5 Hz. Use the following drift tolerances for the corresponding epochs.

| Epoch | Recovery drift tolerance (Hz/s) |
| --- | ---: |
| 2025-05-09 | 0.0190541709 |
| 2025-07-10 | 0.03844263 |
| 2025-07-18 | 0.0384426255 |

`audit_recovery_stages.py` provides stage-by-stage recovery matching. `extras/audit_parameter_sensitivity.py` examines May model and retention choices with the input scores held fixed.

## From search hits to track scores

`run_bliss_science.py` and `run_bliss_one_beam.py` provide the BLISS search interfaces. For the July 18 M14 input, the adopted catalogue was assembled from 32 virtual frequency slices with 1,024 overlap channels at each internal side. `make_h5_frequency_vds.py` and `merge_vds_hits.py` implement slicing and core-region selection. Start from the adopted M14 catalogue when using the archived search inputs.

Given the seven-beam hit table, M01 intensity file and the corresponding OFF reference product, the science branch can be scored as follows.

```sh
python pipeline/postprocess_bliss.py \
  --hits inputs/20250509/bliss_hits_1000_1500.csv \
  --m01 inputs/20250509/K2-18b_M01.h5 \
  --out-dir runs/20250509/preselection --drift-tolerance one-bin

python pipeline/add_beam_track_vote_score_fast.py \
  --csv runs/20250509/preselection/science_candidates_preselected.csv \
  --m01 inputs/20250509/K2-18b_M01.h5 \
  --upper inputs/20250509/OFF_UPPER.h5 \
  --out runs/20250509/science_track_scores.csv \
  --sigma-hz 15 --cutoff-sigma 3 --continuity-core-hz 15
```

The OFF reference is the six-beam mean plus twice the population standard deviation on the same time and frequency grid. For the injection branch, use the injected M01 intensity and its search hits with the original six science OFF hit tables and the corresponding OFF reference. The original injection-generation products and prepared OFF reference are separate inputs to this workflow.

## Frozen normalization comparison

The scripts under `pipeline/revision_r2/` implement the revised comparison of raw and normalized intensity representations. Statistics are estimated per beam from uninjected science intensity over 1050–1450 MHz. Injection M01 uses the corresponding science M01 scale. The raw and normalized science models are fit separately, then held fixed for their corresponding injection evaluations.

Copy `config/normalization.example.json` to a local configuration file, update its input paths, and complete the input verification required by the script. The verified raw baseline directory must contain `science_frozen_clustered.csv` and `injection_frozen_clustered.csv`. With the approved inputs, run these commands from the repository root using the same configuration and output directory. The final step uses HDBSCAN 0.8.40 and the epoch-specific recovery tolerances.

```sh
python pipeline/revision_r2/frozen_normalization.py \
  --config inputs/normalization.json --out-dir runs/20250509/normalization --stage stats
python pipeline/revision_r2/frozen_normalization.py \
  --config inputs/normalization.json --out-dir runs/20250509/normalization --stage score
python pipeline/revision_r2/finish_normalization_fits.py \
  --run-dir runs/20250509/normalization
```

The normalization script's scorer lookup points to the copy shipped under `pipeline/`. The scorer matches the recorded reference by SHA-256, and the lookup change does not alter the scoring or fitting rules.

## Data and reproduction scope

This repository contains analysis code. Researchers can contact the paper's corresponding authors to request the FAST observation products, injection catalogues and derived candidate tables, subject to the applicable data access policy.

The archived primary analysis retained 5,257 science candidates and recovered 2,815 of 3,000 injections. A documented run starting from archived scores in a Windows environment recovered the same number of injections but retained 5,256 science candidates and differed in some retained row identities. Treat these figures as separate results. Full observational reproduction requires the corresponding data products and software environment.
