# NeuroClick: Early Purchase Ranking from Ordered Neuromarketing Visits

Code, frozen results and figure sources for the ICAIN 2026 paper (paper 456):

> Priyadharshini D, Shridevi S. *NeuroClick: Early Purchase Ranking from Ordered Neuromarketing Visits.* ICAIN 2026.

NeuroClick treats each participant-product pair in the NeuMa dataset as a time-ordered sequence of
gaze-defined product visits and estimates a nondecreasing purchase risk after each completed visit,
using only information available up to that visit. Evaluation is leave-one-participant-out with
participant-disjoint inner validation, cross-fitted product propensity, and participant-paired
Wilcoxon tests with Holm correction.

The NeuMa data are **not** redistributed here. Obtain them from the original publication
(Georgiadis et al., *Scientific Data* 10, 508, 2023, https://doi.org/10.1038/s41597-023-02392-9).
The pipeline reads the raw XDF recordings (`DataSource/S01.xdf` ... `S44.xdf`), not the
dataset's supplied preprocessed arrays.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | The analysis pipeline, numbered in execution order (see below). |
| `scripts/legacy/` | The earlier, non-causal cache builder, kept for reference only. Not used for any reported result. |
| `src/model/data/channel_harmonizer.py` | 19-channel EEG montage reconstruction imported by the cache builder. |
| `results/` | Frozen outputs that every number in the paper is computed from (see "Results" below). |
| `figures/manuscript_v1/` | Original manuscript figures (PDF + PNG) with their source manifest and validation record. |
| `figures/scripts/` | Figure style module and the regeneration scripts for Figs. 2 and 7. |
| `camera_ready/` | Scripts used to build the camera-ready Word/PDF (figure regeneration, XML edits, PDF export) and the 600 dpi figure PNGs. |

## Pipeline

All scripts take `--root` (the NeuMa workspace) and `--output` arguments; run `python <script> --help`.
The paths hard-coded as defaults (`/mnt/Neuma_Model/...`) are those of the original A100 workstation.

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_audit_buy_nobuy_events.py` | Frozen audit manifest of eligible participant-product instances and their cutoffs (Buy: click minus 500 ms; NoBuy: end of the last browsing interval on the page). `min_fixation_ms = 100`, `merge_gap_ms = 75`, `pre_click_margin_ms = 500`. |
| 2 | `02_build_preclick_sequence_cache_causal_v1.py` | Time-respecting visit-sequence cache from the raw XDF files: forward-only EEG filtering (causal common-average reference, 1-45 Hz band-pass, 50 Hz notch), 19-channel montage, past-only eye-tracking descriptors, prefix-invariance and cache-boundary self-tests. |
| 3 | `03_run_classical_losocv_benchmarks.py` | Builds the per-visit descriptor cache used by every model (`eeg_visit_features`: log relative band power for delta 1-4, theta, alpha, beta and gamma 30-45 Hz plus Hjorth activity, mobility and complexity on 19 channels = 152 values; `et_visit_features`: 69 ROI-relative gaze, pupil, velocity and disparity summaries), then runs the leave-one-participant-out heuristics, logistic baselines (including `logreg_dwell_propensity`, the information-matched baseline) and CatBoost, with inner-fold MCC thresholds. |
| 4 | `04_run_neuroclick_hazard.py` | NeuroClick (GRU + monotonic cumulative risk) under the same outer loop, for the behaviour, behaviour+ET, behaviour+EEG, behaviour+EEG+ET and no-propensity conditions. |
| 5 | `05_compare_pilot5_matched.py` | Row-matched comparison of the pilot NeuroClick run with the classical behaviour baseline. |
| 6 | `06_generate_manuscript_figures.py` | Figures 1 and 3-6 from the saved predictions. |
| 7 | `07_ensemble_neuroclick_baseline.py` | Average (logit space) and stacked ensembles of NeuroClick and the dwell+propensity baseline, with participant-paired tests. |
| 8 | `08_generate_ensemble_figure.py` | Figure 7. |
| - | `neuroclick_statistical_tests.py` | Primary first1 family: paired two-sided Wilcoxon, Student-t 95% intervals, Holm correction. |
| - | `neuroclick_crc_extension_tests.py` | Camera-ready extension tests (first2/first3 comparisons, ET ablation tests, threshold-matched MCC per participant). Prints the paired tests to the console. |
| - | `neuroclick_crc_mcc_16tests.py` | The 16-test Holm family for threshold-matched MCC reported in Sect. 4.5 (both strategies, both comparators, four horizons). |
| - | `neuroclick_crc_truncation_counts.py` | Sect. 3.2 exclusion and truncation counts from the audit manifest. |
| - | `neuroclick_crc_matched_baseline.py` | Sect. 3.5 sensitivity check: logistic baselines without the two validity fractions that behaviour-only NeuroClick never receives. |

## Results

`results/` holds the exact outputs the paper reports, including per-participant, per-instance
out-of-fold predictions so that every statistic can be recomputed without re-training:

| Folder | Produced by | Key files |
|---|---|---|
| `results/classical_losocv_final_run_03/` | script 03 | `losocv_predictions.csv`, `fold_metrics.csv`, `summary_metrics.csv`, `run_config.json` |
| `results/neuroclick_losocv_final_run_04/` | script 04 | `losocv_predictions.csv`, `fold_metrics.csv`, `summary_metrics.csv`, `run_config.json` |
| `results/neuroclick_primary_v1/` | statistical tests | `per_subject_metrics.csv`, `primary_pairwise_tests.csv`, `first1_model_intervals.csv`, `statistics_report.md` |
| `results/ensemble_v1/` | script 07 | `ensemble_predictions.csv`, `ensemble_per_subject_metrics.csv`, `ensemble_pairwise_tests.csv` |
| `results/catboost_first1_causal_v1/` | script 03 (earlier version, see below), model `catboost_fusion` | `summary_metrics.csv` (the CatBoost row of Table 2), `fold_metrics.csv`, `losocv_predictions.csv`, `run_config.json` |
| `results/crc_extension/` | extension tests | `crc_extension_tests_output.txt` (console output: Table 5 rows for first2/first3 vs dwell+propensity and ET vs behaviour), `crc_ensemble_mcc_threshold_matched.csv` (per-participant threshold-matched MCC), `crc_mcc_16_tests.csv` (the 16-test Holm family quoted in Sect. 4.5, from `scripts/neuroclick_crc_mcc_16tests.py`), `matched_baseline_no_validity.csv` (Sect. 3.5 sensitivity check: both logistic baselines re-evaluated without the two validity fractions, from `scripts/neuroclick_crc_matched_baseline.py`; it needs the visit-feature cache that script 03 writes, SHA-256 `41c486e9...`, and with scikit-learn 1.8 it reproduces the published 10-value baselines to within 0.002 pooled PR AUC) |

| `results/audit_full_42_v2/` | script 01 | `subject_product_audit.csv`, `subject_summary.csv`, `audit_summary.json`, `unmatched_clicks.csv`, `FEASIBILITY_DECISION.txt` |
| `results/cache_preclick_full_42_causal_v1/` | script 02 | `cache_summary.json` (cache counts, parameters and input hashes; the per-participant `.npz` caches are not included) |

Provenance chain: every stage records the SHA-256 of the script and inputs that produced it, and
the files in this repository hash to exactly those values.

| Recorded in | Item | SHA-256 (prefix) |
|---|---|---|
| `cache_summary.json` | `scripts/01_audit_buy_nobuy_events.py` | `35a00364` |
| `cache_summary.json` | `results/audit_full_42_v2/subject_product_audit.csv` | `038e19cb` |
| `cache_summary.json` | `scripts/02_build_preclick_sequence_cache_causal_v1.py` | `66942d9b` |
| `cache_summary.json` | `src/model/data/channel_harmonizer.py` | `824e9fb4` |
| both `run_config.json` | `results/cache_preclick_full_42_causal_v1/cache_summary.json` | `47c6f03b` |
| `classical .../run_config.json` | `scripts/03_run_classical_losocv_benchmarks.py` | `584c4dea` |
| `neuroclick .../run_config.json` | `scripts/04_run_neuroclick_hazard.py` | `d527acfa` |
| `catboost .../run_config.json` | `scripts/legacy/03_run_classical_losocv_benchmarks_catboost_run.py` | `546b37b5` |

The CatBoost baseline was run separately (`--models catboost_fusion`, GPU) with an earlier revision of
script 03 on the same causal caches; that revision is kept under `scripts/legacy/` so its hash can be checked.
Every value in Tables 2 to 6 and in Sect. 4.5 was re-derived from these files when the repository was assembled.

Predictions are indexed by held-out participant, horizon and audit instance; participant identifiers
are the NeuMa subject codes, and labels are the stated purchase selections from the dataset.

## Notes on the evaluation (as stated in the paper)

- The headline PR AUC ensemble results use the parameter-free logit average, which adds no fitting
  step beyond the two leave-one-participant-out components.
- The stacked ensemble and the threshold-matched MCC analysis are refit per held-out participant on
  the other participants' out-of-fold predictions taken from the single outer leave-one-participant-out
  run. They see no held-out label directly, but are not re-nested within each outer training partition
  (Sect. 4.5 and Limitations of the paper).
- All reported values use one training seed per fold (`seed = 42`).

## Environment

Python 3.13.12 with the package versions in `requirements.txt` (NumPy 2.4.6, pandas 2.3.3, SciPy 1.18.0,
scikit-learn 1.9.0, PyTorch 2.13.0 with CUDA 13.0; the cache build recorded pandas 3.0.3). NeuroClick was trained on an NVIDIA A100 80 GB
provided through an NVIDIA Academic Grant Program award. The camera-ready build scripts under
`camera_ready/` additionally need `lxml`, `PyMuPDF`, `OpenCV` and Microsoft Word (COM export).

## Citation

If you use this code or these results, please cite the paper:

> Priyadharshini D and Shridevi S. NeuroClick: Early Purchase Ranking from Ordered Neuromarketing Visits.
> In: Proceedings of the International Conference on Artificial Intelligence and Neuroscience (ICAIN 2026).
> Springer, 2026.

**Authors**

| Author | Role | ORCID |
|---|---|---|
| Priyadharshini D | First author (conceptualization, methodology, software, data curation, analysis, visualization, writing) | https://orcid.org/0009-0004-2291-1814 |
| Shridevi S | Research guide and corresponding author (conceptualization, supervision, resources, funding acquisition, manuscript review) | https://orcid.org/0000-0002-0038-7212 |

Both authors: Centre for Neuroinformatics, Vellore Institute of Technology, Chennai, India.

BibTeX (names are braced so that BibTeX keeps "Priyadharshini D" and "Shridevi S" intact):

```bibtex
@inproceedings{priyadharshini2026neuroclick,
  title     = {NeuroClick: Early Purchase Ranking from Ordered Neuromarketing Visits},
  author    = {{Priyadharshini D} and {Shridevi S}},
  booktitle = {Proceedings of the International Conference on Artificial Intelligence and Neuroscience (ICAIN 2026)},
  publisher = {Springer},
  address   = {Cham},
  year      = {2026},
  note      = {Paper 456. Code: \url{https://github.com/priyadharshini-D29/NeuroClick}}
}
```

A `CITATION.cff` file is included, so GitHub's "Cite this repository" button gives the same reference.

## License

The code and derived result files in this repository are released under the MIT License (see `LICENSE`).
The NeuMa dataset itself is governed by its own terms and is not part of this release.

## Contact

Priyadharshini D, Centre for Neuroinformatics, Vellore Institute of Technology, Chennai
(priyadharshini.2024b@vitstudent.ac.in). Corresponding author: Shridevi S (shridevi.s@vit.ac.in).
