# Leakage-safe IoT intrusion detection with White Shark Optimizer feature selection

Code for the paper:

> A. A. Aburomman, M. Alakhras, M. B. I. Reaz, "Leakage-safe IoT intrusion detection with White Shark Optimizer-based feature selection", Array 30 (2026) 100903. https://doi.org/10.1016/j.array.2026.100903

The pipeline evaluates three models on the CIC IoT-DIAD 2024 flow data under a chronological train/test split:
a baseline HistGradientBoosting (HGB) classifier, an Optuna-tuned HGB (AutoML), and HGB with White Shark Optimizer wrapper feature selection (WSO-FS).
Results are reported as mean and standard deviation over five seeds (42 to 46).

## 1. Install

Python 3.10 or newer.

```
pip install -r requirements.txt
```

## 2. Get the data

The dataset is not included in this repository. Download the flow-based files of CIC IoT-DIAD 2024 from the Canadian Institute for Cybersecurity:

- Dataset page: https://unb.ca/cic/datasets/iot-diad-2024.html
- Use the folder `AD_Flow-based-features` (CICFlowMeter flow features, CSV). The download page may ask for a short registration form.

Put all the flow CSV files under `data/raw/` (sub-folders are fine):

```
data/
  raw/
    ...your downloaded .csv files...
```

To keep the dataset somewhere else, set the environment variable `IOT_DIAD_DATA_DIR` to that folder (it must contain a `raw` sub-folder).

## 3. Run everything

```
python run_all_experiments.py
```

This runs, in order:

1. `merge_csv_to_parquet.py` merges the CSVs and assigns labels. Output: `data/attacks.parquet` and `data/Benign.parquet`.
2. `prepare_iot_diad_train_test_stream.py` removes identifier columns, assigns `y` (1 = attack, 0 = benign) and does the chronological split, first 70 percent of rows to train and last 30 percent to test, separately for the attack file and the benign file. Output: `outputs/train_dataset/` and `outputs/test_dataset/`.
3. `experiment_runner.py` runs baseline, AutoML and WSO-FS for every seed in `config.SEEDS`, and writes per-seed JSON files to `outputs/`, figures to `paper_outputs/figures/` and tables to `paper_outputs/tables/`.
4. `generate_baseline_results_tex.py`, `generate_efficiency_comparison_tex.py` and `ablation_wso.py` build the LaTeX tables from the seed JSON files.

Optional flags:

```
python run_all_experiments.py --from-csv    # force re-merge of the CSVs
python run_all_experiments.py --ablation    # also run the WSO alpha x population sweep (slow)
```

Each script can also be run alone, for example `python wso_feature_selection.py`.
In the author's saved runs, each model took roughly one to a few minutes per seed on a multi-core CPU, plus data preparation time. The dataset has roughly 27 million flows. If memory is tight, lower `BATCH_SIZE` in `train_baseline_hgb_balanced.py`.

### Checking your run against the paper

`reference_results/` holds the per-seed JSON files that the paper's tables were computed from. After a run:

```
python compare_with_reference.py
```

prints F1 and feature counts next to the reference values. In the reference files the WSO-FS run of seed 42 used alpha 0.05 and 10 iterations, while seeds 43 to 46 used alpha 0.02 and 20 iterations, so seed 42 only matches if you run it with those settings.

## 4. Files

| File | Purpose |
| --- | --- |
| `config.py` | All paths, the drop-column list, split ratio, seeds, ablation grid |
| `merge_csv_to_parquet.py` | CSV merging and label assignment |
| `prepare_iot_diad_train_test_stream.py` | Identifier removal, label encoding, chronological split |
| `train_baseline_hgb_balanced.py` | Baseline HGB, balanced by undersampling the training data only |
| `automl_optuna_hgb.py` | Optuna hyperparameter search (30 trials) on training data only |
| `wso_feature_selection.py` | WSO wrapper feature selection and final model |
| `experiment_runner.py` | Runs all models over all seeds, builds figures and tables |
| `ablation_wso.py`, `ablation_sweep.py` | WSO result table and the alpha x population sweep |
| `measure_throughput.py` | Inference throughput timing |
| `compare_with_reference.py` | Compares your results with `reference_results/` |
| `train_baseline_sgd.py` | Extra SGD baseline, not part of `run_all_experiments.py` |

## 5. Settings used in the paper

- Seeds: 42, 43, 44, 45, 46 (`config.SEEDS`).
- Class balancing: random undersampling to 1:1 on the training part only, after the split. The test stream stays imbalanced.
- WSO: population 16, 20 iterations, feature penalty alpha = 0.02, position bounds [-6, 6], sigmoid binarisation at 0.5.
- WSO fitness: attack F1 on a proxy validation split (20,000 benign and 20,000 attack training flows, 80/20 split) minus alpha times (selected features / total features).
- Optuna: 30 trials on a balanced proxy of 50,000 flows per class, 80/20 validation split.
- Ablation sweep (`ablation_sweep.py`): alpha in {0.005, 0.01, 0.02, 0.05} x population in {8, 12, 16, 20}, seed 42, 10 WSO iterations per run.
- Final models are trained on the full balanced training set and evaluated on the whole test stream. ROC-AUC and PR-AUC use a 1,000,000-score reservoir sample of the test stream.

## 6. Known notes

- Row order matters. The chronological split uses the row order of `attacks.parquet` and `Benign.parquet` (no re-sorting by timestamp). `merge_csv_to_parquet.py` keeps files in natural sorted path order and rows in file order. A different file order gives a different train/test split, so exact numbers depend on it.
- Columns removed by the code are exactly `Flow ID`, `Src IP`, `Dst IP`, `Timestamp` (see `DROP_COLS` in `config.py`). `Src Port` and `Dst Port` are not removed and are among the 79 input features in the code that produced the reported results.
- `measure_throughput.py` times a fixed hand-written WSO feature list, not the subset of a particular seed, because only inference speed is measured.
- Exact figures can differ slightly between library versions and CPUs.

## 7. Citation and dataset credit

If you use this code, please cite the paper above.

The dataset is by M. Rabbani, J. Gui, F. Nejati, Z. Zhou, A. Kaniyamattam, M. Mirani, G. Piya, I. Opushnyev, R. Lu and A. A. Ghorbani, "Device Identification and Anomaly Detection in IoT Environments", IEEE Internet of Things Journal, December 2024. Please also cite it.

## Contact

Abdulla Amin Aburomman, a_aburomman@asu.edu.jo
