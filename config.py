# config.py
# Central place for every path and experiment setting.
# Nothing here depends on the machine: all paths are relative to this folder
# unless you override them with environment variables.

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Folder that holds the downloaded dataset.
#   data/raw/        -> the original CIC IoT-DIAD 2024 flow CSV files (any sub-folders)
#   data/            -> attacks.parquet and Benign.parquet (created by merge_csv_to_parquet.py)
# Override with:  set IOT_DIAD_DATA_DIR=E:\my_data   (Windows)
#                 export IOT_DIAD_DATA_DIR=/my_data  (Linux / macOS)
DATA_DIR = Path(os.environ.get("IOT_DIAD_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"

ATTACK_PATH = str(DATA_DIR / "attacks.parquet")
BENIGN_PATH = str(DATA_DIR / "Benign.parquet")

# Results and the chronological train/test split are written here.
OUT_DIR = str(ROOT / "outputs")
TRAIN_DIR = str(Path(OUT_DIR) / "train_dataset")
TEST_DIR = str(Path(OUT_DIR) / "test_dataset")

# Columns removed before any preprocessing or training.
# Exactly as used for the reported results. See README, section "Known notes".
DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]

# Chronological split: first 70 percent of rows -> train, last 30 percent -> test.
TRAIN_RATIO = 0.7

# Five seeds used in the paper.
SEEDS = [42, 43, 44, 45, 46]

# Grid used for the WSO sensitivity (ablation) study.
ABLATION_ALPHAS = [0.005, 0.01, 0.02, 0.05]
ABLATION_POPULATIONS = [8, 12, 16, 20]
ABLATION_ITERS = 10  # the original ablation used 10 iterations (main runs use 20)
ABLATION_SEED = 42
