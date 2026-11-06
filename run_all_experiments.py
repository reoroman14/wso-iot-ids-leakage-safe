# run_all_experiments.py
# Runs the whole pipeline from the project folder, on any machine.
#
#   python run_all_experiments.py              -> prepare split (if missing) + all experiments + tables
#   python run_all_experiments.py --from-csv   -> also merge the raw CSVs into parquet first
#   python run_all_experiments.py --ablation   -> also run the WSO sensitivity sweep (slow)
#
# experiment_runner.py already runs baseline, AutoML and WSO-FS for every seed in config.SEEDS,
# so the three scripts are not run separately here.

import os
import sys
import time
import subprocess

from config import ROOT, ATTACK_PATH, BENIGN_PATH, TRAIN_DIR, TEST_DIR


def run(script):
    print(f"\n>>> Running {script} ...")
    code = subprocess.call([sys.executable, script], cwd=str(ROOT))
    if code != 0:
        print(f"[ERROR] {script} failed with exit code {code}. Stopping.")
        sys.exit(code)
    print(f"<<< Completed {script}\n" + "-" * 60)


def has_parquet(folder):
    return os.path.isdir(folder) and any(f.endswith(".parquet") for _, _, fs in os.walk(folder) for f in fs)


def main():
    t0 = time.time()
    args = sys.argv[1:]

    if "--from-csv" in args or not (os.path.exists(ATTACK_PATH) and os.path.exists(BENIGN_PATH)):
        run("merge_csv_to_parquet.py")

    if not (has_parquet(TRAIN_DIR) and has_parquet(TEST_DIR)):
        run("prepare_iot_diad_train_test_stream.py")
    else:
        print("[INFO] Train/test split already exists in outputs/, skipping preparation.")

    run("experiment_runner.py")
    run("generate_baseline_results_tex.py")
    run("generate_efficiency_comparison_tex.py")
    run("ablation_wso.py")

    if "--ablation" in args:
        run("ablation_sweep.py")

    print(f"\nAll done in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
