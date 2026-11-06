import os
import json
import time
import numpy as np
import pyarrow.dataset as ds

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# =========================
# Paths
# =========================
from config import TRAIN_DIR, TEST_DIR, OUT_DIR  # paths come from config.py



# =========================
# Config
# =========================
BATCH_SIZE = 200_000             # reduce to 100_000 if RAM is tight
CLIP_VALUE = 1e9                 # clip extreme values

# Training balance:
# attack_to_benign = 1.0 => 1:1
ATTACK_TO_BENIGN = 1.0

# For AUC/PR-AUC on test: reservoir sample size (scores+labels)
AUC_SAMPLE = 1_000_000           # increase if you have RAM, decrease if needed


def sanitize_X(X: np.ndarray) -> np.ndarray:
    """
    Make X safe:
    - float64
    - inf -> nan
    - clip
    - nan -> 0
    """
    X = X.astype(np.float64, copy=False)
    if not X.flags.writeable:
        # pandas 3 / copy-on-write returns read-only arrays, so take a writable copy
        X = X.copy()
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.nan
    np.clip(X, -CLIP_VALUE, CLIP_VALUE, out=X)
    nan = np.isnan(X)
    if nan.any():
        X[nan] = 0.0
    return X


def dataset_feature_cols(dataset: ds.Dataset):
    cols = list(dataset.schema.names)
    # Expect: 79 features + y
    feature_cols = [c for c in cols if c != "y"]
    # safety
    if "Label" in feature_cols:
        feature_cols.remove("Label")
    return feature_cols


def iter_batches(dataset_dir: str, feature_cols, batch_size=BATCH_SIZE):
    dataset = ds.dataset(dataset_dir, format="parquet")
    scanner = dataset.scanner(columns=feature_cols + ["y"], batch_size=batch_size)
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        y = df["y"].to_numpy(dtype=np.int8, copy=False)
        X = df.drop(columns=["y"]).to_numpy(copy=False)
        X = sanitize_X(X)
        yield X, y


def count_class0_train() -> int:
    """
    Fast-ish count of benign rows (y=0) by scanning only 'y'.
    """
    dataset = ds.dataset(TRAIN_DIR, format="parquet")
    scanner = dataset.scanner(columns=["y"], batch_size=BATCH_SIZE)
    n0 = 0
    for batch in scanner.to_batches():
        y = batch.to_pandas()["y"].to_numpy(dtype=np.int8, copy=False)
        n0 += int((y == 0).sum())
    return n0


def build_balanced_train(feature_cols, rng):
    """
    Build balanced train arrays:
    - include all benign (y=0)
    - reservoir-sample attacks (y=1) to k = ATTACK_TO_BENIGN * n0
    Returns: X_train, y_train (in memory, manageable)
    """
    n0 = count_class0_train()
    k = int(ATTACK_TO_BENIGN * n0)

    print(f"Benign in TRAIN (y=0): {n0:,}")
    print(f"Target sampled attacks (y=1): {k:,} (ratio {ATTACK_TO_BENIGN}:1)")

    # Collect all benign in memory
    X0_list = []
    y0_list = []

    # Reservoir for attacks
    X1_res = None
    y1_res = None
    seen_attacks = 0

    for X, y in iter_batches(TRAIN_DIR, feature_cols):
        # benign rows
        m0 = (y == 0)
        if m0.any():
            X0_list.append(X[m0])
            y0_list.append(y[m0])

        # attack rows for reservoir
        m1 = (y == 1)
        if m1.any():
            Xa = X[m1]
            ya = y[m1]
            for i in range(len(ya)):
                seen_attacks += 1
                if X1_res is None:
                    # initialize reservoir arrays
                    X1_res = np.empty((k, Xa.shape[1]), dtype=np.float64)
                    y1_res = np.ones((k,), dtype=np.int8)
                    fill = 0

                # Fill first k
                if seen_attacks <= k:
                    X1_res[seen_attacks - 1] = Xa[i]
                else:
                    # Reservoir replace with probability k/seen_attacks
                    j = rng.integers(0, seen_attacks)
                    if j < k:
                        X1_res[j] = Xa[i]

    X0 = np.vstack(X0_list) if X0_list else np.empty((0, len(feature_cols)), dtype=np.float64)
    y0 = np.concatenate(y0_list) if y0_list else np.empty((0,), dtype=np.int8)

    if X1_res is None:
        raise RuntimeError("No attack samples found in TRAIN. Check your train_dataset creation.")

    # Merge + shuffle
    X_train = np.vstack([X0, X1_res])
    y_train = np.concatenate([y0, y1_res])

    idx = rng.permutation(len(y_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    print(f"Balanced TRAIN built: X={X_train.shape}, y0={(y_train==0).sum():,}, y1={(y_train==1).sum():,}")
    return X_train, y_train


def reservoir_auc_sample_update(y_true_buf, y_score_buf, seen, y_true_new, y_score_new, k, rng):
    """
    Reservoir sample for AUC arrays (labels + scores).
    """
    # y_true_buf/y_score_buf are preallocated length k, or None if not yet init
    if y_true_buf is None:
        y_true_buf = np.empty((k,), dtype=np.int8)
        y_score_buf = np.empty((k,), dtype=np.float64)

    for i in range(len(y_true_new)):
        seen += 1
        if seen <= k:
            y_true_buf[seen - 1] = y_true_new[i]
            y_score_buf[seen - 1] = y_score_new[i]
        else:
            j = rng.integers(0, seen)
            if j < k:
                y_true_buf[j] = y_true_new[i]
                y_score_buf[j] = y_score_new[i]

    return y_true_buf, y_score_buf, seen


def main(seed=42):
    t0 = time.time()
    rng = np.random.default_rng(seed)

    train_ds = ds.dataset(TRAIN_DIR, format="parquet")
    feature_cols = dataset_feature_cols(train_ds)
    print("Num features:", len(feature_cols))

    # 1) Build balanced training set (in memory)
    X_train, y_train = build_balanced_train(feature_cols, rng)

    # 2) Scale (important for many features)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)

    # 3) Train robust baseline model
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        max_iter=200,
        random_state=seed,
    )

    print("Training HistGradientBoostingClassifier on balanced subset...")
    clf.fit(X_train_s, y_train)

    # 4) Evaluate on FULL test set (streaming confusion matrix)
    print("Evaluating on full TEST (streaming confusion matrix)...")

    # confusion matrix counters
    tn = fp = fn = tp = 0

    # Reservoir sample for AUC/PR-AUC
    y_auc_buf = None
    s_auc_buf = None
    seen_auc = 0

    seen_rows = 0
    for X, y in iter_batches(TEST_DIR, feature_cols):
        Xs = scaler.transform(X)
        # HGB supports predict_proba
        s = clf.predict_proba(Xs)[:, 1]

        # default threshold (we can tune later)
        yhat = (s >= 0.5).astype(np.int8)

        # update confusion counts
        # 0=benign, 1=attack
        tn += int(((y == 0) & (yhat == 0)).sum())
        fp += int(((y == 0) & (yhat == 1)).sum())
        fn += int(((y == 1) & (yhat == 0)).sum())
        tp += int(((y == 1) & (yhat == 1)).sum())

        # update AUC reservoir
        y_auc_buf, s_auc_buf, seen_auc = reservoir_auc_sample_update(
            y_auc_buf, s_auc_buf, seen_auc, y, s, AUC_SAMPLE, rng
        )

        seen_rows += len(y)
        if seen_rows % 2_000_000 < BATCH_SIZE:
            print(f"  test seen rows: {seen_rows:,}")

    cm = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    # Build a small report from CM
    # precision/recall/f1 for positive class (attack)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # AUC/PR-AUC on sample
    # Note: if seen_auc < AUC_SAMPLE, buffer is only partially filled
    used = min(seen_auc, AUC_SAMPLE)
    y_auc = y_auc_buf[:used]
    s_auc = s_auc_buf[:used]

    roc_auc = roc_auc_score(y_auc, s_auc) if len(np.unique(y_auc)) == 2 else float("nan")
    pr_auc = average_precision_score(y_auc, s_auc) if len(np.unique(y_auc)) == 2 else float("nan")

    results = {
        "model": "HistGradientBoostingClassifier",
        "seed": seed,
        "train_balanced": {
            "attack_to_benign": ATTACK_TO_BENIGN,
            "rows": int(len(y_train)),
            "y0": int((y_train == 0).sum()),
            "y1": int((y_train == 1).sum()),
        },
        "test_full": {
            "rows_streamed": int(seen_rows),
            "confusion_matrix": cm,
            "precision_attack": float(precision),
            "recall_attack": float(recall),
            "f1_attack": float(f1),
        },
        "test_auc_sample": {
            "sample_size": int(used),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },
        "num_features": int(len(feature_cols)),
        "clip_value": float(CLIP_VALUE),
        "batch_size": int(BATCH_SIZE),
        "runtime_sec": float(time.time() - t0),
        "notes": "AUC/PR-AUC computed on reservoir sample of test scores to avoid storing 8.2M scores in RAM.",
    }

    print("\n=== RESULTS (summary) ===")
    print(json.dumps(results, indent=2))

    # Save JSON with seed-specific filename
    out_json = os.path.join(OUT_DIR, f"baseline_results_seed_{seed}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_json}")

    return results


if __name__ == "__main__":
    main()


def train_and_evaluate_hgb(seed=42):
    """
    Wrapper so experiment_runner.py can call this module directly.
    Returns metrics from full test stream evaluation.
    """
    out_json = os.path.join(OUT_DIR, f"baseline_results_seed_{seed}.json")

    # Always run for this seed (no caching)
    print(f"[INFO] Running baseline HGB training for seed {seed}...")
    results = main(seed=seed)

    # Return dictionary compatible with experiment_runner.py
    # Use full test stream metrics from JSON
    return {
        "test_full_f1": results["test_full"]["f1_attack"],
        "test_full_precision": results["test_full"]["precision_attack"],
        "test_full_recall": results["test_full"]["recall_attack"],
        "test_auc_roc": results["test_auc_sample"]["roc_auc"],
        "test_auc_pr": results["test_auc_sample"]["pr_auc"],
        "features_selected": results.get("num_features"),
        "runtime_sec": results.get("runtime_sec"),
    }
