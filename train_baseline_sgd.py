import os
import json
import time
import numpy as np
import pyarrow.dataset as ds

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

# =========================
# Paths
# =========================
from config import TRAIN_DIR, TEST_DIR  # paths come from config.py


# From your split summary:
# TRAIN y counts: 0=278831, 1=18989212
N0_TRAIN = 278_831
N1_TRAIN = 18_989_212

# Streaming config
BATCH_SIZE = 200_000
PRINT_EVERY_ROWS = 2_000_000

# Numeric safety
CLIP_VALUE = 1e9  # clip extreme values to avoid overflow/instability


def get_feature_columns(dataset: ds.Dataset) -> list[str]:
    cols = list(dataset.schema.names)
    feature_cols = [c for c in cols if c != "y"]
    # Safety: drop any accidental non-feature cols
    if "Label" in feature_cols:
        feature_cols.remove("Label")
    return feature_cols


def sanitize_X(X: np.ndarray) -> np.ndarray:
    """
    Make X safe for scaler/model:
    - convert to float64 for stability
    - replace inf/-inf with NaN
    - clip extreme magnitudes
    - replace NaN with 0
    """
    X = X.astype(np.float64, copy=False)

    # Replace inf with NaN
    inf_mask = ~np.isfinite(X)
    if inf_mask.any():
        X[inf_mask] = np.nan

    # Clip extreme values (after inf->nan)
    np.clip(X, -CLIP_VALUE, CLIP_VALUE, out=X)

    # Replace NaN with 0
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = 0.0

    return X


def iter_batches(dataset_dir: str, feature_cols: list[str], batch_size: int = BATCH_SIZE):
    dataset = ds.dataset(dataset_dir, format="parquet")
    scanner = dataset.scanner(columns=feature_cols + ["y"], batch_size=batch_size)

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        y = df["y"].to_numpy(dtype=np.int8, copy=False)
        X = df.drop(columns=["y"]).to_numpy(copy=False)
        X = sanitize_X(X)
        yield X, y


def compute_class_weights(n0: int, n1: int) -> dict:
    n = n0 + n1
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return {0: float(w0), 1: float(w1)}


def main():
    t0 = time.time()

    train_ds = ds.dataset(TRAIN_DIR, format="parquet")
    feature_cols = get_feature_columns(train_ds)
    print("Num features:", len(feature_cols))

    # 1) Fit scaler incrementally
    scaler = StandardScaler(with_mean=False)

    print("Fitting scaler (streaming)...")
    seen = 0
    for X, y in iter_batches(TRAIN_DIR, feature_cols):
        scaler.partial_fit(X)
        seen += len(y)
        if seen % PRINT_EVERY_ROWS < BATCH_SIZE:
            print(f"  scaler seen rows: {seen:,}")
    print(f"Scaler fitted on {seen:,} rows.")

    # 2) Train SGD incrementally
    class_weight = compute_class_weights(N0_TRAIN, N1_TRAIN)
    print("Class weights:", class_weight)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        random_state=42,
        class_weight=class_weight,
    )

    classes = np.array([0, 1], dtype=np.int8)

    print("Training SGDClassifier (streaming)...")
    first = True
    seen = 0
    for X, y in iter_batches(TRAIN_DIR, feature_cols):
        Xs = scaler.transform(X)
        if first:
            clf.partial_fit(Xs, y, classes=classes)
            first = False
        else:
            clf.partial_fit(Xs, y)

        seen += len(y)
        if seen % PRINT_EVERY_ROWS < BATCH_SIZE:
            print(f"  train seen rows: {seen:,}")

    # 3) Evaluate streaming
    print("Evaluating (streaming)...")
    y_true_all = []
    y_score_all = []

    seen = 0
    for X, y in iter_batches(TEST_DIR, feature_cols):
        Xs = scaler.transform(X)
        scores = clf.predict_proba(Xs)[:, 1]
        y_true_all.append(y)
        y_score_all.append(scores.astype(np.float64, copy=False))

        seen += len(y)
        if seen % PRINT_EVERY_ROWS < BATCH_SIZE:
            print(f"  test seen rows: {seen:,}")

    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)

    # default threshold
    y_pred = (y_score >= 0.5).astype(np.int8)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    results = {
        "rows_train_expected": int(N0_TRAIN + N1_TRAIN),
        "rows_test": int(len(y_true)),
        "features": int(len(feature_cols)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "clip_value": float(CLIP_VALUE),
        "scaler": "StandardScaler(with_mean=False)",
        "model": "SGDClassifier(log_loss, class_weight=dict)",
        "batch_size": int(BATCH_SIZE),
        "runtime_sec": float(time.time() - t0),
    }

    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    out_path = os.path.join(os.path.dirname(__file__), "baseline_sgd_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
