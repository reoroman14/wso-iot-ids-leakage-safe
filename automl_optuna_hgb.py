import os
import json
import time
import math
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
import optuna

# === Paths and Config ===
from config import TRAIN_DIR, TEST_DIR, OUT_DIR  # paths come from config.py


os.makedirs(OUT_DIR, exist_ok=True)

# RESULT_JSON will be seed-specific

DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
TUNE_PER_CLASS = 50_000
VAL_RATIO = 0.2
FULL_TRAIN_ATTACK_TO_BENIGN = 1.0
AUC_SAMPLE_SIZE = 1_000_000


# === Utility Functions ===
def list_parquet_files(folder):
    files = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if f.lower().endswith(".parquet"):
                files.append(os.path.join(root, f))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No .parquet files found in: {folder}")
    return files


def detect_label_col(df: pd.DataFrame) -> str:
    """Try common label names and also handle string labels."""
    candidates = ["Label", "label", "y", "Y", "class", "Class"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No label column found. Columns are: {list(df.columns)[:30]} ...")


def to_binary_y(series: pd.Series) -> np.ndarray:
    """Convert label column to binary numeric array."""
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        s = s.map({"benign": 0, "attack": 1})
        y = pd.to_numeric(s, errors="coerce").fillna(0).astype(np.int8).to_numpy()
        return y
    y = pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int8).to_numpy()
    return y


def safe_numeric_Xy(df: pd.DataFrame):
    label_col = detect_label_col(df)
    y = to_binary_y(df[label_col])
    X = df.drop(columns=[label_col])

    for c in DROP_COLS:
        if c in X.columns:
            X = X.drop(columns=[c])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y, label_col


def stream_balanced_sample(parquet_files, per_class, seed=42):
    """Sample a balanced proxy dataset from multiple parquet files."""
    rng = np.random.default_rng(seed)
    pos_parts, neg_parts = [], []
    pos_count = 0
    neg_count = 0

    label_name_seen = None

    for f in parquet_files:
        df = pd.read_parquet(f)
        try:
            X, y, label_col = safe_numeric_Xy(df)
        except KeyError:
            continue

        if label_name_seen is None:
            label_name_seen = label_col
            print(f"[INFO] Detected label column: {label_name_seen}")

        df2 = X.copy()
        df2["__y__"] = y

        idx = np.arange(len(df2))
        rng.shuffle(idx)
        df2 = df2.iloc[idx]

        neg = df2[df2["__y__"] == 0].drop(columns=["__y__"])
        pos = df2[df2["__y__"] == 1].drop(columns=["__y__"])

        if neg_count < per_class and len(neg) > 0:
            take = min(per_class - neg_count, len(neg))
            neg_parts.append(neg.iloc[:take])
            neg_count += take

        if pos_count < per_class and len(pos) > 0:
            take = min(per_class - pos_count, len(pos))
            pos_parts.append(pos.iloc[:take])
            pos_count += take

        if pos_count >= per_class and neg_count >= per_class:
            break

    if pos_count < per_class or neg_count < per_class:
        raise RuntimeError(
            f"Could not sample enough data. Got benign={neg_count}, attack={pos_count}, needed={per_class} each."
        )

    X_neg = pd.concat(neg_parts, ignore_index=True)
    X_pos = pd.concat(pos_parts, ignore_index=True)
    y_neg = np.zeros(len(X_neg), dtype=np.int8)
    y_pos = np.ones(len(X_pos), dtype=np.int8)

    X_all = pd.concat([X_neg, X_pos], ignore_index=True)
    y_all = np.concatenate([y_neg, y_pos])

    perm = rng.permutation(len(X_all))
    X_all = X_all.iloc[perm].reset_index(drop=True)
    y_all = y_all[perm]

    return X_all, y_all


def train_val_split(X, y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    cut = int(math.floor(n * (1.0 - val_ratio)))
    tr_idx = idx[:cut]
    va_idx = idx[cut:]
    return X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx]


def build_full_balanced_train(train_files, attack_to_benign=1.0, seed=42):
    """Build full training set with a balanced ratio of attacks to benign."""
    rng = np.random.default_rng(seed)
    benign_parts, benign_count = [], 0
    label_name_seen = None

    for f in train_files:
        df = pd.read_parquet(f)
        X, y, label_col = safe_numeric_Xy(df)
        if label_name_seen is None:
            label_name_seen = label_col
            print(f"[INFO] Detected label column (full train): {label_name_seen}")
        b = X[y == 0]
        if len(b) > 0:
            benign_parts.append(b)
            benign_count += len(b)

    if benign_count == 0:
        raise RuntimeError("No benign samples found in TRAIN dataset.")

    target_attacks = int(benign_count * attack_to_benign)
    attack_parts, attack_count = [], 0

    for f in train_files:
        if attack_count >= target_attacks:
            break
        df = pd.read_parquet(f)
        X, y, _ = safe_numeric_Xy(df)
        a = X[y == 1]
        if len(a) == 0:
            continue
        remain = target_attacks - attack_count
        take = min(remain, len(a))
        idx = rng.choice(len(a), size=take, replace=False)
        attack_parts.append(a.iloc[idx])
        attack_count += take

    X_b = pd.concat(benign_parts, ignore_index=True)
    X_a = pd.concat(attack_parts, ignore_index=True)
    y_b = np.zeros(len(X_b), dtype=np.int8)
    y_a = np.ones(len(X_a), dtype=np.int8)

    X_all = pd.concat([X_b, X_a], ignore_index=True)
    y_all = np.concatenate([y_b, y_a])

    perm = rng.permutation(len(X_all))
    X_all = X_all.iloc[perm].reset_index(drop=True)
    y_all = y_all[perm]

    return X_all, y_all, benign_count, attack_count


def stream_test_metrics(model, test_files, auc_sample_size=1_000_000, seed=42):
    """Stream test data and compute metrics."""
    rng = np.random.default_rng(seed)
    tn = fp = fn = tp = 0

    scores_buf = np.empty(auc_sample_size, dtype=np.float32)
    y_buf = np.empty(auc_sample_size, dtype=np.int8)
    seen = filled = 0

    for f in test_files:
        df = pd.read_parquet(f)
        X, y, _ = safe_numeric_Xy(df)
        Xnp = X.astype(np.float32, copy=False).to_numpy()

        proba = model.predict_proba(Xnp)[:, 1]
        pred = (proba >= 0.5).astype(np.int8)

        tn += int(((pred == 0) & (y == 0)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        tp += int(((pred == 1) & (y == 1)).sum())

        for i in range(len(y)):
            s = float(proba[i])
            yi = int(y[i])
            seen += 1
            if filled < auc_sample_size:
                scores_buf[filled] = s
                y_buf[filled] = yi
                filled += 1
            else:
                j = rng.integers(0, seen)
                if j < auc_sample_size:
                    scores_buf[j] = s
                    y_buf[j] = yi

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    y_s = y_buf[:filled]
    s_s = scores_buf[:filled]
    roc_auc = float(roc_auc_score(y_s, s_s)) if len(np.unique(y_s)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y_s, s_s)) if len(np.unique(y_s)) > 1 else float("nan")

    return {
        "rows_test_streamed": int(tn + fp + fn + tp),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision_attack": precision,
        "recall_attack": recall,
        "f1_attack": f1,
        "roc_auc_sample": roc_auc,
        "pr_auc_sample": pr_auc,
        "auc_sample_size_used": filled,
    }


def make_objective(X_tr, y_tr, X_va, y_va, seed=42):
    """Objective function for Optuna hyperparameter tuning."""
    X_tr_np = X_tr.astype(np.float32).to_numpy()
    X_va_np = X_va.astype(np.float32).to_numpy()

    def objective(trial: optuna.Trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 500, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1e-1, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 400),
            "random_state": seed,
        }
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X_tr_np, y_tr)
        proba = clf.predict_proba(X_va_np)[:, 1]
        pred = (proba >= 0.5).astype(np.int8)
        return f1_score(y_va, pred, pos_label=1)

    return objective


def main(seed=42):
    """Main training and tuning function."""
    t0 = time.time()
    train_files = list_parquet_files(TRAIN_DIR)
    test_files = list_parquet_files(TEST_DIR)

    print("Building proxy tuning set (balanced)...")
    X_tune, y_tune = stream_balanced_sample(train_files, per_class=TUNE_PER_CLASS, seed=seed)
    X_tr, y_tr, X_va, y_va = train_val_split(X_tune, y_tune, val_ratio=VAL_RATIO, seed=seed)
    print(f"Proxy tuning set: X={X_tune.shape}, train={len(X_tr)}, val={len(X_va)}")

    print("Running Optuna for HGB tuning...")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(X_tr, y_tr, X_va, y_va, seed=seed), n_trials=30)

    best_params = study.best_trial.params
    best_f1_val = float(study.best_value)

    print("Best params:", best_params)
    print("Best val F1 (attack):", best_f1_val)

    print("Building full balanced training set...")
    X_full, y_full, benign_count, attack_count = build_full_balanced_train(
        train_files, attack_to_benign=FULL_TRAIN_ATTACK_TO_BENIGN, seed=seed
    )
    print(f"Full balanced train: X={X_full.shape}, benign={benign_count}, attack_sampled={attack_count}")

    final_model = HistGradientBoostingClassifier(**best_params, random_state=seed)
    print("Training final AutoML-HGB...")
    final_model.fit(X_full.astype(np.float32).to_numpy(), y_full)

    print("Evaluating on full TEST (streaming)...")
    test_metrics = stream_test_metrics(final_model, test_files, auc_sample_size=AUC_SAMPLE_SIZE, seed=seed)

    out = {
        "model": "AutoML-HistGradientBoostingClassifier",
        "seed": seed,
        "optuna": {
            "n_trials": 30,
            "best_params": best_params,
            "best_val_f1_attack": best_f1_val,
        },
        "train_balanced": {
            "attack_to_benign": FULL_TRAIN_ATTACK_TO_BENIGN,
            "benign": int(benign_count),
            "attack_sampled": int(attack_count),
            "rows": int(len(y_full)),
            "features_used": int(X_full.shape[1]),
        },
        "test_full": test_metrics,
        "runtime_sec": float(time.time() - t0),
    }

    result_json = os.path.join(OUT_DIR, f"automl_optuna_hgb_results_seed_{seed}.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved to:", result_json)
    print(json.dumps(out, indent=2))

    return out


if __name__ == "__main__":
    main()


# === Wrapper for Experiment Runner ===
def run_automl_experiment(seed=42):
    """
    Wrapper so experiment_runner.py can call this module directly.
    Always runs for the given seed (no caching).
    Returns metrics from full test stream evaluation.
    """
    print(f"[INFO] Running AutoML-HGB experiment for seed {seed}...")
    results = main(seed=seed)

    # Extract full test stream metrics
    test_full = results.get("test_full", {})
    features = results.get("train_balanced", {}).get("features_used", None)
    runtime_sec = results.get("runtime_sec", 1e-6)

    # Construct return dictionary with full test stream metrics
    return {
        "test_full_f1": test_full.get("f1_attack"),
        "test_full_precision": test_full.get("precision_attack"),
        "test_full_recall": test_full.get("recall_attack"),
        "test_auc_roc": test_full.get("roc_auc_sample"),
        "test_auc_pr": test_full.get("pr_auc_sample"),
        "features_selected": features,
        "runtime_sec": runtime_sec,
    }
