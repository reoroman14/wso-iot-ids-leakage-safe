import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# =========================
# PATHS
# =========================
from config import TRAIN_DIR, TEST_DIR  # paths come from config.py

# You already trained these models in memory before.
# For throughput, we retrain quickly on a small balanced set
# (model structure matters, not final accuracy).


# =========================
# SETTINGS
# =========================
LABEL_CANDIDATES = ["y", "Y", "Label", "label"]
DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]

BATCH_ROWS = 200_000
MAX_TEST_ROWS = 1_000_000   # measure on 1M flows for stable timing
SEED = 42

# Feature sets
FEATURES_FULL = None        # will be inferred
FEATURES_WSO = None         # load from JSON manually below

# =========================
# HELPERS
# =========================
def list_parquet_files(folder):
    files = []
    for root, _, fs in os.walk(folder):
        for f in fs:
            if f.endswith(".parquet"):
                files.append(os.path.join(root, f))
    return sorted(files)


def detect_label(df):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("No label column found")


def load_X(df, feature_cols=None):
    label_col = detect_label(df)
    X = df.drop(columns=[label_col])

    for c in DROP_COLS:
        if c in X.columns:
            X = X.drop(columns=[c])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if feature_cols is not None:
        X = X[feature_cols]

    return X.astype(np.float32, copy=False)


def quick_train_hgb(feature_cols):
    # train a small balanced proxy model for throughput timing
    rng = np.random.default_rng(SEED)
    Xb, Xa = [], []
    files = list_parquet_files(TRAIN_DIR)

    for f in files:
        df = pd.read_parquet(f)
        label = detect_label(df)
        df["__y__"] = df[label]
        X = load_X(df, feature_cols)

        b = X[df["__y__"] == 0]
        a = X[df["__y__"] == 1]

        if len(b) > 0:
            Xb.append(b.iloc[:2000])
        if len(a) > 0:
            Xa.append(a.iloc[:2000])

        if sum(len(x) for x in Xb) >= 20_000:
            break

    Xb = pd.concat(Xb)
    Xa = pd.concat(Xa).iloc[:len(Xb)]

    yb = np.zeros(len(Xb), dtype=np.int8)
    ya = np.ones(len(Xa), dtype=np.int8)

    Xtrain = pd.concat([Xb, Xa])
    ytrain = np.concatenate([yb, ya])

    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=10,
        max_leaf_nodes=127,
        min_samples_leaf=20,
        max_iter=200,
        random_state=SEED
    )
    model.fit(Xtrain.to_numpy(), ytrain)
    return model


def measure_throughput(model, feature_cols):
    files = list_parquet_files(TEST_DIR)
    total_rows = 0
    t0 = time.time()

    for f in files:
        if total_rows >= MAX_TEST_ROWS:
            break
        df = pd.read_parquet(f)
        X = load_X(df, feature_cols)

        for i in range(0, len(X), BATCH_ROWS):
            batch = X.iloc[i:i+BATCH_ROWS]
            model.predict_proba(batch.to_numpy())
            total_rows += len(batch)
            if total_rows >= MAX_TEST_ROWS:
                break

    elapsed = time.time() - t0
    return total_rows / elapsed


# =========================
# MAIN
# =========================
def main():
    global FEATURES_FULL, FEATURES_WSO

    # infer full feature set
    df0 = pd.read_parquet(list_parquet_files(TEST_DIR)[0])
    FEATURES_FULL = list(load_X(df0).columns)

    # LOAD WSO FEATURES (copy-paste from JSON)
    FEATURES_WSO = [
        "Src Port","Dst Port","Protocol","Flow Duration",
        "Total Length of Bwd Packet","Fwd Packet Length Max",
        "Fwd Packet Length Min","Bwd Packet Length Max",
        "Flow IAT Mean","Fwd IAT Std","Bwd IAT Min",
        "Bwd URG Flags","Bwd Header Length","Fwd Packets/s",
        "Packet Length Mean","Packet Length Std","SYN Flag Count",
        "Down/Up Ratio","Fwd Packet/Bulk Avg","Bwd Bytes/Bulk Avg",
        "Bwd Packet/Bulk Avg","Bwd Bulk Rate Avg","Subflow Fwd Bytes",
        "Subflow Bwd Packets","FWD Init Win Bytes","Bwd Init Win Bytes",
        "Fwd Seg Size Min","Active Std","Active Max","Active Min",
        "Idle Mean","Idle Max","Idle Min"
    ]

    print("Training proxy models...")
    hgb_full = quick_train_hgb(FEATURES_FULL)
    hgb_wso  = quick_train_hgb(FEATURES_WSO)

    print("Measuring throughput...")
    thr_full = measure_throughput(hgb_full, FEATURES_FULL)
    thr_wso  = measure_throughput(hgb_wso, FEATURES_WSO)

    print("\n=== THROUGHPUT RESULTS ===")
    print(f"HGB / AutoML-HGB (79 features): {thr_full:.0f} flows/sec")
    print(f"WSO-FS + HGB (33 features):     {thr_wso:.0f} flows/sec")


if __name__ == "__main__":
    main()
