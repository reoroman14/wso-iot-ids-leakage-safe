import os
import json
import time
import math
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import os
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)


# =========================
# PATHS
# =========================
from config import TRAIN_DIR, TEST_DIR, OUT_DIR  # paths come from config.py


os.makedirs(OUT_DIR, exist_ok=True)

# OUT_JSON will be seed-specific


# =========================
# DATA SETTINGS
# =========================
DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]  # keep consistent with your pipeline
LABEL_CANDIDATES = ["y", "Y", "Label", "label", "class", "Class"]

# Proxy data used inside the feature-selection loop (kept small on purpose)
FS_PROXY_PER_CLASS = 20_000     # 20k benign + 20k attack = 40k total
FS_VAL_RATIO = 0.2

# Fitness penalty weight for number of selected features
# score = F1_attack - ALPHA * (k / d)
ALPHA = 0.02

# Final training (same as your HGB baseline: all benign + equal attacks)
FINAL_ATTACK_TO_BENIGN = 1.0

# Test AUC computed on reservoir sample of scores
AUC_SAMPLE_SIZE = 1_000_000


# =========================
# WSO SETTINGS (start reasonable, not too slow)
# =========================



# WSO continuous bounds (for binary transfer via sigmoid)
# [-6, 6] gives strong 0/1 probabilities after sigmoid
LB = -6.0
UB =  6.0

# RANDOM_SEED will be passed as function parameter


# =========================
# HELPERS: FILES / LABEL / NUMERIC
# =========================
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
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"No label column found. Example columns: {list(df.columns)[:25]}")


def to_binary_y(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        s = s.map({"benign": 0, "attack": 1})
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.int8).to_numpy()
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int8).to_numpy()


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


def get_feature_names(train_files):
    df0 = pd.read_parquet(train_files[0])
    X0, _, lab = safe_numeric_Xy(df0)
    return list(X0.columns), lab


# =========================
# BUILD PROXY DATA FOR FS LOOP
# =========================
def stream_balanced_sample(parquet_files, per_class, seed=42):
    rng = np.random.default_rng(seed)
    pos_parts, neg_parts = [], []
    pos_count = 0
    neg_count = 0
    label_seen = None

    for f in parquet_files:
        df = pd.read_parquet(f)
        try:
            X, y, lab = safe_numeric_Xy(df)
        except KeyError:
            continue

        if label_seen is None:
            label_seen = lab
            print(f"[INFO] Detected label column: {label_seen}")

        df2 = X.copy()
        df2["__y__"] = y

        # shuffle within part
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
        raise RuntimeError(f"Not enough samples: benign={neg_count}, attack={pos_count}, need={per_class} each.")

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


# =========================
# FINAL BALANCED TRAIN (same as your baseline)
# =========================
def build_full_balanced_train(train_files, attack_to_benign=1.0, seed=42):
    rng = np.random.default_rng(seed)

    benign_parts = []
    benign_count = 0
    for f in train_files:
        df = pd.read_parquet(f)
        X, y, _ = safe_numeric_Xy(df)
        b = X[y == 0]
        if len(b) > 0:
            benign_parts.append(b)
            benign_count += len(b)

    if benign_count == 0:
        raise RuntimeError("No benign samples found in TRAIN dataset.")

    target_attacks = int(benign_count * attack_to_benign)
    attack_parts = []
    attack_count = 0

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


# =========================
# STREAMING TEST METRICS
# =========================
def stream_test_metrics(model, test_files, feature_cols, auc_sample_size=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    tn = fp = fn = tp = 0

    scores_buf = np.empty(auc_sample_size, dtype=np.float32)
    y_buf = np.empty(auc_sample_size, dtype=np.int8)
    seen = 0
    filled = 0

    for f in test_files:
        df = pd.read_parquet(f)
        X, y, _ = safe_numeric_Xy(df)

        # restrict to selected features
        X = X[feature_cols]
        Xnp = X.astype(np.float32, copy=False).to_numpy()

        proba = model.predict_proba(Xnp)[:, 1]
        pred = (proba >= 0.5).astype(np.int8)

        tn += int(((pred == 0) & (y == 0)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        tp += int(((pred == 1) & (y == 1)).sum())

        # reservoir sample for AUC
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
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision_attack": float(precision),
        "recall_attack": float(recall),
        "f1_attack": float(f1),
        "roc_auc_sample": roc_auc,
        "pr_auc_sample": pr_auc,
        "auc_sample_size_used": int(filled),
    }


# =========================
# BINARY MAPPING + FITNESS
# =========================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def position_to_mask(pos, min_features=1):
    probs = sigmoid(pos)
    mask = (probs >= 0.5)

    if mask.sum() < min_features:
        # force at least one feature: pick largest probability
        top = np.argsort(-probs)[:min_features]
        mask[:] = False
        mask[top] = True
    return mask


def mask_to_key(mask: np.ndarray) -> bytes:
    # compact key for caching
    return np.packbits(mask.astype(np.uint8)).tobytes()


def make_fitness_fn(X_tr, y_tr, X_va, y_va, alpha, fixed_hgb_params, seed=42):
    cache = {}  # mask_key -> (fitness, f1, k)

    X_tr_np_full = X_tr.astype(np.float32).to_numpy()
    X_va_np_full = X_va.astype(np.float32).to_numpy()

    d = X_tr.shape[1]

    def fitness(pos):
        mask = position_to_mask(pos, min_features=1)
        key = mask_to_key(mask)
        if key in cache:
            return cache[key][0]

        cols_idx = np.where(mask)[0]
        k = int(len(cols_idx))

        Xtr = X_tr_np_full[:, cols_idx]
        Xva = X_va_np_full[:, cols_idx]

        clf = HistGradientBoostingClassifier(**fixed_hgb_params, random_state=seed)
        clf.fit(Xtr, y_tr)

        proba = clf.predict_proba(Xva)[:, 1]
        pred = (proba >= 0.5).astype(np.int8)

        f1 = float(f1_score(y_va, pred, pos_label=1))

        score = f1 - alpha * (k / d)
        fit = -score  # minimize

        cache[key] = (fit, f1, k)
        return fit

    def get_cache():
        return cache

    return fitness, get_cache


# =========================
# ORIGINAL WSO (PORTED FROM MATLAB WSO.m)
# =========================
def wso_optimize(n_agents, max_iters, dim, lb, ub, fobj, seed=42):
    rng = np.random.default_rng(seed)

    # initialization
    positions = rng.random((n_agents, dim)) * (ub - lb) + lb
    v = np.zeros_like(positions)

    # evaluate
    fitness = np.array([fobj(positions[i]) for i in range(n_agents)], dtype=np.float64)
    wbest = positions.copy()
    fmin0 = float(fitness.min())
    gbest = wbest[int(np.argmin(fitness))].copy()

    # parameters (from MATLAB)
    fmax = 0.75
    fmin = 0.07
    tau = 4.11
    mu = 2.0 / abs(2 - tau - math.sqrt(tau * tau - 4 * tau))
    pmin = 0.5
    pmax = 1.5
    a0 = 6.250
    a1 = 100
    a2 = 0.0005

    curve = []

    for ite in range(1, max_iters + 1):
        mv = 1.0 / (a0 + math.exp((max_iters / 2.0 - ite) / a1))
        s_s = abs((1 - math.exp(-a2 * ite / max_iters)))

        p1 = pmax + (pmax - pmin) * math.exp(-(4 * ite / max_iters) ** 2)
        p2 = pmin + (pmax - pmin) * math.exp(-(4 * ite / max_iters) ** 2)

        # nu indices (MATLAB: 1..whiteSharks)
        nu = rng.integers(0, n_agents, size=n_agents)

        # update speed
        for i in range(n_agents):
            rmin, rmax = 1.0, 3.0
            rr = rmin + rng.random() * (rmax - rmin)
            wr = abs(((2 * rng.random()) - (rng.random() + rng.random())) / rr)
            v[i] = mu * v[i] + wr * (wbest[nu[i]] - positions[i])

            # optional alternative in MATLAB is commented; we keep main equation

        # update position (wavy motion + boundary)
        for i in range(n_agents):
            f = fmin + (fmax - fmin) / (fmax + fmin)

            a = (positions[i] - ub) > 0
            b = (positions[i] - lb) < 0
            wo = np.logical_xor(a, b)

            if rng.random() < mv:
                # random allocation to boundary where violated
                newpos = positions[i].copy()
                newpos = newpos * (~wo) + (ub * a + lb * b)
                positions[i] = newpos
            else:
                positions[i] = positions[i] + v[i] / f

        # "fish school" correction
        for i in range(n_agents):
            for j in range(dim):
                if rng.random() < s_s:
                    dist = abs(rng.random() * (gbest[j] - positions[i, j]))
                    if i == 0:
                        positions[i, j] = gbest[j] + rng.random() * dist * (1 if rng.random() - 0.5 >= 0 else -1)
                    else:
                        wso_pos_ij = gbest[j] + rng.random() * dist * (1 if rng.random() - 0.5 >= 0 else -1)
                        positions[i, j] = ((wso_pos_ij + positions[i - 1, j]) / 2.0) * rng.random()

        # evaluate and update bests
        for i in range(n_agents):
            # enforce bounds
            if np.all((positions[i] >= lb) & (positions[i] <= ub)):
                fit = float(fobj(positions[i]))
                if fit < fitness[i]:
                    fitness[i] = fit
                    wbest[i] = positions[i].copy()

        # update global best properly
        best_i = int(np.argmin(fitness))
        if float(fitness[best_i]) < fmin0:
            fmin0 = float(fitness[best_i])
            gbest = wbest[best_i].copy()

        curve.append(fmin0)
        print(f"WSO iter {ite}/{max_iters} best_fitness={fmin0:.6f}")

    return fmin0, gbest, curve


# =========================
# MAIN
# =========================
def main(seed=42, alpha=0.02, pop_size=16, iters=20):
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # WSO settings (override from ablation)
    N_SHARKS = pop_size
    MAX_ITERS = iters
    ALPHA = alpha

    train_files = list_parquet_files(TRAIN_DIR)
    test_files = list_parquet_files(TEST_DIR)

    feature_names, label_col = get_feature_names(train_files)
    d = len(feature_names)
    print(f"[INFO] Features after drops: {d}")
    print(f"[INFO] Label column detected in data: {label_col}")

    # 1) Build proxy for FS fitness
    print("Building FS proxy set (balanced)...")
    X_proxy, y_proxy = stream_balanced_sample(train_files, per_class=FS_PROXY_PER_CLASS, seed=seed)
    X_tr, y_tr, X_va, y_va = train_val_split(X_proxy, y_proxy, val_ratio=FS_VAL_RATIO, seed=seed)
    print(f"[INFO] FS proxy: X={X_proxy.shape}, train={len(X_tr)}, val={len(X_va)}")

    # 2) Fixed HGB params used during FS (keep simple + stable)
    # You can later switch these to AutoML best params if you want an extra variant.
    fixed_hgb_params = dict(
        learning_rate=0.1,
        max_depth=10,
        max_leaf_nodes=127,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        max_iter=200,
    )

    fobj, get_cache = make_fitness_fn(X_tr, y_tr, X_va, y_va, alpha=ALPHA, fixed_hgb_params=fixed_hgb_params, seed=seed)

    # 3) Run WSO
    print("Running WSO for feature selection...")
    best_fit, best_pos, curve = wso_optimize(
        n_agents=N_SHARKS,
        max_iters=MAX_ITERS,
        dim=d,
        lb=LB,
        ub=UB,
        fobj=fobj,
        seed=seed,
    )

    best_mask = position_to_mask(best_pos, min_features=1)
    selected_cols = [feature_names[i] for i in np.where(best_mask)[0]]
    k = int(best_mask.sum())

    # best cached f1 for this mask (if present)
    cache_key = mask_to_key(best_mask)
    cached = get_cache().get(cache_key, None)
    best_proxy_f1 = None if cached is None else float(cached[1])

    print(f"[RESULT] Selected features: {k}/{d}")
    print("[RESULT] First 20 selected features:", selected_cols[:20])

    # 4) Train final model on full balanced train using selected features
    print("Building full balanced training set for final model...")
    X_full, y_full, benign_count, attack_count = build_full_balanced_train(
        train_files, attack_to_benign=FINAL_ATTACK_TO_BENIGN, seed=seed
    )
    X_full = X_full[selected_cols]
    print(f"[INFO] Final train: X={X_full.shape}, benign={benign_count}, attack_sampled={attack_count}")

    final_model = HistGradientBoostingClassifier(**fixed_hgb_params, random_state=seed)
    print("Training final WSO-FS + HGB...")
    final_model.fit(X_full.astype(np.float32).to_numpy(), y_full)

    # 5) Evaluate on full test (streaming)
    print("Evaluating on full TEST (streaming)...")
    test_metrics = stream_test_metrics(final_model, test_files, selected_cols, auc_sample_size=AUC_SAMPLE_SIZE, seed=seed)

    out = {
        "method": "WSO feature selection (binary transfer + wrapper HGB)",
        "seed": seed,
        "wso": {
            "n_sharks": N_SHARKS,
            "max_iters": MAX_ITERS,
            "lb": LB,
            "ub": UB,
            "best_fitness": float(best_fit),
            "convergence_curve": [float(x) for x in curve],
        },
        "fitness": {
            "proxy_per_class": FS_PROXY_PER_CLASS,
            "val_ratio": FS_VAL_RATIO,
            "alpha_feature_penalty": ALPHA,
            "proxy_best_f1_attack": best_proxy_f1,
        },
        "model": {
            "classifier": "HistGradientBoostingClassifier",
            "fixed_params": fixed_hgb_params,
        },
        "selected_features": {
            "k": int(k),
            "d": int(d),
            "names": selected_cols,
        },
        "train_balanced": {
            "attack_to_benign": FINAL_ATTACK_TO_BENIGN,
            "benign": int(benign_count),
            "attack_sampled": int(attack_count),
            "rows": int(len(y_full)),
        },
        "test_full": test_metrics,
        "runtime_sec_total": float(time.time() - t0),
    }

    out_json = os.path.join(OUT_DIR, f"wso_fs_results_seed_{seed}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved to:", out_json)
    print(json.dumps(out, indent=2))

    return out


if __name__ == "__main__":
    main()


def run_wso_pipeline(seed=42, alpha=0.02, pop_size=16, iters=20):
    """
    Wrapper so experiment_runner.py can call this module directly.
    Always runs for the given seed (no caching).
    Returns summary info for experiment_runner.py from full test stream.
    """
    print(f"[INFO] Running WSO feature selection for seed {seed}...")
    results = main(seed=seed, alpha=alpha, pop_size=pop_size, iters=iters)

    # Extract key fields from full test stream metrics
    test_full = results.get("test_full", {})
    features_selected = results.get("selected_features", {}).get("k", None)
    runtime_sec = results.get("runtime_sec_total", 1e-6)
    curve = results.get("wso", {}).get("convergence_curve", [])

    # Save convergence data for plotting
    np.save(f"fitness_seed_{seed}.npy", np.array(curve, dtype=np.float32))

    # Return a dictionary that experiment_runner.py expects with full test stream metrics
    return {
        "test_full_f1": test_full.get("f1_attack"),
        "test_full_precision": test_full.get("precision_attack"),
        "test_full_recall": test_full.get("recall_attack"),
        "test_auc_roc": test_full.get("roc_auc_sample"),
        "test_auc_pr": test_full.get("pr_auc_sample"),
        "features_selected": features_selected,
        "runtime_sec": runtime_sec,
    }

