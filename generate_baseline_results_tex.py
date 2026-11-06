import glob
import json
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_seed_metrics(paths):
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)

        test_full = d.get("test_full", {})
        test_auc = d.get("test_auc_sample", {})

        roc_auc = test_full.get("roc_auc_sample", test_auc.get("roc_auc", np.nan))
        pr_auc = test_full.get("pr_auc_sample", test_auc.get("pr_auc", np.nan))

        # Robust feature-count fallback across pipelines:
        # Baseline: num_features
        # AutoML: train_balanced.features_used
        # WSO-FS: selected_features.k
        train_balanced = d.get("train_balanced", {})
        selected_features = d.get("selected_features", {})

        features = d.get("num_features", None)
        if features is None:
            features = train_balanced.get("features_used", None)
        if features is None:
            features = selected_features.get("k", None)

        rows.append({
            "Precision": float(test_full.get("precision_attack", np.nan)),
            "Recall": float(test_full.get("recall_attack", np.nan)),
            "F1": float(test_full.get("f1_attack", np.nan)),
            "ROC-AUC": float(roc_auc),
            "PR-AUC": float(pr_auc),
            "Features": float(features) if features is not None else np.nan,
        })
    return pd.DataFrame(rows)

def fmt_pm(series, decimals=4):
    series = pd.to_numeric(series, errors="coerce")
    m = series.mean()
    s = series.std(ddof=1)
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"

def main():
    os.makedirs("paper_outputs/tables", exist_ok=True)

    baseline_paths = sorted(glob.glob(os.path.join(BASE_DIR, "outputs", "baseline_results_seed_*.json")))
    automl_paths   = sorted(glob.glob(os.path.join(BASE_DIR, "outputs", "automl_optuna_hgb_results_seed_*.json")))
    wso_paths      = sorted(glob.glob(os.path.join(BASE_DIR, "outputs", "wso_fs_results_seed_*.json")))

    df_base = load_seed_metrics(baseline_paths)
    df_auto = load_seed_metrics(automl_paths)
    df_wso  = load_seed_metrics(wso_paths)

    out = pd.DataFrame([
        {
            "Model": "Baseline (HGB)",
            "Precision": fmt_pm(df_base["Precision"]),
            "Recall": fmt_pm(df_base["Recall"]),
            "F1": fmt_pm(df_base["F1"]),
            "ROC-AUC": fmt_pm(df_base["ROC-AUC"]),
            "PR-AUC": fmt_pm(df_base["PR-AUC"]),
            "#Features": fmt_pm(df_base["Features"], 1)
        },
        {
            "Model": "AutoML (Optuna-HGB)",
            "Precision": fmt_pm(df_auto["Precision"]),
            "Recall": fmt_pm(df_auto["Recall"]),
            "F1": fmt_pm(df_auto["F1"]),
            "ROC-AUC": fmt_pm(df_auto["ROC-AUC"]),
            "PR-AUC": fmt_pm(df_auto["PR-AUC"]),
            "#Features": fmt_pm(df_auto["Features"], 1)
        },
        {
            "Model": "WSO-FS",
            "Precision": fmt_pm(df_wso["Precision"]),
            "Recall": fmt_pm(df_wso["Recall"]),
            "F1": fmt_pm(df_wso["F1"]),
            "ROC-AUC": fmt_pm(df_wso["ROC-AUC"]),
            "PR-AUC": fmt_pm(df_wso["PR-AUC"]),
            "#Features": fmt_pm(df_wso["Features"], 1)
        },
    ])

    out_path = "paper_outputs/tables/baseline_results.tex"
    latex = out.to_latex(index=False, escape=False)

    full = (
        "\\begin{table}\n"
        "\\caption{Performance of baseline, AutoML, and WSO-FS models on full test stream (mean $\\pm$ std over 5 seeds).}\n"
        "\\label{tab:baseline_results}\n"
        + latex +
        "\\end{table}\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full)

    print("Saved ->", out_path)

if __name__ == "__main__":
    main()
