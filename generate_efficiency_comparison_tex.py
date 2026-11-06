import glob
import json
import os
import numpy as np
import pandas as pd

def load_seed_metrics(paths, runtime_key_candidates):
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)

        test_full = d.get("test_full", {})
        test_auc = d.get("test_auc_sample", {})

        runtime = np.nan
        for k in runtime_key_candidates:
            if k in d and d[k] is not None:
                runtime = float(d[k])
                break

        # Schema fallback: prefer test_full.*_sample, else test_auc_sample.*
        roc_auc = test_full.get("roc_auc_sample", test_auc.get("roc_auc", np.nan))
        pr_auc = test_full.get("pr_auc_sample", test_auc.get("pr_auc", np.nan))

        rows.append({
            "F1 (Attack)": float(test_full.get("f1_attack", np.nan)),
            "ROC-AUC": float(roc_auc),
            "PR-AUC": float(pr_auc),
            "Runtime (s)": runtime,
        })
    return pd.DataFrame(rows)

def fmt_pm(series, decimals=4):
    series = pd.to_numeric(series, errors="coerce")
    m = series.mean()
    s = series.std(ddof=1) if len(series.dropna()) > 1 else 0.0
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"

def main():
    os.makedirs("paper_outputs/tables", exist_ok=True)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    baseline_paths = sorted(glob.glob(os.path.join(BASE_DIR, "outputs", "baseline_results_seed_*.json")))
    automl_paths   = sorted(glob.glob(os.path.join(BASE_DIR, "outputs", "automl_optuna_hgb_results_seed_*.json")))
    wso_paths      = sorted(glob.glob(os.path.join(BASE_DIR, "outputs", "wso_fs_results_seed_*.json")))

    if not baseline_paths:
        raise FileNotFoundError("Missing: outputs/baseline_results_seed_*.json")
    if not automl_paths:
        raise FileNotFoundError("Missing: outputs/automl_optuna_hgb_results_seed_*.json")
    if not wso_paths:
        raise FileNotFoundError("Missing: outputs/wso_fs_results_seed_*.json")

    df_base = load_seed_metrics(baseline_paths, runtime_key_candidates=["runtime_sec_total", "runtime_sec"])
    df_auto = load_seed_metrics(automl_paths,   runtime_key_candidates=["runtime_sec_total", "runtime_sec"])
    df_wso  = load_seed_metrics(wso_paths,      runtime_key_candidates=["runtime_sec_total", "runtime_sec"])

    out = pd.DataFrame([
        {
            "Method": "Baseline (HGB)",
            "F1 (Attack)": fmt_pm(df_base["F1 (Attack)"], 4),
            "ROC-AUC": fmt_pm(df_base["ROC-AUC"], 4),
            "PR-AUC": fmt_pm(df_base["PR-AUC"], 4),
            "Runtime (s)": fmt_pm(df_base["Runtime (s)"], 1),
        },
        {
            "Method": "AutoML (Optuna-HGB)",
            "F1 (Attack)": fmt_pm(df_auto["F1 (Attack)"], 4),
            "ROC-AUC": fmt_pm(df_auto["ROC-AUC"], 4),
            "PR-AUC": fmt_pm(df_auto["PR-AUC"], 4),
            "Runtime (s)": fmt_pm(df_auto["Runtime (s)"], 1),
        },
        {
            "Method": "WSO-FS",
            "F1 (Attack)": fmt_pm(df_wso["F1 (Attack)"], 4),
            "ROC-AUC": fmt_pm(df_wso["ROC-AUC"], 4),
            "PR-AUC": fmt_pm(df_wso["PR-AUC"], 4),
            "Runtime (s)": fmt_pm(df_wso["Runtime (s)"], 1),
        },
    ])

    out_path = "paper_outputs/tables/efficiency_comparison.tex"

    # Generate LaTeX with manual caption/label
    latex_body = out.to_latex(index=False, escape=False)

    # Wrap with table environment
    full_latex = (
        "\\begin{table}\n"
        "\\caption{Efficiency and performance comparison of all models (mean $\\pm$ std over seeds).}\n"
        "\\label{tab:efficiency}\n"
        + latex_body +
        "\\end{table}\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_latex)

    print(f"Saved -> {out_path}")
    print(f"Seeds found: baseline={len(baseline_paths)}, automl={len(automl_paths)}, wso={len(wso_paths)}")

if __name__ == "__main__":
    main()
