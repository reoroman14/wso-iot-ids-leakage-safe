import glob
import numpy as np
import pandas as pd
import json


paths = sorted(glob.glob("outputs/wso_fs_results_seed_*.json"))
if not paths:
    print("[WARN] No WSO seed JSON files found: outputs/wso_fs_results_seed_*.json")
else:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)

        test_full = d.get("test_full", {})
        test_auc = d.get("test_auc_sample", {})

        # Schema fallback: prefer test_full.*_sample, else test_auc_sample.*
        roc_auc = test_full.get("roc_auc_sample", test_auc.get("roc_auc", np.nan))
        pr_auc = test_full.get("pr_auc_sample", test_auc.get("pr_auc", np.nan))

        rows.append({
            "F1 (Attack)": float(test_full.get("f1_attack", np.nan)),
            "ROC-AUC": float(roc_auc),
            "PR-AUC": float(pr_auc),
            "Runtime (s)": float(d.get("runtime_sec_total", d.get("runtime_sec", np.nan))),
            "#Features": float(d.get("selected_features", {}).get("k", np.nan)),
        })

    df = pd.DataFrame(rows)

    def fmt_pm(series, decimals=4):
        m = series.mean()
        s = series.std(ddof=1) if len(series) > 1 else 0.0
        return f"{m:.{decimals}f} ± {s:.{decimals}f}"

    summary = pd.DataFrame([{
        "F1 (Attack)": fmt_pm(df["F1 (Attack)"], 4),
        "ROC-AUC": fmt_pm(df["ROC-AUC"], 4),
        "PR-AUC": fmt_pm(df["PR-AUC"], 4),
        "Runtime (s)": fmt_pm(df["Runtime (s)"], 1),
        "#Features": fmt_pm(df["#Features"], 1),
    }])

    out_path = "paper_outputs/tables/wso_results.tex"

    # Generate LaTeX with manual caption/label
    latex_body = summary.to_latex(index=False, escape=False)

    # Wrap with table environment
    full_latex = (
        "\\begin{table}\n"
        "\\caption{WSO-FS final evaluation results (mean $\\pm$ std over seeds).}\n"
        "\\label{tab:wso_results}\n"
        + latex_body +
        "\\end{table}\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_latex)

    print(f"Saved -> {out_path}")
    print(f"Read {len(paths)} seed files.")
