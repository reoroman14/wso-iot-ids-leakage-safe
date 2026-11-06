"""
Unified Experiment Runner for the Leakage-Safe IoT IDS paper
Generates publication-ready tables and figures for Overleaf.

Author: Abdulla Amin Aburomman
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# === Import your real experiment scripts ===
from train_baseline_hgb_balanced import train_and_evaluate_hgb
from automl_optuna_hgb import run_automl_experiment
from wso_feature_selection import run_wso_pipeline

# === Reproducibility ===
random.seed(42)
np.random.seed(42)

# === Setup output folders ===
BASE_DIR = os.path.join(os.getcwd(), "paper_outputs")
FIG_DIR = os.path.join(BASE_DIR, "figures")
TAB_DIR = os.path.join(BASE_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# === Actual model runners ===
def run_baseline(seed):
    """Run baseline HGB model and return full test stream metrics."""
    results = train_and_evaluate_hgb(seed=seed)
    return {
        "Model": "Baseline",
        "Seed": seed,
        "Precision": results["test_full_precision"],
        "Recall": results["test_full_recall"],
        "F1": results["test_full_f1"],
        "ROC-AUC": results["test_auc_roc"],
        "PR-AUC": results["test_auc_pr"],
        "Features": results.get("features_selected"),
        "Throughput": 1.0 / (results["runtime_sec"] / 60.0) if results.get("runtime_sec") else None
    }

def run_automl(seed):
    """Run AutoML + Optuna experiment and return full test stream metrics."""
    results = run_automl_experiment(seed=seed)
    return {
        "Model": "AutoML",
        "Seed": seed,
        "Precision": results["test_full_precision"],
        "Recall": results["test_full_recall"],
        "F1": results["test_full_f1"],
        "ROC-AUC": results["test_auc_roc"],
        "PR-AUC": results["test_auc_pr"],
        "Features": results.get("features_selected"),
        "Throughput": 1.0 / (results["runtime_sec"] / 60.0) if results.get("runtime_sec") else None
    }

def run_wso(seed):
    """Run White Shark Optimizer-based feature selection pipeline and return full test stream metrics."""
    results = run_wso_pipeline(seed=seed)
    return {
        "Model": "WSO-FS",
        "Seed": seed,
        "Precision": results["test_full_precision"],
        "Recall": results["test_full_recall"],
        "F1": results["test_full_f1"],
        "ROC-AUC": results["test_auc_roc"],
        "PR-AUC": results["test_auc_pr"],
        "Features": results.get("features_selected"),
        "Throughput": 1.0 / (results["runtime_sec"] / 60.0) if results.get("runtime_sec") else None
    }

# === Main experiment ===
from config import SEEDS  # seeds are defined in config.py
records = []

print("\n Starting multi-seed experiments...\n")

for seed in SEEDS:
    print(f"Running seed {seed}...")
    for runner in [run_baseline, run_automl, run_wso]:
        res = runner(seed)
        records.append(res)

# === Aggregate and summarize ===
df = pd.DataFrame(records)

# Drop 'Seed' column before aggregation
df_agg = df.drop(columns=['Seed'])

summary = df_agg.groupby("Model").agg(["mean", "std"]).round(3)
summary.columns = ['_'.join(col) for col in summary.columns]
summary.to_csv(os.path.join(TAB_DIR, "results_summary.csv"))
print("\n Saved summary CSV -> tables/results_summary.csv")

# === Generate LaTeX tables ===
def fmt_pm(mean_val, std_val, decimals=4):
    """Format value as 'mean ± std' string."""
    return f"{mean_val:.{decimals}f} $\\pm$ {std_val:.{decimals}f}"

def generate_baseline_table():
    """Generate Table 1 with mean ± std format from seed JSONs."""
    import glob
    import json

    baseline_paths = sorted(glob.glob(os.path.join("outputs", "baseline_results_seed_*.json")))
    automl_paths = sorted(glob.glob(os.path.join("outputs", "automl_optuna_hgb_results_seed_*.json")))
    wso_paths = sorted(glob.glob(os.path.join("outputs", "wso_fs_results_seed_*.json")))

    def extract_metrics(paths, model_name):
        """Extract metrics with schema fallback for ROC-AUC/PR-AUC."""
        data = {"precision": [], "recall": [], "f1": [], "roc_auc": [], "pr_auc": [], "features": [], "throughput": []}

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)

            test_full = d.get("test_full", {})
            test_auc = d.get("test_auc_sample", {})

            # Extract basic metrics
            data["precision"].append(test_full.get("precision_attack", np.nan))
            data["recall"].append(test_full.get("recall_attack", np.nan))
            data["f1"].append(test_full.get("f1_attack", np.nan))

            # Schema fallback: prefer test_full.*_sample, else test_auc_sample.*
            roc = test_full.get("roc_auc_sample", test_auc.get("roc_auc", np.nan))
            pr = test_full.get("pr_auc_sample", test_auc.get("pr_auc", np.nan))
            data["roc_auc"].append(roc)
            data["pr_auc"].append(pr)

            # Features
            if model_name == "WSO-FS":
                features = d.get("selected_features", {}).get("k", np.nan)
            else:
                features = d.get("num_features", 79)
            data["features"].append(features)

            # Throughput (flows/min)
            runtime_sec = d.get("runtime_sec", d.get("runtime_sec_total", np.nan))
            if not np.isnan(runtime_sec) and runtime_sec > 0:
                rows_processed = test_full.get("rows_streamed", test_full.get("rows_test_streamed", 0))
                throughput = rows_processed / runtime_sec if rows_processed > 0 else np.nan
            else:
                throughput = np.nan
            data["throughput"].append(throughput)

        return data

    baseline_data = extract_metrics(baseline_paths, "Baseline")
    automl_data = extract_metrics(automl_paths, "AutoML")
    wso_data = extract_metrics(wso_paths, "WSO-FS")

    # Build table rows
    table_rows = []
    for model_name, data in [("Baseline", baseline_data), ("AutoML", automl_data), ("WSO-FS", wso_data)]:
        row = {
            "Model": model_name,
            "Precision": fmt_pm(np.nanmean(data["precision"]), np.nanstd(data["precision"], ddof=1)),
            "Recall": fmt_pm(np.nanmean(data["recall"]), np.nanstd(data["recall"], ddof=1)),
            "F1": fmt_pm(np.nanmean(data["f1"]), np.nanstd(data["f1"], ddof=1)),
            "ROC-AUC": fmt_pm(np.nanmean(data["roc_auc"]), np.nanstd(data["roc_auc"], ddof=1)),
            "PR-AUC": fmt_pm(np.nanmean(data["pr_auc"]), np.nanstd(data["pr_auc"], ddof=1)),
            "Features": fmt_pm(np.nanmean(data["features"]), np.nanstd(data["features"], ddof=1), decimals=1),
            "Throughput (flows/s)": fmt_pm(np.nanmean(data["throughput"]), np.nanstd(data["throughput"], ddof=1), decimals=0)
        }
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows)

    # Generate LaTeX
    latex_str = df_table.to_latex(
        index=False,
        escape=False,
        caption="Performance of baseline, AutoML, and WSO-FS models on full test stream (mean $\\pm$ std over 5 seeds).",
        label="tab:baseline_results",
        column_format="l" + "r" * (len(df_table.columns) - 1)
    )

    with open(os.path.join(TAB_DIR, "baseline_results.tex"), "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"Saved baseline_results.tex with {len(baseline_paths)} baseline, {len(automl_paths)} automl, {len(wso_paths)} wso seeds")

generate_baseline_table()

# === Visualization Theme (Elsevier-style) ===
sns.set_theme(context="paper", style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# === 1. Real Convergence Plot ===
# Expecting each WSO run to save its fitness curve as fitness_seed_<seed>.npy
curves = []
for seed in SEEDS:
    fpath = os.path.join(os.getcwd(), f"fitness_seed_{seed}.npy")
    if os.path.exists(fpath):
        curves.append(np.load(fpath))

if curves:
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    iterations = np.arange(1, len(mean_curve)+1)

    plt.figure(figsize=(6,4))
    plt.plot(iterations, mean_curve, color="#0072B2", linewidth=1.8,
             marker='o', markersize=5, label="Mean Fitness")
    plt.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve,
                     color="#0072B2", alpha=0.2, label="± Std. Dev.")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Convergence Curve of White Shark Optimizer", weight='bold')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "convergence_plot.pdf"),
                bbox_inches='tight', dpi=400)
    print("Saved real convergence plot -> figures/convergence_plot.pdf")
else:
    print(" No fitness files found; skipping convergence plot.")

# === 2. Efficiency–Accuracy Trade-off Plot ===
plt.figure(figsize=(6,4))
models = summary.index
f1 = summary["F1_mean"].values
throughput = summary["Throughput_mean"].fillna(0).values

colors = ["#E69F00", "#009E73", "#0072B2"]
markers = ["o", "s", "^"]

for i, model in enumerate(models):
    plt.scatter(throughput[i], f1[i],
                s=100, marker=markers[i],
                color=colors[i], edgecolor='black',
                linewidth=0.8, label=model)

plt.xlabel("Throughput (flows/sec)")
plt.ylabel("F1-score")
plt.title("Efficiency–Accuracy Trade-off", weight='bold')
plt.legend(title="Model", loc='lower right', frameon=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "tradeoff_plot.pdf"),
            bbox_inches='tight', dpi=400)
print("Saved trade-off plot -> figures/tradeoff_plot.pdf")

print("\n All experiments completed successfully.")