# ablation_sweep.py
# WSO sensitivity study over alpha (feature penalty) and population size.
# Restored from the original ablation script (git history, first commit), with two fixes:
#   1. paths and grid come from config.py instead of D:/code
#   2. the F1 value is read from the key "test_full_f1" (the original read "f1_attack",
#      which does not exist in the returned dictionary, so every F1 came out as NaN)
# Note: the original sweep used 10 WSO iterations (the main experiments use 20). Kept as in the original.
# Slow: 16 full WSO runs. Run it only if you need the ablation figure and table.
# Safe: the main result files of the sweep seed are backed up and restored (see PROTECTED below).

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from wso_feature_selection import run_wso_pipeline
from config import ABLATION_ALPHAS, ABLATION_POPULATIONS, ABLATION_ITERS, ABLATION_SEED, OUT_DIR  # grid is set in config.py

# === Configuration ===
ALPHA_VALUES = ABLATION_ALPHAS
POP_SIZES = ABLATION_POPULATIONS
ITERS = ABLATION_ITERS
SEED = ABLATION_SEED

# === Ensure output directories exist ===
os.makedirs("paper_outputs/figures", exist_ok=True)
os.makedirs("paper_outputs/tables", exist_ok=True)

results = []
print("Running Ablation Study for WSO-FS...\n")

# The pipeline writes its result files with a fixed name per seed. The sweep reuses the seed of the
# main experiments, so it would overwrite those files. Back them up now and restore them at the end
# (also when the sweep is stopped with Ctrl+C), so the main results are never touched.
PROTECTED = [
    os.path.join(OUT_DIR, f"wso_fs_results_seed_{SEED}.json"),
    f"fitness_seed_{SEED}.npy",
]
backup = {}
for p in PROTECTED:
    if os.path.exists(p):
        with open(p, "rb") as f:
            backup[p] = f.read()

# === Run experiments ===
try:
    for alpha in ALPHA_VALUES:
        for pop in POP_SIZES:
            print(f"Running with alpha={alpha:.3f}, population={pop}, iters={ITERS}")
            t0 = time.time()
            try:
                res = run_wso_pipeline(seed=SEED, alpha=alpha, pop_size=pop, iters=ITERS)
            except Exception as e:
                print(f"Error at alpha={alpha}, population={pop}: {e}")
                continue
            runtime = time.time() - t0

            results.append({
                "alpha": alpha,
                "population": pop,
                "f1_score": res.get("test_full_f1", np.nan),
                "features": res.get("features_selected", np.nan),
                "runtime_sec": runtime
            })
finally:
    for p in PROTECTED:
        if p in backup:
            with open(p, "wb") as f:
                f.write(backup[p])
        elif os.path.exists(p):
            os.remove(p)
    print("[INFO] Main result files for seed", SEED, "restored.")

# === Convert results to structured matrices ===
alphas = sorted(set(r["alpha"] for r in results))
pops = sorted(set(r["population"] for r in results))

F1 = np.zeros((len(alphas), len(pops)))
FEATS = np.zeros_like(F1)
RUNTIME = np.zeros_like(F1)

for i, a in enumerate(alphas):
    for j, p in enumerate(pops):
        match = [r for r in results if r["alpha"] == a and r["population"] == p]
        if match:
            F1[i, j] = match[0]["f1_score"]
            FEATS[i, j] = match[0]["features"]
            RUNTIME[i, j] = match[0]["runtime_sec"]

# === Plot 3-panel figure ===
plt.figure(figsize=(14, 4))

# 1. F1-score plot
plt.subplot(1, 3, 1)
for i, a in enumerate(alphas):
    plt.plot(pops, F1[i], marker="o", label=f"α={a}")
plt.xlabel("Population Size (N)")
plt.ylabel("F1-Score (Attack Class)")
plt.title("Effect of α and N on F1-Score")
plt.grid(True)
plt.legend()

# 2. Runtime plot
plt.subplot(1, 3, 2)
for i, a in enumerate(alphas):
    plt.plot(pops, RUNTIME[i], marker="s", label=f"α={a}")
plt.xlabel("Population Size (N)")
plt.ylabel("Runtime (s)")
plt.title("Effect of α and N on Runtime")
plt.grid(True)

# 3. Selected features plot
plt.subplot(1, 3, 3)
for i, a in enumerate(alphas):
    plt.plot(pops, FEATS[i], marker="^", label=f"α={a}")
plt.xlabel("Population Size (N)")
plt.ylabel("Selected Features")
plt.title("Effect of α and N on Feature Count")
plt.grid(True)

plt.tight_layout()

# === Save outputs ===
plt.savefig("paper_outputs/figures/ablation_plot.pdf", bbox_inches="tight")

with open("paper_outputs/tables/ablation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
# === Also save the LaTeX table for the paper ===
import pandas as pd

df_ablation = pd.DataFrame(results).sort_values(["alpha", "population"])
# F1 is kept in the JSON only; the paper's Table 5 reports features and runtime only
df_ablation = df_ablation.drop(columns=["f1_score"])
df_ablation["runtime_sec"] = df_ablation["runtime_sec"].map(lambda x: f"{x:.1f}")
df_ablation["features"] = df_ablation["features"].astype(int, errors="ignore")

ablation_tex_path = "paper_outputs/tables/ablation_results.tex"
with open(ablation_tex_path, "w", encoding="utf-8") as f:
    f.write(df_ablation.to_latex(index=False, escape=False,
                                 caption="Ablation study on α and population size for WSO-FS",
                                 label="tab:ablation"))

print(f"Saved LaTeX table -> {ablation_tex_path}")
print("Saved figure -> paper_outputs/figures/ablation_plot.pdf")
print("Saved data   -> paper_outputs/tables/ablation_results.json")
