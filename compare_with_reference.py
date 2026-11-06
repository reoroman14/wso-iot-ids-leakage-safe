# compare_with_reference.py
# Compares the JSON results you just produced (outputs/) with the saved reference results
# of the paper (reference_results/). Prints the difference per model and seed.
#
# Usage:
#   python compare_with_reference.py                 -> all models, all seeds found in outputs/
#   python compare_with_reference.py --tol 0.002     -> change the F1 tolerance (default 0.001)
#
# Note: in the reference files, the WSO-FS run of seed 42 used alpha 0.05 and 10 iterations,
# while seeds 43 to 46 used alpha 0.02 and 20 iterations. The script prints these settings so you
# can see whether your run used the same ones.

import os
import json
import sys

from config import ROOT, OUT_DIR, SEEDS

REF_DIR = os.path.join(str(ROOT), "reference_results")
TOL = 0.001
if "--tol" in sys.argv:
    TOL = float(sys.argv[sys.argv.index("--tol") + 1])

MODELS = [
    ("Baseline", "baseline_results_seed_{s}.json"),
    ("AutoML", "automl_optuna_hgb_results_seed_{s}.json"),
    ("WSO-FS", "wso_fs_results_seed_{s}.json"),
]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def features(d):
    # number of features used by the model
    if "selected_features" in d:
        return d["selected_features"].get("k")
    if "num_features" in d:
        return d.get("num_features")
    return d.get("train_balanced", {}).get("features_used")


def wso_settings(d):
    if "wso" in d:
        return f"alpha={d.get('fitness', {}).get('alpha_feature_penalty')}, iters={d['wso'].get('max_iters')}, pop={d['wso'].get('n_sharks')}"
    return ""


print(f"F1 tolerance: {TOL}\n")
print(f"{'model':9s} {'seed':5s} {'F1 ref':8s} {'F1 now':8s} {'diff':9s} {'feat ref':9s} {'feat now':9s} status")
bad = 0
checked = 0
for name, pattern in MODELS:
    for s in SEEDS:
        fn = pattern.format(s=s)
        ref = load(os.path.join(REF_DIR, fn))
        now = load(os.path.join(OUT_DIR, fn))
        if ref is None or now is None:
            if now is None and ref is not None:
                print(f"{name:9s} {s:<5d} (no result in outputs/ yet)")
            continue
        f_ref = ref["test_full"]["f1_attack"]
        f_now = now["test_full"]["f1_attack"]
        diff = f_now - f_ref
        k_ref, k_now = features(ref), features(now)
        ok = abs(diff) <= TOL and k_ref == k_now
        checked += 1
        bad += (not ok)
        print(f"{name:9s} {s:<5d} {f_ref:<8.4f} {f_now:<8.4f} {diff:<+9.4f} {str(k_ref):9s} {str(k_now):9s} {'OK' if ok else 'DIFFERENT'}")
        if name == "WSO-FS" and wso_settings(ref) != wso_settings(now):
            print(f"          settings ref: {wso_settings(ref)} | now: {wso_settings(now)}")

print(f"\nChecked {checked} results, {bad} differ.")
