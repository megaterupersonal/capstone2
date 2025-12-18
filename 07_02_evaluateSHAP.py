# 07_02_evaluateSHAP.py
# Build global SHAP importance tables for all models
# - Loads *_shap_values_test.npy and *_shap_meta.json
# - Computes mean(|SHAP|) per feature (per model)
# - Normalizes importance within each model (sum = 1)
# - Saves:
#     * {key}_shap_global_importance.csv  (per model)
#     * shap_global_importance_all_models.csv (combined, union of features)

import os
import json
import numpy as np
import pandas as pd

FEAT_DIR  = r"H:\A\parquet files\features"
SHAP_DIR = os.path.join(FEAT_DIR, "shap_outputs")

MODELS = [
    ("rf",  "Random Forest"),
    ("dt",  "Decision Tree"),
    ("xgb", "XGBoost"),
    ("lr",  "Logistic Regression"),
    ("svm", "SVM (RBF)"),
    ("knn", "KNN"),
    ("nb",  "Naive Bayes"),
]

os.makedirs(SHAP_DIR, exist_ok=True)

per_model_tables = {}

# ============================================================
# 1. Per-model importance tables
# ============================================================

for key, pretty_name in MODELS:
    meta_path = os.path.join(SHAP_DIR, f"{key}_shap_meta.json")
    vals_path = os.path.join(SHAP_DIR, f"{key}_shap_values_test.npy")

    if not (os.path.exists(meta_path) and os.path.exists(vals_path)):
        print(f"[SKIP] {pretty_name} ({key}) → missing SHAP files")
        continue

    print(f"\n[MODEL] {pretty_name} ({key})")
    print(f"[LOAD] {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if "feature_names" not in meta:
        raise KeyError(f"'feature_names' missing in {meta_path}")

    feature_names = list(meta["feature_names"])

    print(f"[LOAD] {vals_path}")
    shap_vals = np.load(vals_path)
    shap_vals = np.array(shap_vals)

    # Ensure 2D (n_samples, n_features)
    # Backwards safety: if shape is [classes, n_samples, n_features],
    # take last index (assumed positive class)
    if shap_vals.ndim > 2:
        shap_vals = shap_vals[-1]

    if shap_vals.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP shape for {key}: {shap_vals.shape} "
            f"(expected 2D: n_samples × n_features)."
        )

    if shap_vals.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature mismatch for {key}: "
            f"shap_vals has {shap_vals.shape[1]} cols, "
            f"feature_names has {len(feature_names)} entries."
        )

    # Global importance = mean absolute SHAP over samples
    mean_abs = np.mean(np.abs(shap_vals), axis=0)

    # Normalize within model (sum to 1)
    total = mean_abs.sum()
    if total == 0:
        importance_norm = np.zeros_like(mean_abs)
    else:
        importance_norm = mean_abs / total

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "importance_norm": importance_norm,
    })

    df_imp.sort_values("mean_abs_shap", ascending=False, inplace=True)
    df_imp["rank"] = np.arange(1, len(df_imp) + 1)

    per_model_tables[key] = df_imp

    out_csv = os.path.join(SHAP_DIR, f"{key}_shap_global_importance.csv")
    df_imp.to_csv(out_csv, index=False)
    print(f"[SAVE] Global importance table → {out_csv}")

# ============================================================
# 2. Combined + averaged table (union of all features)
# ============================================================

if not per_model_tables:
    raise RuntimeError("No per-model SHAP tables were created; nothing to combine.")

print("\n[COMBINE] Building combined importance table for all models...")

# Union of all features across all models
all_features = set()
for df_imp in per_model_tables.values():
    all_features.update(df_imp["feature"].tolist())

# Sort for deterministic output.
all_features = sorted(all_features)

combined = pd.DataFrame({"feature": all_features})

present_keys = []
for key, pretty_name in MODELS:
    if key not in per_model_tables:
        continue

    present_keys.append(key)
    df_imp = per_model_tables[key]

    s_imp = df_imp.set_index("feature")["importance_norm"]
    s_rank = df_imp.set_index("feature")["rank"]

    combined[f"imp_{key}"] = combined["feature"].map(s_imp)
    combined[f"rank_{key}"] = combined["feature"].map(s_rank)

# Count in how many models each feature appears
imp_cols = [f"imp_{k}" for k in present_keys]
combined["model_count"] = combined[imp_cols].notna().sum(axis=1)

# Average normalized importance across available models (skip NaNs)
combined["imp_mean"] = combined[imp_cols].mean(axis=1, skipna=True)

# Rank features by averaged importance (1 = most important)
combined["imp_mean_rank"] = (
    combined["imp_mean"]
    .rank(ascending=False, method="min")
    .astype(int)
)

# Sort by global rank for easier inspection
combined.sort_values(["imp_mean_rank", "feature"], inplace=True)

combined_out = os.path.join(SHAP_DIR, "shap_global_importance_all_models.csv")
combined.to_csv(combined_out, index=False)
print(f"[SAVE] Combined importance table → {combined_out}")

print("\n[DONE] SHAP global importance tables created.")
