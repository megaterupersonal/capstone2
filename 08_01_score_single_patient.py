# 08_01_score_single_patient.py
# Score a single patient using the trained and saved XGBoost model and compute SHAP explanations.

import os
import json
import argparse
import numpy as np
import polars as pl
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================
FEAT_DIR  = r"H:\A\parquet files\features"
LLM_DIR  = r"H:\A\parquet files\features\llm demo"

# Model + feature metadata (still live in FEAT_DIR)
XGB_MODEL_PATH      = os.path.join(FEAT_DIR, "models_xgboost", "xgb_best_model.joblib") #Change to other models (But this is the one set in the scope of the project)
FEATURE_LIST_PATH = os.path.join(FEAT_DIR, "final_tree_feature_list.txt") #BE SURE TO CHANGE IF USING LINEAR MODEL OR TREE MODEL BUT THIS PROJECT USES TREE MODEL
BACKGROUND_PARQUET = os.path.join(FEAT_DIR, "final_tree_features.parquet") #BE SURE TO CHANGE IF USING LINEAR MODEL OR TREE MODEL


# All single-patient outputs go here (JSON + PNG)
OUTPUT_DIR = LLM_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE   = 67
BACKGROUND_N   = 500  # rows for SHAP background


# ==========================================================
# UTILITIES
# ==========================================================
def load_feature_list(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature list not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        feats = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(feats)} feature names from {path}")
    return feats


def load_single_patient_json(path: str):
    """Load a single-row JSON file and return it as a dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Allow either { .. } or [ { .. } ]
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError("Input JSON list must contain exactly one row.")
        row = data[0]
    elif isinstance(data, dict):
        row = data
    else:
        raise ValueError("Input JSON must be a dict or a single-element list.")

    print(f"[INFO] Loaded single patient row from {path}")
    return row


def build_feature_vector(row_dict, feature_names):
    """Make a 1 x n_features numpy array in the correct order."""
    values = []
    missing = []

    for feat in feature_names:
        if feat in row_dict:
            values.append(row_dict[feat])
        else:
            # Some rows might not contain all features; default to 0.0
            values.append(0.0)
            missing.append(feat)

    if missing:
        print(
            f"[WARN] {len(missing)} features missing in JSON. "
            f"Filled with 0.0 (first few: {missing[:5]})"
        )

    X = np.array([values], dtype=float)
    print(f"[INFO] Built feature vector with shape {X.shape}")
    return X


def sample_background(background_parquet, feature_names, n_rows, random_state=67):
    """Sample background data for SHAP TreeExplainer."""
    print(f"[INFO] Loading background data from {background_parquet}")
    df = pl.read_parquet(background_parquet).select(feature_names)
    n_rows = min(n_rows, df.height)
    if n_rows <= 0:
        raise ValueError("No rows available for SHAP background.")

    df_sample = df.sample(n=n_rows, seed=random_state)
    print(f"[INFO] Sampled {df_sample.height} background rows for SHAP.")
    return df_sample.to_numpy()


def compute_shap_single_patient(model, X_patient, background, feature_names):
    """
    Compute SHAP values for a single patient using TreeExplainer.
    Returns:
        shap_row: (n_features,) array
        expected_value: float
    """
    print("[INFO] Building TreeExplainer for XGBoost...")
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",  # log-odds
    )

    shap_vals = explainer.shap_values(X_patient)
    exp_val = explainer.expected_value

    # Handle list/array styles
    if isinstance(shap_vals, list):
        # binary classification: [class0, class1]
        shap_pos = np.array(shap_vals[1])
        if isinstance(exp_val, (list, np.ndarray)):
            expected_value = np.asarray(exp_val)[1]
        else:
            expected_value = exp_val
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3 and shap_vals.shape[2] == 2:
        # (n_samples, n_features, n_classes)
        shap_pos = shap_vals[:, :, 1]
        if isinstance(exp_val, (list, np.ndarray)):
            expected_value = np.asarray(exp_val)[1]
        else:
            expected_value = exp_val
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
        shap_pos = shap_vals
        expected_value = exp_val
    else:
        raise ValueError(
            f"Unexpected SHAP shape: {getattr(shap_vals, 'shape', None)}"
        )

    if isinstance(expected_value, np.ndarray):
        if expected_value.size != 1:
            raise ValueError(f"Expected scalar expected_value, got {expected_value.shape}")
        expected_value = expected_value.item()

    shap_row = shap_pos[0, :]  # (n_features,)
    print("[INFO] SHAP values computed for single patient.")
    return shap_row, float(expected_value)


def save_patient_shap_json(
    out_path, feature_names, shap_row, original_row_dict, base_value, risk_score
):
    abs_shap = np.abs(shap_row)
    order = np.argsort(-abs_shap)  # descending by |SHAP|

    entries = []
    for idx in order:
        feat = feature_names[idx]
        entry = {
            "feature": feat,
            "value": original_row_dict.get(feat, 0.0),
            "shap_value": float(shap_row[idx]),
            "abs_shap": float(abs_shap[idx]),
        }
        entries.append(entry)

    payload = {
        "base_value_logit": base_value,
        "predicted_risk": risk_score,
        "n_features": len(feature_names),
        "contributions_sorted": entries,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVE] Patient SHAP explanation → {out_path}")


def save_patient_bar_plot(
    out_png, feature_names, shap_row, top_k=15, title="XGBoost patient explanation"
):
    abs_shap = np.abs(shap_row)
    order = np.argsort(-abs_shap)[:top_k]
    feats_top = [feature_names[i] for i in order][::-1]  # reverse for nice plotting
    shap_top = shap_row[order][::-1]

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(feats_top))
    plt.barh(y_pos, shap_top)
    plt.yticks(y_pos, feats_top, fontsize=8)
    plt.xlabel("SHAP value (impact on log-odds of readmission)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SAVE] Patient SHAP bar plot → {out_png}")


# ==========================================================
# MAIN
# ==========================================================
def main():

    # --------------------------
    # 1) Hard-coded patient JSON
    # --------------------------
    INPUT_JSON = os.path.join(
        LLM_DIR,
        "patient_145121.json"     # <<< change this to patient you want to score
    )

    print(f"[INFO] Using hard-coded patient JSON: {INPUT_JSON}")

    # 2) Load resources
    feature_names = load_feature_list(FEATURE_LIST_PATH)
    row_dict = load_single_patient_json(INPUT_JSON)
    X_patient = build_feature_vector(row_dict, feature_names)

    print(f"[INFO] Loading XGBoost model from {XGB_MODEL_PATH}")
    model = joblib.load(XGB_MODEL_PATH)

    # 3) Predict risk
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_patient)[0, 1])
        print(f"[RESULT] Predicted 30-day readmission risk (probability): {proba:.4f}")
    else:
        scores = model.decision_function(X_patient)
        logit = float(scores[0])
        proba = 1.0 / (1.0 + np.exp(-logit))
        print(f"[RESULT] Predicted 30-day readmission risk: {proba:.4f}")

    # 4) SHAP explanation
    background = sample_background(
        BACKGROUND_PARQUET, feature_names, BACKGROUND_N,
        random_state=RANDOM_STATE
    )

    shap_row, base_value = compute_shap_single_patient(
        model, X_patient, background, feature_names
    )

    # 5) Save outputs
    hadm_raw = row_dict.get("HADM_ID", "unknown")

    # Clean up HADM_ID for filenames (avoid '145121.0')
    if isinstance(hadm_raw, float) and hadm_raw.is_integer():
        hadm_id = int(hadm_raw)
    else:
        hadm_id = hadm_raw

    base_name = f"patient_{hadm_id}"


    json_out = os.path.join(OUTPUT_DIR, f"{base_name}_shap_explanation.json")
    png_out  = os.path.join(OUTPUT_DIR, f"{base_name}_shap_bar_top15.png")

    save_patient_shap_json(
        json_out, feature_names, shap_row, row_dict, base_value, proba
    )
    save_patient_bar_plot(
        png_out, feature_names, shap_row,
        top_k=15, title=f"XGBoost — SHAP for HADM_ID {hadm_id}"
    )

    print("\n[DONE] Single patient scoring + SHAP completed.")


if __name__ == "__main__":
    main()

