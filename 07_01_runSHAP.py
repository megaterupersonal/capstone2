# 07_01_run_SHAP.py
# Global SHAP explanations for all 7 models
# - Uses *separate* feature sets per model group (tree vs linear-like)
# - Rebuilds the same custom Train/Test split used in 06_0x scripts
#   using a shared label vector (must align row-wise across feature sets)
# - Uses model-appropriate SHAP explainers:
#     * TreeExplainer    → RF, DT, XGBoost
#     * LinearExplainer  → Logistic Regression
#     * KernelExplainer  → SVM (RBF), KNN, Naive Bayes
# - Samples a subset of the TEST set for SHAP to keep runtime manageable
# - Saves:
#     * SHAP values as .npy
#     * summary plots as .png
#     * per-model metadata as .json (including feature names)

import os
import pathlib
import warnings
import json

import numpy as np
import polars as pl
import joblib
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# 0. PATHS & GLOBAL SETTINGS
# ============================================================

FEAT_DIR  = r"H:\A\parquet files\features"

# --- Feature sets per group ---
# Adjust these filenames if your tree-based features are saved under different names.
TREE_X_PATH   = os.path.join(FEAT_DIR, "final_tree_features.parquet")
TREE_y_PATH   = os.path.join(FEAT_DIR, "final_tree_labels.csv")

LINEAR_X_PATH = os.path.join(FEAT_DIR, "final_linear_features.parquet")
LINEAR_y_PATH = os.path.join(FEAT_DIR, "final_linear_labels.csv")

SHAP_DIR = os.path.join(FEAT_DIR, "shap_outputs")
pathlib.Path(SHAP_DIR).mkdir(parents=True, exist_ok=True)

LABEL_COL = "LBL_READMIT_30D"
ID_COL    = "HADM_ID"

# ---------- Split settings (must match 06_0x scripts) ----------
RANDOM_STATE      = 67
TEST_FRACTION     = 0.20
MAX_TEST_POS_FRAC = 0.0552
rng = np.random.RandomState(RANDOM_STATE)

# SHAP sampling settings
MAX_TEST_SAMPLES          = 200  # max rows from test set to explain per model
BACKGROUND_SIZE_LINEAR    = 75   # background size for LinearExplainer
BACKGROUND_SIZE_NONLINEAR = 75   # background size for KernelExplainer (SVM / KNN / NB)
NSAMPLES_SAMPLING         = 100  # KernelExplainer nsamples

# ============================================================
# 1. FEATURE GROUP CONFIGURATION
# ============================================================

# Feature groups describe which X / y each model family uses.
# "feat_group" is about the *feature space* (tree vs linear-like),
# while "model_type" is about the SHAP explainer type.
FEAT_GROUPS = {
    "tree": {
        "X_path": TREE_X_PATH,
        "y_path": TREE_y_PATH,
    },
    "linear": {
        "X_path": LINEAR_X_PATH,
        "y_path": LINEAR_y_PATH,
    },
}

# Each entry: (key, pretty_name, model_type, feat_group, model_path)
# model_type ∈ {"tree", "linear", "sampling"}
MODELS = [
    ("rf",  "Random Forest",       "tree",     "tree",
     os.path.join(FEAT_DIR, "models_random_forest", "rf_best_model.joblib")),
    ("dt",  "Decision Tree",       "tree",     "tree",
     os.path.join(FEAT_DIR, "models_decision_tree", "dt_best_model.joblib")),
    ("xgb", "XGBoost",             "tree",     "tree",
     os.path.join(FEAT_DIR, "models_xgboost", "xgb_best_model.joblib")),

    ("lr",  "Logistic Regression", "linear",   "linear",
     os.path.join(FEAT_DIR, "models_logistic", "lr_best_model.joblib")),
    ("svm", "SVM (RBF)",           "sampling", "linear",
     os.path.join(FEAT_DIR, "models_svm", "svm_best_model.joblib")),
    ("knn", "KNN",                 "sampling", "linear",
     os.path.join(FEAT_DIR, "models_knn", "knn_best_model.joblib")),
    ("nb",  "Naive Bayes",         "sampling", "linear",
     os.path.join(FEAT_DIR, "models_nb", "nb_best_model.joblib")),
]

# ============================================================
# 2. UTILS
# ============================================================

def sample_background(X_source: np.ndarray, n: int, desc: str) -> np.ndarray:
    """Sample background rows from a source matrix for SHAP."""
    n = min(n, X_source.shape[0])
    if n <= 0:
        raise ValueError(f"No rows available for {desc} background.")
    idx = rng.choice(np.arange(X_source.shape[0]), size=n, replace=False)
    print(f"[SHAP] Background ({desc}): {n} rows")
    return X_source[idx]


def compute_shap_for_model(
    model,
    model_type: str,
    model_key: str,
    X_shap: np.ndarray,
    background_linear: np.ndarray,
    background_nonlinear: np.ndarray,
):
    """
    Compute SHAP values for a given model using the appropriate explainer.

    Returns
    -------
    shap_values: np.ndarray of shape (n_samples, n_features)
        SHAP values for the positive class (or scalar output).
    expected_value: float
        Expected value (baseline) used by SHAP.
    """
    if model_type == "tree":
        print(f"[EXPLAIN] {model_key}: TreeExplainer")

        explainer = shap.TreeExplainer(
            model,
            data=background_linear,           # background in this model's feature space
            feature_perturbation="interventional",
            model_output="raw",               # log-odds for stability
        )

        shap_vals = explainer.shap_values(X_shap)
        exp_val = explainer.expected_value

        # Case 1: classic SHAP → list per class
        if isinstance(shap_vals, list):
            # positive class = index 1
            shap_pos = np.array(shap_vals[1])
            if isinstance(exp_val, (list, np.ndarray)):
                expected_value = np.asarray(exp_val)[1]
            else:
                expected_value = exp_val

        # Case 2: new SHAP style → single array (n_samples, n_features, n_classes)
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3 and shap_vals.shape[2] == 2:
            shap_pos = shap_vals[:, :, 1]
            if isinstance(exp_val, (list, np.ndarray)):
                expected_value = np.asarray(exp_val)[1]
            else:
                expected_value = exp_val

        # Case 3: already 2D (rare, but just in case)
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            shap_pos = shap_vals
            expected_value = exp_val

        else:
            raise ValueError(
                f"Unexpected tree SHAP shape for {model_key}: "
                f"{getattr(shap_vals, 'shape', None)}"
            )

        # ensure scalar float
        if isinstance(expected_value, np.ndarray):
            if expected_value.size != 1:
                raise ValueError(
                    f"Expected a single expected_value for {model_key}, "
                    f"got shape {expected_value.shape}"
                )
            expected_value = expected_value.item()

        return shap_pos, float(expected_value)

    elif model_type == "linear":
        print(f"[EXPLAIN] {model_key}: LinearExplainer")

        explainer = shap.LinearExplainer(
            model,
            background_linear,
            feature_perturbation="interventional",
        )
        shap_pos = explainer.shap_values(X_shap)
        expected_value = explainer.expected_value

        # For binary, expected_value may be array-like
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.asarray(expected_value)
            if expected_value.size > 1:
                expected_value = expected_value[-1]
            else:
                expected_value = expected_value.item()

        return np.array(shap_pos), float(expected_value)

    elif model_type == "sampling":
        print(f"[EXPLAIN] {model_key}: KernelExplainer (for non-linear, non-tree model)")

        # ----- define robust model output function -----
        if hasattr(model, "predict_proba"):
            print("         → using predict_proba as model output")

            def f_output(X_batch):
                X_batch = np.array(X_batch)
                if X_batch.ndim == 1:
                    X_batch = X_batch.reshape(1, -1)
                if X_batch.shape[0] == 0:
                    return np.zeros((0,), dtype=float)
                out = model.predict_proba(X_batch)[:, 1]
                return np.asarray(out, dtype=float)

        elif hasattr(model, "decision_function"):
            print("         → predict_proba not available; using decision_function as model output")

            def f_output(X_batch):
                X_batch = np.array(X_batch)
                if X_batch.ndim == 1:
                    X_batch = X_batch.reshape(1, -1)
                if X_batch.shape[0] == 0:
                    return np.zeros((0,), dtype=float)
                scores = model.decision_function(X_batch)
                scores = np.asarray(scores, dtype=float)
                if scores.ndim == 2 and scores.shape[1] == 1:
                    scores = scores.ravel()
                return scores
        else:
            raise AttributeError(
                f"Model '{model_key}' has neither predict_proba nor decision_function."
            )

        explainer = shap.KernelExplainer(
            f_output,
            background_nonlinear,
        )

        shap_vals = explainer.shap_values(
            X_shap,
            nsamples=NSAMPLES_SAMPLING,
        )

        shap_pos = np.array(shap_vals, dtype=float)
        expected_value = np.asarray(explainer.expected_value, dtype=float).ravel()[0]

        # final safety: replace any NaNs/Infs with 0 so plots don't break
        if not np.all(np.isfinite(shap_pos)):
            print(f"[WARN] Non-finite SHAP values for {model_key}; replacing NaNs/Infs with 0.")
            shap_pos = np.nan_to_num(shap_pos, nan=0.0, posinf=0.0, neginf=0.0)

        return shap_pos, float(expected_value)

    else:
        raise ValueError(f"Unknown model_type '{model_type}' for {model_key}")


# ============================================================
# 3. LOAD FEATURE GROUPS & BUILD CONSISTENT SPLIT
# ============================================================

print("[LOAD] Loading feature groups...")

datasets = {}          # per feat_group: X_np, y, feature_names, etc.
master_y = None        # reference label vector (for common split)
master_group = None

for group_name, paths in FEAT_GROUPS.items():
    X_path = paths["X_path"]
    y_path = paths["y_path"]

    print(f"\n[GROUP] {group_name}")
    print(f"[LOAD] X ← {X_path}")
    X_pl = pl.read_parquet(X_path)

    print(f"[LOAD] y ← {y_path}")
    y_tab = pl.read_csv(y_path)

    if LABEL_COL not in y_tab.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in {y_path}.")

    y_arr = y_tab[LABEL_COL].to_numpy()

    if X_pl.height != len(y_arr):
        raise ValueError(
            f"Row mismatch in group '{group_name}': X has {X_pl.height}, y has {len(y_arr)}."
        )

    X_np = X_pl.to_numpy()
    feature_names = list(X_pl.columns)

    datasets[group_name] = {
        "X_np": X_np,
        "y": y_arr,
        "feature_names": feature_names,
    }

    if master_y is None:
        master_y = y_arr
        master_group = group_name
    else:
        if len(y_arr) != len(master_y):
            raise ValueError(
                f"Label length mismatch between groups '{master_group}' and '{group_name}'."
            )
        if not np.array_equal(y_arr, master_y):
            raise ValueError(
                f"Label vectors differ between groups '{master_group}' and '{group_name}'. "
                f"All feature sets must align row-wise for a common split."
            )

n_total = len(master_y)
all_idx = np.arange(n_total)

print(f"\n[INFO] Total rows (common across feature groups): {n_total}")

# ---------- Rebuild Train/Test split (same as modelling scripts) ----------
pos_idx = np.where(master_y == 1)[0]
neg_idx = np.where(master_y == 0)[0]

n_test       = int(round(TEST_FRACTION * n_total))
max_pos_test = int(round(MAX_TEST_POS_FRAC * n_test))
max_pos_test = min(max_pos_test, len(pos_idx))
n_neg_test   = n_test - max_pos_test
n_neg_test   = min(n_neg_test, len(neg_idx))

test_pos_idx = rng.choice(pos_idx, size=max_pos_test, replace=False)
remaining_neg_idx = np.setdiff1d(neg_idx, test_pos_idx, assume_unique=True)
test_neg_idx = rng.choice(remaining_neg_idx, size=n_neg_test, replace=False)

test_idx = np.concatenate([test_pos_idx, test_neg_idx])
rng.shuffle(test_idx)
train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)

print(f"[SPLIT] Requested test size: {n_test}")
print(f"[SPLIT] Actual test size:    {len(test_idx)}")
print(f"[SPLIT] Train size:          {len(train_idx)}")

# Choose a common test subset for SHAP for all models (indices over master_y)
if len(test_idx) > MAX_TEST_SAMPLES:
    shap_test_idx = rng.choice(test_idx, size=MAX_TEST_SAMPLES, replace=False)
else:
    shap_test_idx = test_idx.copy()

print(f"[SHAP] Using {len(shap_test_idx)} test rows for SHAP explanations.")

# ---------- Build per-group splits + backgrounds ----------
for group_name, ds in datasets.items():
    X_np = ds["X_np"]
    y_arr = ds["y"]

    X_train_full = X_np[train_idx]
    y_train_full = y_arr[train_idx]
    X_test_full  = X_np[test_idx]
    y_test_full  = y_arr[test_idx]

    X_shap = X_np[shap_test_idx]
    y_shap = y_arr[shap_test_idx]

    print(f"\n[GROUP SPLIT] {group_name}")
    print(f"  Train: {X_train_full.shape[0]} rows")
    print(f"  Test:  {X_test_full.shape[0]} rows")
    print(f"  Pos rate (train/test): {y_train_full.mean():.3f} / {y_test_full.mean():.3f}")

    # Background subsets for SHAP (per feature group)
    background_linear    = sample_background(X_train_full, BACKGROUND_SIZE_LINEAR,
                                             f"{group_name}-linear")
    background_nonlinear = sample_background(X_train_full, BACKGROUND_SIZE_NONLINEAR,
                                             f"{group_name}-nonlinear")

    ds.update({
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_test_full":  X_test_full,
        "y_test_full":  y_test_full,
        "X_shap":       X_shap,
        "y_shap":       y_shap,
        "background_linear":    background_linear,
        "background_nonlinear": background_nonlinear,
    })

print("\n[INFO] Per-group splits and backgrounds prepared.")

# ============================================================
# 4. MAIN SHAP LOOP
# ============================================================

for key, pretty_name, model_type, feat_group, mpath in MODELS:
    if not os.path.exists(mpath):
        print(f"\n[SKIP] {pretty_name} ({key}) → model file not found: {mpath}")
        continue

    if feat_group not in datasets:
        print(f"\n[SKIP] {pretty_name} ({key}) → unknown feat_group '{feat_group}'")
        continue

    ds = datasets[feat_group]
    X_shap = ds["X_shap"]
    background_linear    = ds["background_linear"]
    background_nonlinear = ds["background_nonlinear"]
    feature_names        = ds["feature_names"]

    print("\n==============================")
    print(f"[MODEL] {pretty_name} ({key})  | feat_group = {feat_group}")
    print(f"[LOAD]  Model ← {mpath}")
    model = joblib.load(mpath)

    try:
        shap_values, expected_value = compute_shap_for_model(
            model=model,
            model_type=model_type,
            model_key=key,
            X_shap=X_shap,
            background_linear=background_linear,
            background_nonlinear=background_nonlinear,
        )
        print(f"[SHAP] Computed SHAP values: shape {shap_values.shape}")
    except Exception as e:
        print(f"[ERROR] SHAP failed for {pretty_name} ({key}): {e}")
        continue

    # ---------- Save raw SHAP values + metadata ----------
    shap_out = os.path.join(SHAP_DIR, f"{key}_shap_values_test.npy")
    meta_out = os.path.join(SHAP_DIR, f"{key}_shap_meta.json")

    np.save(shap_out, shap_values)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": pretty_name,
                "model_key": key,
                "model_type": model_type,
                "feat_group": feat_group,
                "expected_value": float(expected_value),
                "n_samples": int(shap_values.shape[0]),
                "n_features": int(shap_values.shape[1]),
                "feature_names": feature_names,
            },
            f,
            indent=2,
        )
    print(f"[SAVE] SHAP values → {shap_out}")
    print(f"[SAVE] SHAP meta   → {meta_out}")

    # ---------- Summary plot (beeswarm, top 20 features) ----------
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_shap,
            feature_names=feature_names,
            show=False,
            max_display=20,
            plot_size=None,  # let matplotlib figsize control it
        )
        plt.title(f"{pretty_name} — SHAP summary (test subset, top 20 features)")

        # Give more room on the left for long feature names
        plt.gcf().subplots_adjust(left=0.35)

        png_out = os.path.join(SHAP_DIR, f"{key}_shap_summary_top20.png")
        plt.savefig(png_out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVE] SHAP summary plot → {png_out}")
    except Exception as e:
        print(f"[WARN] Could not create summary plot for {pretty_name}: {e}")

print("\n[DONE] SHAP explanations generated for available models.")
