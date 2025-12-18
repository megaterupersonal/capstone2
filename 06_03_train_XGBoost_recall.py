# 06_03_train_XGBoost.py  (XGBoost with VALIDATION-selected F2 / recall-weighted threshold policy)
# - Custom split: Train/Valid/Test with test prevalence ≈ 5.52% positives (realistic base rate)
# - RandomizedSearchCV scored by average_precision (PR-AUC)
# - Operating policy: choose a probability threshold on VALIDATION that maximizes F2
#   (recall-weighted threshold selection, then apply same threshold to TEST)
# - Saves: model, metrics, predictions, feature importances, CV summary, best params, policy details

import os, pathlib, json
import numpy as np
import polars as pl
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, precision_recall_curve, classification_report,
)

from xgboost import XGBClassifier
import joblib

# ---------- Paths ----------
FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

X_PATH   = os.path.join(FEAT_DIR, "final_tree_features.parquet")
y_PATH   = os.path.join(FEAT_DIR, "final_tree_labels.csv")

OUT_DIR  = os.path.join(FEAT_DIR, "models_xgboost")
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ---- Output suffix to differentiate this operating policy from top-K workload scripts ----
SUFFIX = "_f2policy"   # changed to recall for the testing scenario. Is not better than top-k in terms of practicality but meets the requirement of high recall.

MODEL_PATH       = os.path.join(OUT_DIR, f"xgb_best_model{SUFFIX}.joblib")
METRICS_PATH     = os.path.join(OUT_DIR, f"xgb_test_metrics{SUFFIX}.json")
PRED_PATH        = os.path.join(OUT_DIR, f"xgb_test_predictions{SUFFIX}.csv")
FEATIMP_PATH     = os.path.join(OUT_DIR, f"xgb_feature_importances{SUFFIX}.csv")
CVRESULTS_PATH   = os.path.join(OUT_DIR, f"xgb_cv_results{SUFFIX}.csv")
BESTPARAMS_PATH  = os.path.join(OUT_DIR, f"xgb_best_params{SUFFIX}.json")
POLICY_PATH      = os.path.join(OUT_DIR, f"xgb_decision_policy{SUFFIX}.json")
PRVAL_PATH       = os.path.join(OUT_DIR, f"xgb_val_pr_curve{SUFFIX}.csv")  # diagnostic

LABEL_COL    = "LBL_READMIT_30D"
ID_COL       = "HADM_ID"
RANDOM_STATE = 67

POS_UPWEIGHT   = 2.0    # same as rf

# --------- Global split fractions (Train / Val / Test) ---------
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.20
TEST_FRAC  = 0.10

if not np.isclose(TRAIN_FRAC + VAL_FRAC + TEST_FRAC, 1.0):
    raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0")

# For custom test prevalence (cap on positive rate in TEST)
MAX_TEST_POS_FRAC = 0.0552    # ~ real readmission rate (5.52%)
rng = np.random.RandomState(RANDOM_STATE)

# ---------- Load ----------
print(f"[LOAD] X  ← {X_PATH}")
X = pl.read_parquet(X_PATH)

print(f"[LOAD] y  ← {y_PATH}")
y_tab = pl.read_csv(y_PATH)

if LABEL_COL not in y_tab.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found in {y_PATH}.")

hadm_ids = (
    y_tab[ID_COL].to_numpy()
    if ID_COL in y_tab.columns
    else np.arange(len(y_tab))
)
y = y_tab[LABEL_COL].to_numpy()

if X.height != len(y):
    raise ValueError(f"Row mismatch: X has {X.height}, y has {len(y)}.")

X_np = X.to_numpy()
feat_names = X.columns

# ---------- Custom 3-way split (Train / Val / Test) with capped test prevalence ----------
n_total = len(y)
all_idx = np.arange(n_total)

pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]

# 1) Build TEST first (respecting size + prevalence cap)
n_test_target = int(round(TEST_FRAC * n_total))

max_pos_test = int(round(MAX_TEST_POS_FRAC * n_test_target))
max_pos_test = min(max_pos_test, len(pos_idx))

n_neg_test = n_test_target - max_pos_test
n_neg_test = min(n_neg_test, len(neg_idx))

test_pos_idx = (
    rng.choice(pos_idx, size=max_pos_test, replace=False)
    if max_pos_test > 0 else np.array([], dtype=int)
)

remaining_neg_for_test = np.setdiff1d(neg_idx, test_pos_idx, assume_unique=True)
test_neg_idx = (
    rng.choice(remaining_neg_for_test, size=n_neg_test, replace=False)
    if n_neg_test > 0 else np.array([], dtype=int)
)

test_idx = np.concatenate([test_pos_idx, test_neg_idx])
rng.shuffle(test_idx)

X_test  = X_np[test_idx]
y_test  = y[test_idx]
id_test = hadm_ids[test_idx]

# 2) Split remaining into TRAIN / VAL with desired global fractions
remaining_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)

X_rem  = X_np[remaining_idx]
y_rem  = y[remaining_idx]
id_rem = hadm_ids[remaining_idx]

train_size_within_remaining = TRAIN_FRAC / (TRAIN_FRAC + VAL_FRAC)

X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
    X_rem,
    y_rem,
    id_rem,
    train_size=train_size_within_remaining,
    stratify=y_rem,
    random_state=RANDOM_STATE,
)

print(
    f"[CUSTOM SPLIT] Train: {X_train.shape[0]}  "
    f"Val: {X_val.shape[0]}  Test: {X_test.shape[0]}"
)
print(
    "[CUSTOM SPLIT] Pos rate (train/val/test): "
    f"{y_train.mean():.3f}/{y_val.mean():.3f}/{y_test.mean():.3f}"
)

# ---------- Compute scale_pos_weight for imbalance ----------
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
scale_pos_weight = n_neg / max(1, n_pos)
print(f"[IMB] scale_pos_weight (neg/pos) = {scale_pos_weight:.3f}")

# ---------- Base XGBoost model & search space ----------
xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
    random_state=RANDOM_STATE,
    scale_pos_weight=scale_pos_weight,
)

param_dist = {
    "n_estimators":       [200, 300, 400, 600],
    "learning_rate":      [0.03, 0.05, 0.1],
    "max_depth":          [3, 4, 5],
    "min_child_weight":   [1, 3, 5],
    "subsample":          [0.6, 0.8, 1.0],
    "colsample_bytree":   [0.6, 0.8, 1.0],
    "gamma":              [0.0, 0.1, 0.3],
    "reg_lambda":         [1.0, 5.0, 10.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Sample weights: slight additional upweight for positives (optional)
sample_w = np.ones_like(y_train, dtype=float)
sample_w[y_train == 1] *= POS_UPWEIGHT

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=40,
    scoring="average_precision",
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=RANDOM_STATE,
    refit=True,
)

print("[FIT] XGBoost RandomizedSearchCV (PR-AUC) starting…")
search.fit(X_train, y_train, sample_weight=sample_w)
best_xgb = search.best_estimator_
print(f"[FIT] Best params: {search.best_params_}")
print(f"[FIT] Best CV (PR-AUC): {search.best_score_:.4f}")

# Defensive final refit on Train with the same weights
best_xgb.fit(X_train, y_train, sample_weight=sample_w)

# ---------- Helper: threshold selection by max F2 on VALIDATION ----------
def select_threshold_max_fbeta(y_true: np.ndarray, y_score: np.ndarray, beta: float = 2.0):
    """
    Uses precision_recall_curve to evaluate thresholds and selects the threshold that maximizes F-beta.
    Returns: (best_threshold, best_precision, best_recall, best_fbeta, flag_rate)
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)

    # precision_recall_curve returns:
    # - prec/rec arrays of length (len(thr)+1)
    # - thr array of length len(thr)
    # Metrics at threshold thr[i] correspond to prec[i], rec[i]
    rows = []
    for pr, rc, th in zip(prec[:-1], rec[:-1], thr):
        fbeta = ((1.0 + beta**2) * pr * rc) / ((beta**2) * pr + rc + 1e-12)
        rows.append((float(th), float(pr), float(rc), float(fbeta)))

    if len(rows) == 0:
        # fallback: predict all zeros
        return 1.0, 0.0, 0.0, 0.0, 0.0

    # choose max F-beta; tie-breaker: higher recall, then higher precision
    rows.sort(key=lambda x: (x[3], x[2], x[1]), reverse=True)
    best_th, best_pr, best_rc, best_fb = rows[0]

    y_pred = (y_score >= best_th).astype(int)
    flag_rate = float(y_pred.mean())

    return best_th, best_pr, best_rc, best_fb, flag_rate

# ---------- Policy selection on VALIDATION (threshold maximizing F2) ----------
proba_val = best_xgb.predict_proba(X_val)[:, 1]
best_th, best_pr, best_rc, best_f2, best_flag_rate = select_threshold_max_fbeta(
    y_true=y_val, y_score=proba_val, beta=2.0
)

val_pred = (proba_val >= best_th).astype(int)

val_policy = {
    "policy": "threshold_max_f2_on_val",
    "beta": 2.0,
    "val_threshold": float(best_th),
    "val_precision": float(precision_score(y_val, val_pred, zero_division=0)),
    "val_recall": float(recall_score(y_val, val_pred, zero_division=0)),
    "val_f2": float(fbeta_score(y_val, val_pred, beta=2, zero_division=0)),
    "val_flag_rate": float(val_pred.mean()),
}

print(f"[POLICY] Selected policy (VAL): {val_policy}")
print(f"[VAL] Classification report @threshold={best_th:.6f} (max F2 on VAL):")
print(classification_report(y_val, val_pred, digits=3))

# Save PR curve diagnostics on VAL (threshold-based)
prec_curve, rec_curve, thr_curve = precision_recall_curve(y_val, proba_val)
rows = []
for pr, rc, th in zip(prec_curve[:-1], rec_curve[:-1], thr_curve):
    f2 = (5.0 * pr * rc) / (4.0 * pr + rc + 1e-12)
    rows.append(
        {
            "threshold": float(th),
            "precision": float(pr),
            "recall": float(rc),
            "f2": float(f2),
        }
    )
pd.DataFrame(rows).to_csv(PRVAL_PATH, index=False)

# ---------- Evaluate on TEST (apply SAME threshold chosen on VAL) ----------
proba_test = best_xgb.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= best_th).astype(int)

metrics = {
    "decision_policy": "threshold_max_f2_on_val",
    "beta": 2.0,
    "val_policy_stats": val_policy,
    "test_threshold_used": float(best_th),
    "roc_auc": float(roc_auc_score(y_test, proba_test)),
    "pr_auc": float(average_precision_score(y_test, proba_test)),
    "accuracy": float(accuracy_score(y_test, pred_test)),
    "precision": float(precision_score(y_test, pred_test, zero_division=0)),
    "recall": float(recall_score(y_test, pred_test, zero_division=0)),
    "f1": float(f1_score(y_test, pred_test, zero_division=0)),
    "f2": float(fbeta_score(y_test, pred_test, beta=2, zero_division=0)),
    "test_flag_rate": float(pred_test.mean()),
}

cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
metrics["confusion_matrix"] = {
    "tn": int(cm[0, 0]),
    "fp": int(cm[0, 1]),
    "fn": int(cm[1, 0]),
    "tp": int(cm[1, 1]),
}

print("[TEST] Metrics @threshold policy:", json.dumps(metrics, indent=2))

# ---------- Save artifacts ----------
joblib.dump(best_xgb, MODEL_PATH)
print(f"[SAVE] Model → {MODEL_PATH}")

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"[SAVE] Metrics → {METRICS_PATH}")

pred_df = pl.DataFrame(
    {
        ID_COL: id_test,
        "y_true": y_test,
        "y_pred": pred_test,
        "y_proba": proba_test,
    }
)
pred_df.write_csv(PRED_PATH)
print(f"[SAVE] Test predictions → {PRED_PATH}")

# XGBoost feature importance
importances = getattr(best_xgb, "feature_importances_", None)
if importances is not None:
    fi = (
        pl.DataFrame({"feature": feat_names, "importance": importances})
        .sort("importance", descending=True)
    )
    fi.write_csv(FEATIMP_PATH)
    print(f"[SAVE] Feature importances → {FEATIMP_PATH}")

pd.DataFrame(search.cv_results_).to_csv(CVRESULTS_PATH, index=False)
print(f"[SAVE] CV results → {CVRESULTS_PATH}")

with open(BESTPARAMS_PATH, "w", encoding="utf-8") as f:
    json.dump(search.best_params_, f, indent=2)
print(f"[SAVE] Best params → {BESTPARAMS_PATH}")

with open(POLICY_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "policy": "threshold_max_f2_on_val",
            "beta": 2.0,
            "val_threshold": float(best_th),
        },
        f,
        indent=2,
    )
print(f"[SAVE] Decision policy details → {POLICY_PATH}")

print("[DONE]")
