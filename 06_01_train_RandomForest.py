# 06_01_train_RandomForest.py  (Random Forest with exact top-20% workload policy)
# - Custom split: Train/Valid/Test with test prevalence ≈ 5.52% positives (realistic base rate)
# - RandomizedSearchCV scored by average_precision (PR-AUC)
# - Operating policy: flag EXACTLY the top 20% highest-risk patients on each set
#   (rank-based top-K, not probability threshold)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# ---------- Paths ----------
FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

X_PATH   = os.path.join(FEAT_DIR, "final_tree_features.parquet")
y_PATH   = os.path.join(FEAT_DIR, "final_tree_labels.csv")

OUT_DIR  = os.path.join(FEAT_DIR, "models_random_forest")
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

MODEL_PATH       = os.path.join(OUT_DIR, "rf_best_model.joblib")
METRICS_PATH     = os.path.join(OUT_DIR, "rf_test_metrics.json")
PRED_PATH        = os.path.join(OUT_DIR, "rf_test_predictions.csv")
FEATIMP_PATH     = os.path.join(OUT_DIR, "rf_feature_importances.csv")
CVRESULTS_PATH   = os.path.join(OUT_DIR, "rf_cv_results.csv")
BESTPARAMS_PATH  = os.path.join(OUT_DIR, "rf_best_params.json")
THRESHOLD_PATH   = os.path.join(OUT_DIR, "rf_decision_policy.json")
PRVAL_PATH       = os.path.join(OUT_DIR, "rf_val_pr_curve.csv")  # optional diagnostic

LABEL_COL    = "LBL_READMIT_30D"
ID_COL       = "HADM_ID"
RANDOM_STATE = 67

# Workload policy: flag EXACTLY top 20% highest-risk patients
WORKLOAD_FRAC = 0.20   # 20%

POS_UPWEIGHT = 2.0    # increase weight of positive class during training

# --------- Global split fractions (Train / Val / Test) ---------
# Example: 70% train, 20% validation, 10% test
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

# upper bound on positives in TEST
max_pos_test = int(round(MAX_TEST_POS_FRAC * n_test_target))
max_pos_test = min(max_pos_test, len(pos_idx))

# remaining test slots are negatives
n_neg_test = n_test_target - max_pos_test
n_neg_test = min(n_neg_test, len(neg_idx))

# sample test positives / negatives (if any)
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

# build TEST arrays
X_test  = X_np[test_idx]
y_test  = y[test_idx]
id_test = hadm_ids[test_idx]

# 2) Split remaining into TRAIN / VAL with desired global fractions
remaining_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)

X_rem  = X_np[remaining_idx]
y_rem  = y[remaining_idx]
id_rem = hadm_ids[remaining_idx]

# within the "non-test" pool, what fraction should be TRAIN?
# ensures global ≈ TRAIN_FRAC / VAL_FRAC
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



# ---------- Model & Search Space ----------
rf = RandomForestClassifier(
    class_weight="balanced",    # base balancing
    n_jobs=-1,
    random_state=RANDOM_STATE,
    bootstrap=True,
)

param_dist = {
    "n_estimators":      [500, 700, 900],
    "max_depth":         [None, 12, 20, 30],
    "min_samples_split": [2, 3, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 0.4, 0.6],
    "max_samples":       [None, 0.85, 0.95],  # sub-sampling for robustness
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Sample weights (increase cost of FN; applied within CV and refit)
base_w = compute_sample_weight(class_weight="balanced", y=y_train)
base_w[y_train == 1] *= POS_UPWEIGHT

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=35,                    # RF is heavy, so not as many iterations as DT
    scoring="average_precision",  # PR-AUC
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=RANDOM_STATE,
    refit=True,
)

print("[FIT] RandomizedSearchCV (PR-AUC) starting…")
search.fit(X_train, y_train, sample_weight=base_w)
best_rf = search.best_estimator_
print(f"[FIT] Best params: {search.best_params_}")
print(f"[FIT] Best CV (PR-AUC): {search.best_score_:.4f}")

# Defensive final refit on the full Train_in
best_rf.fit(X_train, y_train, sample_weight=base_w)

# ---------- Helper: rank-based top-K predictions ----------
def topk_pred(y_score: np.ndarray, frac: float):
    """
    Returns binary predictions where exactly top frac of samples (by score, descending)
    are labeled 1, and the rest 0. Also returns the score at the K-th position.
    """
    n = len(y_score)
    k = max(1, int(round(frac * n)))
    order = np.argsort(y_score)[::-1]  # descending
    topk_idx = order[:k]
    y_pred = np.zeros(n, dtype=int)
    y_pred[topk_idx] = 1
    kth_score = float(y_score[order[k - 1]])
    return y_pred, kth_score, k

# ---------- Policy selection on VALIDATION (rank-based top-20%) ----------
proba_val = best_rf.predict_proba(X_val)[:, 1]
val_pred, val_kth_score, val_k = topk_pred(proba_val, WORKLOAD_FRAC)

val_policy = {
    "policy": "workload_top20pct_rank",
    "workload_frac": float(WORKLOAD_FRAC),
    "val_K": int(val_k),
    "val_threshold_kth_score": float(val_kth_score),
    "val_precision": float(precision_score(y_val, val_pred, zero_division=0)),
    "val_recall": float(recall_score(y_val, val_pred, zero_division=0)),
    "val_f2": float(fbeta_score(y_val, val_pred, beta=2, zero_division=0)),
    "val_flag_rate": float(val_pred.mean()),
}

print(f"[POLICY] Selected policy (VAL): {val_policy}")
print("[VAL] Classification report @top-20% workload:")
print(classification_report(y_val, val_pred, digits=3))

# Save PR curve diagnostics (independent of top-K policy)
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

# ---------- Evaluate on TEST (same rank-based top-20% policy) ----------
proba_test = best_rf.predict_proba(X_test)[:, 1]
pred_test, test_kth_score, test_k = topk_pred(proba_test, WORKLOAD_FRAC)

metrics = {
    "decision_policy": "workload_top20pct_rank",
    "workload_frac": float(WORKLOAD_FRAC),
    "val_policy_stats": val_policy,
    "test_K": int(test_k),
    "test_threshold_kth_score": float(test_kth_score),
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

print("[TEST] Metrics @top-20% workload:", json.dumps(metrics, indent=2))

# ---------- Save artifacts ----------
joblib.dump(best_rf, MODEL_PATH)
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

importances = getattr(best_rf, "feature_importances_", None)
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

with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "workload_frac": float(WORKLOAD_FRAC),
            "val_K": int(val_k),
            "val_threshold_kth_score": float(val_kth_score),
            "test_K": int(test_k),
            "test_threshold_kth_score": float(test_kth_score),
        },
        f,
        indent=2,
    )
print(f"[SAVE] Decision policy details → {THRESHOLD_PATH}")

print("[DONE]")
