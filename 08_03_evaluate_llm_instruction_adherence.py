# 08_03_evaluate_llm_instruction_adherence.py
#
# Evaluates multiple LLM explanations for the SAME patient (or multiple patients)
# where filenames look like:
#
#   SHAP JSON:
#       patient_{HADM_ID}_shap_explanation.json
#   LLM outputs (many runs):
#       patient_{HADM_ID}_llm_explanation_*.txt
#
# For each (patient_id, run) pair it checks:
#   - word count in 150–200 range
#   - presence of risk label (low/moderate/high)
#   - correctness of risk label vs SHAP probability
#   - presence + correctness of probability (%)
#   - absence of forbidden terms
#
# Outputs:
#   - llm_instruction_adherence_summary.csv in LLM_DIR
#   - aggregate metrics printed to console

import os
import json
import re
import csv
from typing import Optional, Tuple

# ==========================
# CONFIG
# ==========================
FEAT_DIR  = r"H:\A\parquet files\features"
LLM_DIR  = os.path.join(FEAT_DIR, "LLM demo")

MIN_WORDS = 150
MAX_WORDS = 200

FORBIDDEN_TERMS = [
    "shap",
    "log-odds",
    "log odds",
    "treeexplainer",
    "kernelexplainer",
    "samplingexplainer",
]


# ==========================
# HELPERS
# ==========================

def classify_risk(prob: float) -> str:
    pct = prob * 100.0
    if pct < 60.0:
        return "low"
    elif pct < 80.0:
        return "moderate"
    else:
        return "high"


def extract_label_from_text(text: str) -> Optional[str]:
    lower = text.lower()
    matches = []
    for lab in ["low", "moderate", "high"]:
        m = re.search(rf"\b{lab}\b", lower)
        if m:
            matches.append((m.start(), lab))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[0][1]


def extract_prob_from_text(text: str) -> Optional[float]:
    m = re.search(r"(\d+(\.\d+)?)\s*%", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def count_words(text: str) -> int:
    tokens = re.findall(r"\w+", text)
    return len(tokens)


def contains_forbidden(text: str) -> Tuple[bool, list]:
    lower = text.lower()
    found = [term for term in FORBIDDEN_TERMS if term in lower]
    return (len(found) > 0, found)


# ==========================
# MAIN
# ==========================

def main():
    if not os.path.exists(LLM_DIR):
        raise FileNotFoundError(f"LLM_DIR does not exist: {LLM_DIR}")

    # Find all SHAP JSONs
    shap_files = [
        f for f in os.listdir(LLM_DIR)
        if f.startswith("patient_") and f.endswith("_shap_explanation.json")
    ]

    if not shap_files:
        print(f"[INFO] No SHAP explanation JSONs found in {LLM_DIR}.")
        return

    results = []

    for shap_fname in shap_files:
        # patient_{HADM}_shap_explanation.json
        base = shap_fname.replace("_shap_explanation.json", "")
        patient_id_str = base.replace("patient_", "")
        shap_path = os.path.join(LLM_DIR, shap_fname)

        # Load SHAP JSON for the true risk
        with open(shap_path, "r", encoding="utf-8") as f:
            shap_payload = json.load(f)

        risk_prob = shap_payload.get("predicted_risk", None)
        if risk_prob is None:
            print(f"[WARN] No 'predicted_risk' in {shap_fname} → skipping.")
            continue

        true_pct   = risk_prob * 100.0
        true_label = classify_risk(risk_prob)

        # Find ALL LLM outputs for this patient:
        # patient_{id}_llm_explanation_*.txt
        llm_txt_files = [
            f for f in os.listdir(LLM_DIR)
            if f.startswith(base + "_llm_explanation_") and f.endswith(".txt")
        ]

        if not llm_txt_files:
            print(f"[WARN] No LLM explanation txt files for {base} → skipping.")
            continue

        for txt_fname in llm_txt_files:
            llm_txt_path = os.path.join(LLM_DIR, txt_fname)

            # try to derive a "run id" from filename (everything after base_)
            run_id = txt_fname.replace(base + "_llm_explanation_", "")
            run_id = run_id.replace(".txt", "")

            with open(llm_txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            n_words = count_words(text)
            word_ok = (MIN_WORDS <= n_words <= MAX_WORDS)

            found_label = extract_label_from_text(text)
            label_present = found_label is not None
            label_correct = (label_present and (found_label == true_label))

            pct_in_text = extract_prob_from_text(text)
            prob_present = pct_in_text is not None
            prob_matches = (
                prob_present and abs(pct_in_text - true_pct) <= 1.0
            )

            has_forb, forb_list = contains_forbidden(text)

            results.append({
                "patient_id": patient_id_str,
                "run_id": run_id,
                "true_risk_prob": risk_prob,
                "true_risk_pct": true_pct,
                "true_label": true_label,
                "n_words": n_words,
                "word_count_ok_150_200": int(word_ok),
                "label_present": int(label_present),
                "found_label": found_label if found_label is not None else "",
                "label_correct": int(label_correct),
                "prob_present": int(prob_present),
                "prob_in_text_pct": pct_in_text if pct_in_text is not None else "",
                "prob_matches_model": int(prob_matches),
                "has_forbidden_terms": int(has_forb),
                "forbidden_terms": ";".join(forb_list),
                "file_name": txt_fname,
            })

    if not results:
        print("[INFO] No (SHAP + LLM) pairs evaluated.")
        return

    # Save CSV
    out_csv = os.path.join(LLM_DIR, "llm_instruction_adherence_summary.csv")
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[SAVE] Per-run adherence summary → {out_csv}")

    # Aggregate metrics
    n = len(results)

    def rate(key):
        return sum(r[key] for r in results) / n * 100.0

    print("\n[AGGREGATE INSTRUCTION ADHERENCE]")
    print(f"Total explanations evaluated (all runs): {n}")
    print(f"Word-count within 150–200:   {rate('word_count_ok_150_200'):.1f}%")
    print(f"Risk label present:          {rate('label_present'):.1f}%")
    print(f"Risk label correct:          {rate('label_correct'):.1f}%")
    print(f"Probability mentioned:       {rate('prob_present'):.1f}%")
    print(f"Probability matches model:   {rate('prob_matches_model'):.1f}%")
    print(f"No forbidden terms:          {100.0 - rate('has_forbidden_terms'):.1f}%")

    print("\n[DONE] Instruction adherence evaluation complete.")


if __name__ == "__main__":
    main()
