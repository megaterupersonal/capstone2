# 08_02_llm_explain_patient.py
# Generate LLM-based explanations for a single patient's readmission risk

import os
import json
from openai import OpenAI
from datetime import datetime

# =====================================================
# CONFIG 
# =====================================================
FEAT_DIR  = r"H:\A\parquet files\features"
LLM_DIR  = r"H:\A\parquet files\features\llm demo"

PATIENT_HADM_ID = 145121        # CHANGE THIS PATIENT ID FOR DIFFERENT CASES
MODEL_NAME      = "XGBoost"     # For wording in the prompt only
OPENAI_MODEL    = "gpt-5-mini"
TOP_K_FEATURES  = 15            # Number of SHAP contributors to show the LLM

# =====================================================
# FEATURE DICTIONARY (EXTERNAL JSON)
# =====================================================
DICT_PATH = os.path.join(LLM_DIR, "feature_dictionary.json")


def load_feature_dictionary(path: str) -> dict:
    """Load human-readable descriptions for feature names from JSON."""
    if not os.path.exists(path):
        print(f"[WARN] No feature dictionary found at {path}. Using raw feature names.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # Optionally ignore meta keys if you added e.g. "_version"
    meta_keys = {k for k in d.keys() if k.startswith("_")}
    for mk in meta_keys:
        d.pop(mk, None)

    print(f"[INFO] Loaded feature dictionary with {len(d)} entries.")
    return d


FEATURE_DICTIONARY = load_feature_dictionary(DICT_PATH)


def humanize_feature_name(raw_name: str) -> str:
    """
    Turn 'DX_CHF' into something nicer by:
    - looking it up in FEATURE_DICTIONARY if available
    - otherwise doing a basic underscore→space + title-case fallback.
    """
    if raw_name in FEATURE_DICTIONARY:
        return FEATURE_DICTIONARY[raw_name]

    # Fallback: simple prettification if not in dictionary
    return raw_name.replace("_", " ").title()


# =====================================================
# OPENAI CLIENT
# =====================================================
KEY_PATH = os.path.join(LLM_DIR, "openai_key.txt") #You need a valid openai key here, saved in THIS file.

with open(KEY_PATH, "r") as f:
    key = f.read().strip()

client = OpenAI(api_key=key)
# =====================================================
# LOAD SHAP JSON FOR THIS PATIENT
# =====================================================
# SHAP file naming convention from 08_01_score_single_patient.py:
#   patient_{HADM_ID}_shap_explanation.json
patient_id_int = int(PATIENT_HADM_ID)

shap_path = os.path.join(
    LLM_DIR,
    f"patient_{patient_id_int}_shap_explanation.json"
)

if not os.path.exists(shap_path):
    raise FileNotFoundError(
        f"SHAP file not found: {shap_path}\n"
        f"Make sure you ran 08_01_score_single_patient.py first for HADM_ID={patient_id_int}."
    )

print(f"[LOAD] SHAP explanation ← {shap_path}")
with open(shap_path, "r", encoding="utf-8") as f:
    shap_payload = json.load(f)

risk_prob  = shap_payload.get("predicted_risk", None)
base_value = shap_payload.get("base_value_logit", None)
contribs   = shap_payload.get("contributions_sorted", [])

if risk_prob is None or base_value is None or not contribs:
    raise ValueError("SHAP JSON missing required keys (predicted_risk / base_value_logit / contributions_sorted).")

top_contribs = contribs[:TOP_K_FEATURES]

# =====================================================
# BUILD TEXT SUMMARY OF TOP FEATURES
# =====================================================
feature_lines = []
for item in top_contribs:
    feat_code = item["feature"]
    val       = item["value"]
    sv        = item["shap_value"]

    direction = "increases" if sv > 0 else "decreases"
    nice_name = humanize_feature_name(feat_code)

    # Show both raw code + human name if available
    if feat_code in FEATURE_DICTIONARY:
        name_str = f"{feat_code} ({nice_name})"
    else:
        name_str = nice_name  # fallback

    line = (
        f"- {name_str}: value={val} | SHAP={sv:.4f} "
        f"({direction} readmission risk)"
    )
    feature_lines.append(line)

features_block = "\n".join(feature_lines)
risk_percent = risk_prob * 100.0

# =====================================================
# BUILD PROMPT FOR THE LLM
# =====================================================
user_prompt = f"""
You are helping a hospital doctor understand a 30-day readmission risk score
from a machine-learning model ({MODEL_NAME}) for ICU patients.

Patient ID: {patient_id_int}
Model-predicted probability of 30-day readmission: {risk_percent:.1f}% 

The model explanation identified the following top contributors
(sorted by absolute importance):

{features_block}

Instructions:
1. Classify the risk level as low (less than 60), moderate (60-80), or high (80+) based on the percentage.
2. Explain in clear clinical language why this patient is at that risk level,
   referring to the factors above.
3. Highlight which factors increased the model's estimated readmission risk and which factors reduced it.
4. Suggest 2–3 reasonable follow-up actions or care-planning considerations
   (e.g., discharge planning, follow-up visits, monitoring), but DO NOT give direct treatment orders, merely suggestions.

Important:
- Do NOT mention SHAP, log-odds, or internal model details.
- Speak as if you are summarizing for a busy clinician.
- Keep it around 150–200 words.
- Be conservative and cautious in your recommendations.
- Do not claim a value is clinically normal or abnormal unless the input explicitly gives that context.
  Instead use phrasing like: “this factor increased the model’s estimated readmission risk”.
- When referring to factors, explicitly mention the patient’s value
  (e.g., “the patient’s albumin was 2.8 g/dL”, not just “low albumin”).
- Describe which factors reduced the model’s estimated risk rather than calling them ‘protective’ in a clinical sense.
- Avoid overconfident language; use neutral phrases like “is associated with” rather than “will cause”.
- For features with mapping found in dictionary that are impactful like blood urine colour, mention the true meaning (e.g. 0 means YELLOW, 1 means STRAW, based on the dictionary provided).
- Do not mention any feature related to ADM_YEAR, as it is not properly documented due to privacy reasons.
"""

system_prompt = (
    "You are a careful, conservative clinical decision-support assistant. "
    "You explain model-generated risk scores to clinicians in plain language, "
    "without giving treatment orders or overriding clinical judgment."
)

# =====================================================
# CALL OPENAI
# =====================================================
print(f"[CALL] OpenAI model = {OPENAI_MODEL}")
response = client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ],
)

llm_text = response.choices[0].message.content.strip()

print("\n[LLM OUTPUT]\n")
print(llm_text)

# =====================================================
# SAVE OUTPUT
# =====================================================


# Build a run tag – timestamp-based so every run is unique
run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

out_txt = os.path.join(
    LLM_DIR,
    f"patient_{int(PATIENT_HADM_ID)}_llm_explanation_{run_tag}.txt"
)

with open(out_txt, "w", encoding="utf-8") as f:
    f.write(llm_text)

print(f"\n[SAVE] LLM explanation → {out_txt}")

