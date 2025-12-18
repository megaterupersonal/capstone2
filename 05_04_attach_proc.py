# 05_04_attach_proc.py  
# This file attaches procedure features to the main spine table.

import os, pathlib, re
import polars as pl

# ---------- Paths ----------
CLEAN_DIR = r"H:\A\parquet files\cleaned"
FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

PROC_PATH = os.path.join(CLEAN_DIR, "PROCEDURES_ICD_cleaned.parquet")
SPINE_IN  = os.path.join(FEAT_DIR, "05_03_attach_diagnosis.parquet")
OUT_PARQUET = os.path.join(FEAT_DIR, "05_04_attach_proc.parquet")
OUT_SUMMARY = os.path.join(FEAT_DIR, "05_04_attach_proc_summary.csv")

# ---------- Load ----------
proc  = pl.read_parquet(PROC_PATH)
spine = pl.read_parquet(SPINE_IN)

print(f"[LOAD] PROCEDURES_ICD_cleaned: {proc.height:,} rows, {proc.width} cols")
print(f"[LOAD] Spine in: {spine.height:,} rows, {spine.width} cols  ← {os.path.basename(SPINE_IN)}")

# ---------- Choose code column (NO duplicate creation) ----------
if "ICD9_NORM" in proc.columns:
    code_col_name = "ICD9_NORM"
elif "ICD9_CODE" in proc.columns:
    code_col_name = "ICD9_CODE"
    # If raw codes might contain dots/lowercase, normalize into a TEMP name to avoid collisions
    tmp_name = "_ICD9_TMP_"
    if tmp_name in proc.columns:
        proc = proc.drop(tmp_name)
    proc = proc.with_columns(
        pl.col("ICD9_CODE").cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9]", "", literal=False)
        .alias(tmp_name)
    )
    code_col_name = tmp_name
else:
    raise ValueError("PROCEDURES_ICD must have ICD9_NORM or ICD9_CODE")

# Ensure titles exist as strings (no renaming)
if "SHORT_TITLE" not in proc.columns: proc = proc.with_columns(pl.lit(None, dtype=pl.Utf8).alias("SHORT_TITLE"))
if "LONG_TITLE"  not in proc.columns: proc = proc.with_columns(pl.lit(None, dtype=pl.Utf8).alias("LONG_TITLE"))

# ---------- ICD-9 predicate helpers ----------
def rx(prefixes: list[str]) -> str:
    # anchors at start; accepts family prefixes like "885" or full codes like "0331"
    return r"^(?:" + "|".join(map(re.escape, prefixes)) + ")"

def contains_any(col: pl.Expr, words: list[str]) -> pl.Expr:
    if not words:
        return pl.lit(False)
    pat = "(?i)" + "|".join(re.escape(w) for w in words)
    return col.cast(pl.Utf8).str.contains(pat, literal=False).fill_null(False)

title_any = lambda words: (contains_any(pl.col("SHORT_TITLE"), words) | contains_any(pl.col("LONG_TITLE"), words))
code = pl.col(code_col_name)

FLAG_SPECS = [  #This was built through MIMIC-III documentation research, other references, and trial-and-error. This works in tandem with 05_04a_validation.py
    # =========================
    # CORE ICU SUPPORT
    # =========================
    ("PRC_MECH_VENT_row",           ["967","9604"],                   ["mech","ventilat","endotracheal"]),
    ("PRC_INTUBATION_row",          ["9605","9607"],                  ["intubat"]),
    ("PRC_TRACHEOSTOMY_row",        ["311"],                          ["tracheost"]),
    ("PRC_ENTERAL_NUTRITION_row",   ["966"],                          ["enteral","tube feed","nutrit"]),
    ("PRC_TRANSFUSION_row",         ["990","9903","9904"],            ["transfusion","packed"]),
    ("PRC_DIALYSIS_row",            ["3995","5498"],                  ["dialysis","hemodialysis","peritoneal"]),
    ("PRC_CENTRAL_LINE_row",        ["3893"],                         ["central","venous cath"]),
    ("PRC_ART_LINE_row",            ["3891"],                         ["arterial catheter","art line"]),

    # Hemodynamic monitoring (unified) 
    ("PRC_HEMO_MONITOR_row",        ["8960","8961","8963","8964","8968","0069"], 
                                    ["pulmonary artery","swan","swan-ganz","wedge","cardiac output","intravascular pressure"]),

    # =========================
    # CARDIAC SURGERY / DEVICES / RESUSCITATION
    # =========================
    ("PRC_CABG_row",                ["361"],                          ["CABG","bypass"]),
    ("PRC_VALVE_REPLACE_row",       ["3521","3522","3523","3524"],    ["valve replacement","aortic valve","mitral valve"]),
    ("PRC_EXTRACORP_CIRC_row",      ["3961"],                         ["extracorporeal circulation","cardiopulmonary bypass","bypass pump"]),
    ("PRC_CARDIOPLEGIA_HYPOTH_row", ["3962","3963"],                  ["cardioplegia","hypothermia"]),
    ("PRC_BALLOON_PUMP_row",        ["3761","3762"],                  ["pulsation balloon","IABP","intraaortic balloon"]),
    ("PRC_HEART_ASSIST_row",        ["3768","9744","9749"],           ["heart assist","assist system","device removal"]),
    ("PRC_CARDIAC_DEVICE_row",      ["3772","3778","3783"],           ["pacemaker","defibrillator","CRT","AICD","temporary pacemaker"]),
    ("PRC_RESUSCITATION_row",       ["9960","9961","9962","3791"],    ["CPR","cardioversion","defibrillation","countershock","open chest massage"]),

    # =========================
    # CORONARY / CARDIAC CATH & ANGIO / EP
    # =========================
    ("PRC_PCI_row",                 ["0066","3606","3607"],           ["PCI","PTCA","stent"]),
    ("PRC_CORONARY_OTHER_row",      ["3601","3605","3699"],           ["coronary","heart vessel","revascular"]),
    ("PRC_CORONARY_ANGIO_row",      ["8853","8854","8855","8856","8857"], 
                                    ["coronary arteriog","angiocardiogram"]),
    ("PRC_CARDIAC_CATH_row",        ["3721","3722","3723","3727","3728"], 
                                    ["cardiac cath","right heart cath","left heart cath","intracardiac echo","cardiac mapping"]),
    ("PRC_PERICARDIAL_PROC_row",    ["370","3712"],                    ["pericardiocentesis","pericardiotomy"]),

    # =========================
    # VASCULAR (NON-CORONARY) & MAJOR VESSEL SURGERY
    # =========================
    ("PRC_VASC_ENDO_OPEN_row",      ["3950","3951","3979","3844","3845","3846"], 
                                    ["angioplasty","aneurysm repair","endovascular","vessel replacement","aortic graft","thoracic vessel"]),
    ("PRC_VASC_ACCESS_REPAIR_row",  ["3807","3808","3931","3949","3886"], 
                                    ["incision of vessel","suture of artery","revision of vascular","occlusion of vessel"]),
    ("PRC_VASC_GEN_row",            ["0040","0041","0042","0043","0044"], 
                                    ["procedure on vessel","bifurcation","vessel operation"]),
    ("PRC_VENA_CAVA_row",           ["387","391"],                    ["vena cava","venous shunt"]),
    ("PRC_UMBIL_VEIN_CATH_row",     ["3892"],                         ["umbilical vein","venous catheter"]),

    # Diagnostic vascular studies
    ("PRC_VASC_DIAGNOSTIC_row",     ["8841","8842","8845","8847","8848","8864","8867","8872","8873","8875","8897","8826"], 
                                    ["arteriogram","angiogram","phlebography","venogram","ultrasound","echo","MRI","CT"]),

    # =========================
    # THORAX / PLEURA / AIRWAY ENDOSCOPY & REPAIR
    # =========================
    ("PRC_CHEST_TUBE_row",          ["3404"],                         ["intercostal catheter","chest tube"]),
    ("PRC_THORACENTESIS_row",       ["3491"],                         ["thoracentesis"]),
    ("PRC_PLEURAL_THORAX_OP_row",   ["3403","3409","3421","3451","3472","3479"], 
                                    ["pleural incision","thoracoscopy","decortication","chest wall repair"]),
    ("PRC_TRACHEAL_ENDO_REP_row",   ["3141","3142","3179","3199"],    ["laryngoscopy","tracheoscopy","tracheal repair"]),
    ("PRC_BRONCHOSCOPY_row",        ["3321","3322","3323","3324"],    ["bronchoscopy"]),

    # =========================
    # NEURO (EVD / SHUNTS / CRANIAL / MENINGES / MONITORING)
    # =========================
    ("PRC_EVD_SHUNT_row",
                                    ["022","0221","0234","0239","0243"],
                                    ["external ventricular drain","ventricular shunt","EVD","VP shunt"]),
    ("PRC_MENINGES_CRANIAL_row",    ["0124","0131","0139","0212","016","0205","0206"], 
                                    ["craniotomy","cerebral meninges","skull","plate","osteoplasty"]),
    ("PRC_NEURO_IOM_row",           ["0094"],                         ["intra-operative neurophysiologic monitoring"]),
    ("PRC_SPINAL_TAP_row",          ["0331"],                         ["spinal tap","lumbar puncture"]),

    # =========================
    # GI SURGERY / ENDOSCOPY / HEMOSTASIS / STOMA / ANASTOMOSIS
    # =========================
    ("PRC_ENDO_GI_row",             ["4513","4516","4523","4524","4413","4223","4292","5185","5188"], 
                                    ["endoscopy","egd","biopsy","colonoscopy","sigmoidoscopy","gastroscopy","esophagoscopy","esophageal dilation","ercp","sphincterotomy"]),
    ("PRC_ENDO_GI_HEMOSTASIS_row",  ["4443","4444"],                  ["endoscopic control","hemostasis","bleeding"]),
    ("PRC_STOMA_CREATE_CLOSE_row",
                                    ["4601","4603","4611","4620","4621","4623","4651","4652","4639"],
                                    ["ileostomy","colostomy","exteriorization","permanent","temporary","stoma closure","enterostomy"]),

    ("PRC_BOWEL_ANAST_REPAIR_row",  ["4591","4593","4594","4595","4673","4674","4675","4679","4681"], 
                                    ["intestinal anastomosis","repair intestine","suture intestine","bowel manipulation","fistula"]),
    ("PRC_LAPAROTOMY_row",          ["5411","5412","5419"],           ["laparotomy","reopen laparotomy"]),
    ("PRC_GI_MAJOR_row",            ["4561","4562","4573","4574","4701","5122","5102","5103","503","504","5091"], 
                                    ["resection","colectomy","hemicolectomy","appendectomy","cholecystectomy","cholecystostomy","liver","hepatectomy","liver aspiration"]),
    ("PRC_GB_ASP_row",              ["5101"],                         ["gallbladder aspiration","percutaneous gb"]),
    ("PRC_ABD_DRAIN_row",           ["5491"],                         ["abdominal drainage","paracentesis"]),
    ("PRC_PEG_GJ_row",              ["4311","4432","4438","4632","9703"], 
                                    ["gastrostomy","peg","gastrojejunostomy","jejunostomy","gastroenterostomy","tube replacement"]),
    ("PRC_TRANSPLANT_row",
                                    ["0091","0092","0093","5059","5569"],
                                    ["transplant","donor","cadaver","liver transplant","kidney transplant"]),


    # =========================
    # UROLOGY / GU ACCESS
    # =========================
    ("PRC_URO_ACCESS_row",          ["598","5794","5795","5994"],     ["ureteral catheter","urinary catheter","cystostomy tube replacement"]),
    ("PRC_CYSTOSCOPY_row",          ["5732"],                         ["cystoscopy"]),
    ("PRC_NEPRHOSTOMY_row",         ["5503","5593"],                  ["nephrostomy"]),
    ("PRC_ILEOURETER_row",          ["5651"],                         ["ileoureterostomy"]),
    ("PRC_CIRCUMCISION_row", ["640"], ["circumcision"]),


    # =========================
    # ORTHO / JOINTS / BONE / DEVICES / AMPUTATION
    # =========================
    ("PRC_JOINT_REPLACE_row",       ["8151","8152","8154"],           ["hip replacement","knee replacement"]),
    ("PRC_ARTHROCENTESIS_row",      ["8191"],                         ["arthrocentesis","joint aspiration"]),
    ("PRC_SPINE_FUSION_row",        ["8103","8105","8106","8162","8163","8164"], 
                                    ["spinal fusion","lumbar","cervical"]),
    ("PRC_SPINE_DISC_row",          ["8051"],                         ["intervertebral disc"]),
    ("PRC_FX_FIX_DEVICE_row",       ["7815","7817","7818","7857","7867","7869","7932","7935"], 
                                    ["external fixator","fixation","implanted device","removal device","fracture reduction"]),
    ("PRC_BONE_EXC_GRAFT_row",      ["7761","7767","7768","7771","7779"], 
                                    ["bone graft","excision bone","rib","sternum","tibia","fibula"]),
    ("PRC_SOFT_GRAFT_row",          ["8382"],                         ["muscle graft","fascia graft","soft tissue graft"]),
    ("PRC_AMPUTATION_row",          ["8411","8412","8415"],           ["amputation","toe","below knee"]),

    # =========================
    # WOUND / SKIN / RECONSTRUCTIVE
    # =========================
    ("PRC_WOUND_CARE_row",          ["8622","8628","8659","8604","8605","8674","8669"], 
                                    ["debridement","wound","skin closure","incision and drainage","pedicle graft","flap graft","skin graft"]),
    ("PRC_HERNIA_REPAIR_row",       ["5310","5349","5359","5369","5472"], 
                                    ["hernia repair","umbilical","abdominal wall"]),
    ("PRC_ABD_REPAIR_row",          ["5461","5475"],                  ["abdominal repair","mesenteric repair","postoperative disruption"]),

    # =========================
    # RESPIRATORY / GI TUBES & LAVAGE
    # =========================
    ("PRC_RESP_LAVAGE_row",         ["9656","9659"],                  ["lavage","bronchial lavage"]),
    ("PRC_GASTRIC_LAVAGE_row",      ["9633","9635"],                  ["gastric lavage","gavage"]),
    ("PRC_INTEST_RECTAL_TUBE_row",  ["9608","9609"],                  ["intestinal tube","rectal tube"]),

    # =========================
    # THERAPIES / INFUSIONS / MEDICATION ADMIN
    # =========================
    ("PRC_INFUSION_row",            ["0013","0014","9925","9929","9910"], 
                                    ["infusion","injection","chemotherapy","therapeutic substance","thrombolytic"]),
    ("PRC_VASOPRESSOR_row",         ["0017"],                         ["vasopressor"]),
    ("PRC_PLASMA_LEUKA_PHOTO_row",  ["9971","9972","9974","9983"],    ["plasmapheresis","leukapheresis","phototherapy"]),
    ("PRC_VACCINE_row",             ["9955"],                         ["vaccine","vaccination","immunization"]),
    ("PRC_INHALED_NO_row",          ["0012"],                         ["inhaled nitric oxide"]),
    ("PRC_OXY_NEB_row",             ["9394","9396"],                  ["nebulizer","oxygen enrichment"]),
    ("PRC_RADIOTHERAPY_row",        ["9229"],                         ["radiotherapy","radiotherapeutic"]),
    ("PRC_COUNSELING_row",          ["9468","9543"],                  ["counseling","audiological evaluation","detox"]),

    # =========================
    # FOREIGN BODY / ENT MINORS
    # =========================
    ("PRC_FB_AIRWAY_row",           ["9815"],                         ["foreign body trachea","bronchus"]),
    ("PRC_FB_ESOPH_row",            ["9802"],                         ["foreign body esophagus"]),
    ("PRC_EPISTAXIS_row",           ["2103"],                         ["epistaxis","cauterization"]),
]


# ---------- Build row-level flags programmatically ----------
row_flag_exprs = []
for flag_name, prefixes, kws in FLAG_SPECS:
    pat = rx(prefixes) if prefixes else None
    expr = (code.str.contains(pat, literal=False) if pat else pl.lit(False)) | title_any(kws)
    row_flag_exprs.append(expr.alias(flag_name))

proc_flags = proc.with_columns(row_flag_exprs)

# ---------- Aggregate to admission level ----------
flag_cols = [name for name, _, _ in FLAG_SPECS]
agg_exprs = [
    pl.len().alias("PRC_N_ROWS"),
    code.n_unique().alias("PRC_N_UNIQUE"),
]
agg_exprs += [pl.col(c).any().cast(pl.Int8).alias(c.replace("_row","")) for c in flag_cols]

agg = (
    proc_flags.group_by("HADM_ID")
    .agg(agg_exprs)
)

print(f"[AGG] Procedures aggregated to admissions: {agg.height:,} rows")

# ---------- Collapse many flags into compact macro groups ----------
# These are admission-level column names (no "_row" suffix)
child_flags = [c.replace("_row", "") for c, _, _ in FLAG_SPECS]

GROUP_DEFS: dict[str, list[str]] = { #Aggregation group definitions
    # ICU / Airway / Monitoring / Life support
    "ICU_SUPPORT": [
        "PRC_MECH_VENT", "PRC_INTUBATION", "PRC_TRACHEOSTOMY",
        "PRC_ENTERAL_NUTRITION", "PRC_DIALYSIS",
        "PRC_CENTRAL_LINE", "PRC_ART_LINE", "PRC_HEMO_MONITOR",
        "PRC_TRANSFUSION"
    ],

    # Cardiac surgery / devices / resuscitation
    "CARDIAC_PROC": [
        "PRC_CABG", "PRC_VALVE_REPLACE", "PRC_EXTRACORP_CIRC",
        "PRC_CARDIOPLEGIA_HYPOTH", "PRC_BALLOON_PUMP",
        "PRC_HEART_ASSIST", "PRC_CARDIAC_DEVICE", "PRC_RESUSCITATION"
    ],

    # Coronary / cath / angiography / EP / pericardial
    "CORONARY_CATH_EP": [
        "PRC_PCI", "PRC_CORONARY_OTHER", "PRC_CORONARY_ANGIO",
        "PRC_CARDIAC_CATH", "PRC_PERICARDIAL_PROC"
    ],

    # Vascular (non-coronary) including access/repair & diagnostics
    "VASCULAR": [
        "PRC_VASC_ENDO_OPEN", "PRC_VASC_ACCESS_REPAIR",
        "PRC_VASC_GEN", "PRC_VENA_CAVA", "PRC_UMBIL_VEIN_CATH",
        "PRC_VASC_DIAGNOSTIC"
    ],

    # Thorax / pleura / airway endoscopy & repair
    "THORAX_RESP": [
        "PRC_CHEST_TUBE", "PRC_THORACENTESIS", "PRC_PLEURAL_THORAX_OP",
        "PRC_TRACHEAL_ENDO_REP", "PRC_BRONCHOSCOPY"
    ],

    # Neuro (EVD/shunts/cranium/LP/IOM)
    "NEURO": [
        "PRC_EVD_SHUNT", "PRC_MENINGES_CRANIAL", "PRC_NEURO_IOM",
        "PRC_SPINAL_TAP"
    ],

    # GI endoscopy / hemostasis / stoma / anastomosis / major GI / drains / PEG-GJ / transplant
    "GI_GENERAL": [
        "PRC_ENDO_GI", "PRC_ENDO_GI_HEMOSTASIS", "PRC_STOMA_CREATE_CLOSE",
        "PRC_BOWEL_ANAST_REPAIR", "PRC_LAPAROTOMY", "PRC_GI_MAJOR",
        "PRC_GB_ASP", "PRC_ABD_DRAIN", "PRC_PEG_GJ", "PRC_TRANSPLANT"
    ],

    # GU / urology
    "GU_URO": [
        "PRC_URO_ACCESS", "PRC_CYSTOSCOPY", "PRC_NEPRHOSTOMY",
        "PRC_ILEOURETER", "PRC_CIRCUMCISION"
    ],

    # Ortho / joints / spine / bone / devices / amputation
    "ORTHO": [
        "PRC_JOINT_REPLACE", "PRC_ARTHROCENTESIS", "PRC_SPINE_FUSION",
        "PRC_SPINE_DISC", "PRC_FX_FIX_DEVICE", "PRC_BONE_EXC_GRAFT",
        "PRC_SOFT_GRAFT", "PRC_AMPUTATION"
    ],

    # Wound / skin / reconstructive / abdominal wall
    "WOUND_SKIN": [
        "PRC_WOUND_CARE", "PRC_HERNIA_REPAIR", "PRC_ABD_REPAIR"
    ],

    # Tubes & lavage (respiratory / gastric / intestinal)
    "TUBES_LAVAGE": [
        "PRC_RESP_LAVAGE", "PRC_GASTRIC_LAVAGE", "PRC_INTEST_RECTAL_TUBE"
    ],

    # Therapies / infusions / admin / radiation / counseling
    "THERAPIES": [
        "PRC_INFUSION", "PRC_VASOPRESSOR", "PRC_PLASMA_LEUKA_PHOTO",
        "PRC_VACCINE", "PRC_INHALED_NO", "PRC_OXY_NEB",
        "PRC_RADIOTHERAPY", "PRC_COUNSELING"
    ],

    # Foreign body / ENT (minor)
    "FB_ENT": [
        "PRC_FB_AIRWAY", "PRC_FB_ESOPH", "PRC_EPISTAXIS"
    ],
}

# Sanity: only keep children that actually exist in agg
GROUP_DEFS = {g: [c for c in cols if c in agg.columns] for g, cols in GROUP_DEFS.items()}

# Build group indicators and counts
group_exprs = []
for gname, cols in GROUP_DEFS.items():
    if not cols:
        continue
    any_col = pl.max_horizontal([pl.col(c).cast(pl.Int8) for c in cols]).alias(f"PRC_{gname}_ANY")
    sum_col = pl.sum_horizontal([pl.col(c).cast(pl.Int8) for c in cols]).alias(f"PRC_{gname}_N")
    group_exprs += [any_col, sum_col]

agg2 = agg.with_columns(group_exprs)

# Optional: also an overall "any procedure hit" and total count of child flags
overall_any = pl.max_horizontal([pl.col(c).cast(pl.Int8) for c in child_flags]).alias("PRC_ANY")
overall_n   = pl.sum_horizontal([pl.col(c).cast(pl.Int8) for c in child_flags]).alias("PRC_TOTAL_CHILD_HITS")

agg2 = agg2.with_columns([overall_any, overall_n])


# ---------- Join to spine and fill ----------
PRC_FILL_COLS = [
    c for c in agg2.columns
    if c.startswith("PRC_") and c not in ("PRC_N_ROWS", "PRC_N_UNIQUE")
]

spine1 = (
    spine.join(agg2, on="HADM_ID", how="left")
    .with_columns(
        pl.col("PRC_N_ROWS").fill_null(0),
        pl.col("PRC_N_UNIQUE").fill_null(0),
        *[
            pl.coalesce([pl.col(c), pl.lit(0)])  # fill nulls with 0
              .cast(pl.Int16)                    # safer than Int8 for counts
              .alias(c)                          # <-- IMPORTANT: keep the name
            for c in PRC_FILL_COLS
        ]
    )
)


# ---------- One-row summary ----------
def one_row_summary(df: pl.DataFrame, flag_cols_row: list[str]) -> pl.DataFrame:
    def _item(expr: pl.Expr):
        out = df.select(expr)
        return None if out.height == 0 or out.width == 0 else out.item()

    # convert row-flag names ("..._row") to admission-level names (without suffix)
    flags_out = [c.replace("_row", "") for c in flag_cols_row]

    rec = {
        "n_rows": df.height,
        "mean_prc_rows": _item(pl.col("PRC_N_ROWS").mean().round(3)),
        "mean_prc_unique": _item(pl.col("PRC_N_UNIQUE").mean().round(3)),
    }
    for f in flags_out:
        # proportion of admissions with the procedure flag
        if f in df.columns:
            rec[f"pct_{f.lower()}"] = _item(pl.col(f).mean().round(4))

    return pl.DataFrame([rec])

summary = one_row_summary(spine1, flag_cols)

# ---------- Write outputs ----------
summary.write_csv(OUT_SUMMARY)
print(f"[REPORT] Wrote summary → {OUT_SUMMARY}")

spine1.write_parquet(OUT_PARQUET, compression="zstd")
print(f"[SAVE] Wrote → {OUT_PARQUET}")

print("[DONE]")