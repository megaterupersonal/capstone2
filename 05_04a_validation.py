# 05_04_proc_flag_coverage_check.py
# Simple coverage audit using UPDATED FLAG_SPECS:
# - % of PROCEDURES_ICD_cleaned rows not caught by ANY ICD-9 prefix or title phrase.

import polars as pl
import os, re

CLEAN_DIR = r"H:\A\parquet files\cleaned"
FEAT_DIR  = r"H:\A\parquet files\features"
os.makedirs(FEAT_DIR, exist_ok=True)

PROC_PATH = os.path.join(CLEAN_DIR, "PROCEDURES_ICD_cleaned.parquet")
OUT_UNMATCHED = os.path.join(FEAT_DIR, "05_04_proc_unmatched_sample.csv")

# ---------- Load ----------
proc = pl.read_parquet(PROC_PATH)
print(f"[LOAD] PROCEDURES_ICD_cleaned: {proc.height:,} rows, {proc.width} cols")

# ---------- Normalization (match your 05_04_attach_proc.py logic) ----------
if "ICD9_NORM" in proc.columns:
    code = pl.col("ICD9_NORM").cast(pl.Utf8)
elif "ICD9_CODE" in proc.columns:
    code = (
        pl.col("ICD9_CODE")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9]", "", literal=False)
    )
else:
    raise ValueError("No ICD9_NORM or ICD9_CODE column found.")

if "SHORT_TITLE" not in proc.columns:
    proc = proc.with_columns(pl.lit("").alias("SHORT_TITLE"))
if "LONG_TITLE" not in proc.columns:
    proc = proc.with_columns(pl.lit("").alias("LONG_TITLE"))

def rx(prefixes: list[str]) -> str:
    return r"^(?:" + "|".join(map(re.escape, prefixes)) + ")"

def title_any(words: list[str]) -> pl.Expr:
    patt = "(?i)" + "|".join(map(re.escape, words))
    return pl.any_horizontal(
        pl.col("SHORT_TITLE").cast(pl.Utf8).str.contains(patt, literal=False),
        pl.col("LONG_TITLE").cast(pl.Utf8).str.contains(patt, literal=False),
    ).fill_null(False)

# ---------- UPDATED FLAG_SPECS (update this too!) ----------
FLAG_SPECS = [
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


# ---------- Apply flags ----------
flags = []
for name, prefixes, kws in FLAG_SPECS:
    pat = rx(prefixes)
    expr = (code.str.contains(pat, literal=False) | title_any(kws)).alias(name)
    flags.append(expr)

proc_f = proc.with_columns(flags)

# ---------- Unmatched ----------
flag_cols = [f.meta.output_name() for f in flags]
proc_f = proc_f.with_columns(
    pl.sum_horizontal(*[pl.col(c).cast(pl.Int8) for c in flag_cols]).alias("FLAG_SUM")
)

total_rows = proc_f.height
unmatched = proc_f.filter(pl.col("FLAG_SUM") == 0)
n_unmatched = unmatched.height
pct_unmatched = 0 if total_rows == 0 else round(100 * n_unmatched / total_rows, 2)

n_adm_total = proc_f.select(pl.col("HADM_ID").n_unique()).item()
n_adm_unmatched = unmatched.select(pl.col("HADM_ID").n_unique()).item()
pct_adm_unmatched = 0 if n_adm_total == 0 else round(100 * n_adm_unmatched / n_adm_total, 2)

print("\n--- Procedure flag coverage check (UPDATED) ---")
print(f"Total procedure rows:       {total_rows:,}")
print(f"Unmatched rows (no flag):   {n_unmatched:,}  ({pct_unmatched}%)")
print(f"Distinct HADM_ID with any unmatched rows: {n_adm_unmatched:,} / {n_adm_total:,} ({pct_adm_unmatched}%)")

# ---------- Top unmatched 3-digit groups ----------
diag = (
    unmatched
    .with_columns(
        code.alias("ICD9_NORM"),
        pl.col("ICD9_CODE").cast(pl.Utf8).alias("ICD9_RAW"),
        pl.col("ICD9_CODE").cast(pl.Utf8).str.slice(0, 3).alias("ICD9_3DIG"),
    )
    .group_by("ICD9_3DIG")
    .agg(pl.len().alias("n_rows"))
    .sort("n_rows", descending=True)
    .head(15)
)
print("\nTop unmatched ICD9 3-digit groups:")
print(diag)

# ---------- Sample for manual inspection ----------
unmatched.select("HADM_ID","ICD9_CODE","SHORT_TITLE","LONG_TITLE").head(500).write_csv(OUT_UNMATCHED)
print(f"[WROTE] Sample of unmatched rows → {OUT_UNMATCHED}")
print("[DONE]")
