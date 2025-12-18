# 05_03a_elix_from_icd9.R
# This file computes Elixhauser comorbidity flags from ICD-9 diagnosis codes.
# This is an R file.
# For this to work in IDEs like VSC, ensure you have an R extension installed like R Tools that require a local R system installed and
# directories configured. You can also run this in RStudio directly.

library(arrow)
library(dplyr)
library(comorbidity)

in_parquet  <- "H:/A/parquet files/cleaned/DIAGNOSES_ICD_cleaned.parquet"
out_parquet <- "H:/A/parquet files/features/elix_flags.parquet"

# Expect columns: HADM_ID, ICD9_CODE
dx <- read_parquet(in_parquet) |>
  mutate(
    ICD9_CODE = gsub("[^A-Za-z0-9]", "", toupper(ICD9_CODE))  # strip dots/spaces
  ) |>
  distinct(HADM_ID, ICD9_CODE)

# Elixhauser flags (Quan ICD-9 mapping)
elix <- comorbidity(
  x       = dx,
  id      = "HADM_ID",
  code    = "ICD9_CODE",
  map     = "elixhauser_icd9_quan",
  assign0 = TRUE
)

# Try to compute Van Walraven score if available in your version
if ("score" %in% getNamespaceExports("comorbidity")) {
  elix$ELIX_VW_SCORE <- score(elix, weights = "vw")
} else {
  message("comorbidity::score() not available; flags only (no VW score).")
}

# Basic sanity print
cat("Rows:", nrow(elix), "Cols:", ncol(elix), "\n")
print(head(elix, 3))

write_parquet(elix, out_parquet)
cat("[OK] Wrote:", out_parquet, "\n")
