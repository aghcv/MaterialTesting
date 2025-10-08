# mtlab — Material Testing Lab (arterial tissue)

Clean, modular toolkit for:
- Loading material test data
- Fitting constitutive models
- Summarizing parameters by group/region
- Computing strain energy metrics
- Verification/validation of constitutive models
- Publication-ready plots

## Install (editable)
```bash
pip install -e .
```

## Quickstart
```bash
# ---------------------------------------------------------------------
# 1️⃣  Ingest raw experimental data
# ---------------------------------------------------------------------
# Combine proximal / distal stress–stretch data and thickness sheet into one dataset.
# This command parses each Excel file, extracts group / rabbit info, 
# merges thickness, and saves a unified dataset in ./mt_out/data.pkl
mt ingest \
    --proximal inputs/proximal.xlsx \
    --distal inputs/distal.xlsx \
    --thickness inputs/Thickness_Normalized.xlsx \
    --out mt_out/data.pkl
# ---------------------------------------------------------------------
# 2️⃣  Compute strain energy (area under stress–stretch curves)
# ---------------------------------------------------------------------
# Integrates experimental stress–stretch data over default stretch ranges.
mt strain-energy
# ---------------------------------------------------------------------
# 3️⃣  Fit constitutive models
# ---------------------------------------------------------------------
# Fits all supported models (linear + hyperelastic) to each specimen.
# Uses predefined stretch ranges and saves results to ./model/.
mt fit
# ---------------------------------------------------------------------
# 4️⃣  Summarize results
# ---------------------------------------------------------------------
# Aggregates fitted parameters, computes group-level statistics,
# merges with strain energy metrics, and creates augmented summary tables.
mt summarize
# ---------------------------------------------------------------------
# 5️⃣  Verify model robustness
# ---------------------------------------------------------------------
# Performs k-fold verification (default 5-fold) across all models.
mt verify
# ---------------------------------------------------------------------
# 6️⃣  Analyze and visualize
# ---------------------------------------------------------------------
# Generates all plots, comparisons, and validation reports in one go.
mt analyze

```
