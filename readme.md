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

## Output Overview

mt_out/
├── data/
│ ├── specimens/ # Individual specimen stress–stretch CSVs
│ │ ├── CoA5_R01_distal.csv
│ │ └── ...
│ ├── strain-energy/ # Integrated strain energy metrics
│ │ └── strain_energy_stats.csv
│ └── specimens_master.csv # Unified dataset with metadata and arrays
│
├── model/
│ ├── fits/ # Core model-fitting outputs (per model / range)
│ │ ├── fits.csv
│ │ ├── constitutive_coupling_reliability.csv
│ │ ├── group_params_stats.csv
│ │ ├── goodness_of_fit_<region><range>.csv
│ │ ├── se_effects_boot<region><range>.csv
│ │ ├── se_param_coupling_boot<region><range><model>.csv
│ │ └── specimen_param_long.csv
│ │
│ ├── master_augmented.csv # Merged summary of model fits + strain energy + metadata
│ └── model_performance.csv # Mean ± SD metrics across models
│
├── plots/ # Publication-ready figures
│ ├── stress_stretch_summary_.png
│ ├── strain_energy_.png
│ └── model_params_*.png
│
└── verify/ # Optional model verification (if run)
└── crossval_metrics.csv



---

### 🧾 Folder Descriptions

| Folder / File | Description |
|----------------|-------------|
| **`data/specimens/`** |
| Raw experimental stress–stretch data (`Stretch`, `Stress`) per specimen × region. |
| **`data/specimens_master.csv`** |
| Master dataset with specimen metadata, group labels, rabbit ID, and serialized stress–stretch arrays. |
| **`data/strain-energy/strain_energy_stats.csv`** |
| Per-specimen strain energy integrated across stretch ranges (`default`, `physiological`, `diastolic`, `systolic`). |
| **`model/fits/fits.csv`** |
| Core table of per-specimen model fits (`Params`, `RMSE`, `NRMSE`, `R2`). |
| **`model/fits/group_params_stats.csv`** | 
| Aggregated statistics (mean, SD, median) for each parameter across groups and regions. |
| **`model/fits/goodness_of_fit_<region>_<range>.csv`** |
| Average model performance per region × stretch range (RMSE, NRMSE, R²). |
| **`model/fits/constitutive_coupling_reliability.csv`** |
| Bootstrap-derived reliability of parameter coupling and sign consistency across fits. |
| **`model/fits/se_effects_boot_<region>_<range>.csv`** ||
 Bootstrapped regression of strain energy vs. experimental variables (e.g., BW, duration). |
| **`model/fits/se_param_coupling_boot_<region>_<range>_<model>.csv`** |
| Bootstrap coupling analysis between model parameters and derived terms. |
| **`model/fits/specimen_param_long.csv`** |
| Long-format parameter table (`SpecimenID`, `Parameter`, `Value`) for visualization. |
| **`model/master_augmented.csv`** |
| Comprehensive merged dataset combining model fits, strain energy, and specimen metadata (used for cross-feature analysis). |
| **`model/model_performance.csv`** |
| Mean ± SD summary of error metrics (RMSE, NRMSE) across all models. |
| **`plots/`** |
| Group-averaged stress–stretch plots with fits, strain-energy boxplots, and parameter comparisons. |
| **`verify/`** |
| Cross-validation or robustness results (if verification is executed). |

---

| Stage          | Command           | Output                                                      |
|----------------|-------------------|-------------------------------------------------------------|
| Ingest         | `mt ingest`       | `data/specimens/`, `data/specimens_master.csv`              |
| Strain energy  | `mt strain-energy`| `data/strain-energy/strain_energy_stats.csv`                |
| Fit models     | `mt fit`          | `model/fits/` (individual fit files + coupling analyses)    |
| Summarize      | `mt summarize`    | `model/master_augmented.csv`, `model/model_performance.csv` |
| Verify         | `mt verify`       | `verify/`                                                   |
| Analyze        | `mt analyze`      | `plots/`                                                    |
