import pandas as pd
from pathlib import Path
from ..models.core import MODEL_REGISTRY
from ..data.schema import PARAM_COLUMNS, MODEL_PARAM_MAP

def expand_param_column(fits_df: pd.DataFrame, params_col: str="Params") -> pd.DataFrame:
    expanded = fits_df[params_col].apply(lambda d: pd.Series(d))
    out = pd.concat([fits_df.drop(columns=[params_col]), expanded], axis=1)
    return out

def specimen_level_param_table(fits_df: pd.DataFrame) -> pd.DataFrame:
    tmp = fits_df.copy()
    if "Params" in tmp.columns:
        tmp = expand_param_column(tmp, "Params")

    id_cols = [c for c in ["SpecimenID","Region","GroupName","Model"] if c in tmp.columns]

    rows = []
    for model, meta in MODEL_REGISTRY.items():
        param_list = meta["params"]
        sub = tmp[tmp["Model"] == model]
        if sub.empty:
            continue
        long = sub.melt(
            id_vars=id_cols,
            value_vars=[p for p in param_list if p in sub.columns],
            var_name="Parameter",
            value_name="Value"
        )
        rows.append(long)

    return pd.concat(rows, ignore_index=True)

def per_group_stats(long_param_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Region","GroupName","Model","Parameter"]
    return (long_param_df
            .groupby(group_cols, dropna=False)["Value"]
            .agg(["count","mean","std","median","min","max"])
            .reset_index())

def build_master_table(strain_df, fits_df, metrics_df):
    rows = []

    for _, fit in fits_df.iterrows():
        specimen = fit["SpecimenID"]
        model = fit["Model"]
        params = fit["Params"]

        # map params into p1…p6
        param_values = [None] * len(PARAM_COLUMNS)
        for idx, name in enumerate(MODEL_PARAM_MAP[model]):
            param_values[idx] = params.get(name, None)

        # get metrics
        metric_rows = metrics_df[
            (metrics_df["SpecimenID"] == specimen) &
            (metrics_df["Model"] == model)
        ]
        nrmse = metric_rows["NRMSE"].median() if not metric_rows.empty else None

        # get strain energies
        strain_rows = strain_df[strain_df["SpecimenID"] == specimen]
        for _, se in strain_rows.iterrows():
            rows.append({
                "SpecimenID": specimen,
                "GroupName": se["GroupName"],
                "Region": se["Region"],
                "Range": se["Range"],          # default, physio, diastolic, systolic
                "Model": model,
                "NRMSE": nrmse,
                "StrainEnergy": se["StrainEnergy"],
                **dict(zip(PARAM_COLUMNS, param_values)),
            })

    return pd.DataFrame(rows)

def enrich_master_with_specimens(
    master,                 # pd.DataFrame or str/Path to master_augmented.csv
    specimens,              # pd.DataFrame or str/Path to specimens_master.csv
    *,
    on="SpecimenID",
    drop_array_cols=True,   # drop bulky arrays from specimens
    array_cols=("Stretch", "Stress"),
    prefer_master=True      # keep existing master values; fill only missing ones
) -> pd.DataFrame:
    """
    Enrich the already-built master_augmented table with any columns
    from specimens_master that are missing (or NaN) in master.

    - Does a left-merge on `on` (default: SpecimenID).
    - If a column exists in both, keeps master values and fills NaNs from specimens
      (unless prefer_master=False, which will prefer specimens values).
    - Optionally drops large array columns from specimens.
    - Adds a Severity column derived from GroupLabel (mild / intermediate / severe).
    """

    # Load if file paths were passed
    if isinstance(master, (str, Path)):
        master_df = pd.read_csv(master)
    else:
        master_df = master.copy()

    if isinstance(specimens, (str, Path)):
        specimens_df = pd.read_csv(specimens)
    else:
        specimens_df = specimens.copy()

    # Optionally drop bulky array columns from specimens
    if drop_array_cols:
        for c in array_cols:
            if c in specimens_df.columns:
                specimens_df = specimens_df.drop(columns=[c])

    # Merge with suffix for overlapping columns
    merged = master_df.merge(specimens_df, how="left", on=on, suffixes=("", "__spec"))

    # Resolve overlaps: fill NaNs from specimens, then drop the __spec copies
    for c in specimens_df.columns:
        if c == on:
            continue
        spec_col = f"{c}__spec"
        if spec_col in merged.columns and c in merged.columns:
            if prefer_master:
                merged[c] = merged[c].where(merged[c].notna(), merged[spec_col])
            else:
                merged[c] = merged[spec_col].where(merged[spec_col].notna(), merged[c])
            merged.drop(columns=[spec_col], inplace=True)
        elif spec_col in merged.columns and c not in merged.columns:
            merged.rename(columns={spec_col: c}, inplace=True)

    # --- Add Severity column from GroupLabel ---
    def classify_severity(label: str) -> str:
        if not isinstance(label, str):
            return None
        l = label.lower()
        if "mild" in l:
            return "Mild"
        elif "intermediate" in l or "moderate" in l:
            return "Intermediate"
        elif "severe" in l:
            return "Severe"
        return None

    if "GroupLabel" in merged.columns:
        merged["Severity"] = merged["GroupLabel"].apply(classify_severity)
    else:
        merged["Severity"] = None

    return merged

def summarize_coupling_results(input_dir, output_file="constitutive_coupling_reliability.csv"):
    """
    Summarize SE-parameter coupling bootstraps across all models, regions, and ranges.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing `se_param_coupling_boot_<region>_<range>_<model>.csv`.
    output_file : str or Path
        Path to save the summarized reliability table.
    """

    input_dir = Path(input_dir)
    rows = []

    for file in input_dir.glob("se_param_coupling_boot_*.csv"):
        # Parse file name: se_param_coupling_boot_<region>_<range>_<model>.csv
        parts = file.stem.split("_")
        try:
            region, rng, model = parts[4], parts[5], "_".join(parts[6:])
        except Exception:
            print(f"[WARN] Skipping {file} (unexpected name format)")
            continue

        df = pd.read_csv(file)

        # Expect columns: term, support, sign_consistency, beta_median, beta_lo, beta_hi
        if df.empty:
            continue

        # Define a reliability score (you can adjust weighting scheme)
        df["reliability"] = (df["support"].fillna(0)/100) * (df["sign_consistency"].fillna(0)/100)

        # Pick top parameter for this model
        best = df.sort_values("reliability", ascending=False).iloc[0]

        rows.append(dict(
            Region=region,
            Range=rng,
            Model=model,
            BestParam=best["term"],
            Reliability=best["reliability"],
            Support=best["support"],
            SignConsistency=best["sign_consistency"],
            BetaMedian=best["beta_median"],
            BetaLo=best["beta_lo"],
            BetaHi=best["beta_hi"],
            n_boots=best.get("n_boots", len(df))
        ))

    summary = pd.DataFrame(rows)

    # For each region & range, rank models by reliability
    summary["Rank"] = summary.groupby(["Region", "Range"])["Reliability"].rank(ascending=False, method="first")

    # Save to disk
    outpath = Path(input_dir) / output_file
    summary.to_csv(outpath, index=False)
    print(f"[INFO] Wrote summarized coupling results to {outpath}")

    return summary

