import ast
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from ..models.core import get_model, stress_fn

from ..data.schema import GROUP_COLORS, GROUP_LABELS, GROUP_ORDER

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------
def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def _group_color(name):
    return GROUP_COLORS.get(name, (0.5, 0.5, 0.5))  # fallback grey

def _group_label(name):
    return GROUP_LABELS.get(name, name)

def parse_array(val):
    """Convert stringified list from CSV into a real numpy array."""
    if isinstance(val, str):
        try:
            return np.array(ast.literal_eval(val), dtype=float)
        except Exception:
            return np.array([], dtype=float)
    elif isinstance(val, (list, np.ndarray)):
        return np.array(val, dtype=float)
    else:
        return np.array([], dtype=float)

# -------------------------------------------------------------------
# 1. Stress–stretch curves
# -------------------------------------------------------------------
def plot_stress_stretch(specimens_df, outdir):
    """
    Plot raw experimental stress–stretch curves grouped by region and group.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for (region, group), g in specimens_df.groupby(["Region", "GroupName"]):
        plt.figure(figsize=(4, 3))
        for _, row in g.iterrows():
            # In plot_stress_stretch
            x = parse_array(row["Stretch"])
            y = parse_array(row["Stress"])
            if len(x) == 0 or len(y) == 0:
                continue
            plt.plot(x, y, alpha=0.5)
        plt.title(f"{region} - {group}")
        plt.xlabel("Stretch (λ)")
        plt.ylabel("Stress")
        plt.tight_layout()
        plt.savefig(outdir / f"stress_stretch_{region}_{group}.png", dpi=200)
        plt.close()

# -------------------------------------------------------------------
# 2. Strain energy plots
# -------------------------------------------------------------------
def plot_strain_energy(se_df, out_dir):
    """
    Boxplot of strain energy per group × region.
    """
    _ensure_dir(out_dir)

    regions = se_df["Region"].unique()
    for region in regions:
        plt.figure(figsize=(10, 6))
        reg_df = se_df[se_df["Region"] == region]
        reg_df["GroupLabel"] = reg_df["GroupName"].map(GROUP_LABELS)

        sns.boxplot(
            data=reg_df,
            x="GroupLabel", y="StrainEnergy",
            order=[GROUP_LABELS[g] for g in GROUP_ORDER if g in reg_df["GroupName"].unique()],
            palette={g: _group_color(g) for g in GROUP_ORDER}
        )
        sns.stripplot(
            data=reg_df,
            x="GroupLabel", y="StrainEnergy",
            color="black", size=3, jitter=True,
            order=[GROUP_LABELS[g] for g in GROUP_ORDER if g in reg_df["GroupName"].unique()]
        )
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Strain Energy by Group ({region})")
        plt.tight_layout()
        out_file = Path(out_dir) / f"strain_energy_{region}.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"[reporting] Wrote {out_file}")

# -------------------------------------------------------------------
# 3. Model performance summary
# -------------------------------------------------------------------
def summarize_model_metrics(metrics_df, out_dir):
    """
    Summarize metrics across specimens (mean ± SD) and plot bar chart.
    """
    _ensure_dir(out_dir)

    summary = (metrics_df
               .groupby("Model")
               .agg(RMSE_mean=("RMSE", "mean"),
                    RMSE_std=("RMSE", "std"),
                    NRMSE_mean=("NRMSE", "mean"),
                    NRMSE_std=("NRMSE", "std"))
               .reset_index())
    out_csv = Path(out_dir) / "model_performance.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[reporting] Wrote {out_csv}")

    # Bar plot of mean NRMSE
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x="Model", y="NRMSE_mean", yerr=summary["NRMSE_std"])
    plt.ylabel("NRMSE (mean ± SD)")
    plt.title("Model Performance (NRMSE)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_file = Path(out_dir) / "model_performance.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[reporting] Wrote {out_file}")

# -------------------------------------------------------------------
# 4. Constitutive parameter plots
# -------------------------------------------------------------------
def plot_model_params(param_long_df, out_dir, model="holz_iso"):
    """
    Boxplot of model parameters across groups for a given model.
    """
    _ensure_dir(out_dir)

    df = param_long_df[param_long_df["Model"] == model].copy()
    if df.empty:
        print(f"[reporting] No parameters found for model {model}")
        return

    regions = df["Region"].unique()
    for region in regions:
        reg_df = df[df["Region"] == region]
        for param in reg_df["Parameter"].unique():
            plt.figure(figsize=(10, 6))
            sub = reg_df[reg_df["Parameter"] == param]
            sub["GroupLabel"] = sub["GroupName"].map(GROUP_LABELS)

            sns.boxplot(
                data=sub,
                x="GroupLabel", y="Value",
                order=[GROUP_LABELS[g] for g in GROUP_ORDER if g in sub["GroupName"].unique()],
                palette={g: _group_color(g) for g in GROUP_ORDER}
            )
            sns.stripplot(
                data=sub,
                x="GroupLabel", y="Value",
                color="black", size=3, jitter=True,
                order=[GROUP_LABELS[g] for g in GROUP_ORDER if g in sub["GroupName"].unique()]
            )
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{model} parameter {param} ({region})")
            plt.tight_layout()
            out_file = Path(out_dir) / f"{model}_{param}_{region}.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"[reporting] Wrote {out_file}")

def overlay_model_fits(specimens_df, fits_df, outdir, model="holz_iso"):
    """
    Overlay experimental stress–stretch data with predicted model fits.

    Parameters
    ----------
    specimens_df : pd.DataFrame
        Must contain SpecimenID, Stretch, Stress.
    fits_df : pd.DataFrame
        Must contain SpecimenID, Model, Params (as dict or JSON string).
    outdir : Path
        Output directory for plots.
    model : str
        Model name to overlay (default 'holz_iso').
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare stress function
    model_info = get_model(model)
    sigma_fn = stress_fn(model)

    # Ensure Params is parsed into dicts
    import ast, pandas as pd
    if isinstance(fits_df.loc[0, "Params"], str):
        fits_df = fits_df.copy()
        fits_df["Params"] = fits_df["Params"].apply(ast.literal_eval)

    for specimen_id, group in specimens_df.groupby("SpecimenID"):
        # experimental
        x = parse_array(group["Stretch"].iloc[0])
        y = parse_array(group["Stress"].iloc[0])
        if len(x) == 0 or len(y) == 0:
            continue

        # find fitted params
        fit_row = fits_df.query("SpecimenID == @specimen_id and Model == @model")
        if fit_row.empty:
            continue
        params = fit_row["Params"].iloc[0]
        theta = [params[p] for p in model_info["params"]]

        # predicted
        y_pred = sigma_fn(x, *theta)

        # plot
        plt.figure(figsize=(4,3))
        plt.plot(x, y, "o", label="Experimental", alpha=0.7)
        plt.plot(x, y_pred, "-", label=f"{model} fit")
        plt.xlabel("Stretch (λ)")
        plt.ylabel("Stress")
        plt.title(f"{specimen_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{specimen_id}_{model}_overlay.png", dpi=200)
        plt.close()
