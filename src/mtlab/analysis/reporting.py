import ast
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from ..models.core import get_model, stress_fn

from ..data.schema import GROUP_COLORS, GROUP_LABELS, GROUP_ORDER, STRAIN_ENERGY_RANGES

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
# 1b. Group-average stress–stretch plots with fits
# -------------------------------------------------------------------
def plot_stress_stretch_summary(
    specimens_df,
    outdir,
    model_name="holz_iso",
    control="Control",
    exp_alpha=1.0,
    stretch_step=0.01,
    exp_skip=5,
):
    """
    Multi-panel (3×3) plot of mean ± SE stress–stretch curves by group with optional model fits.

    Parameters
    ----------
    specimens_df : pd.DataFrame
        Must contain ["GroupName","Region","Stretch","Stress"].
    outdir : Path or str
        Output directory for plots.
    model_name : str
        Model key recognized by get_model() / stress_fn().
    control : str
        Control group name to overlay.
    exp_alpha : float
        Transparency for experimental data points.
    stretch_step : float
        Step size (Δλ) for stretch sampling. Default = 0.01.
    exp_skip : int
        Plot every Nth experimental point to declutter error bars. Default = 5.
    """

    from ..models.core import get_model, stress_fn
    from ..fitting.core import fit_curve

    _ensure_dir(outdir)
    outdir = Path(outdir)

    # --- Single source of truth for all fit ranges ---
    default_fit_range = tuple(STRAIN_ENERGY_RANGES["default"])
    fit_ranges = STRAIN_ENERGY_RANGES  # for labeled loops later

    # --- helper functions ---
    def group_average(df_group, stretch_step, fit_range):
        """Compute group-average stress at evenly spaced stretch points (clamped to fit_range)."""
        if df_group.empty:
            return np.array([]), np.array([]), np.array([])
        x_arrays = [parse_array(r["Stretch"]) for _, r in df_group.iterrows()]
        x_arrays = [x for x in x_arrays if x.size > 1]
        if not x_arrays:
            return np.array([]), np.array([]), np.array([])

        data_min = float(np.nanmin([np.nanmin(x) for x in x_arrays]))
        data_max = float(np.nanmax([np.nanmax(x) for x in x_arrays]))

        lo, hi = float(fit_range[0]), float(fit_range[1])
        stretch_min = max(data_min, lo)
        stretch_max = min(data_max, hi)

        if not np.isfinite(stretch_min) or not np.isfinite(stretch_max) or stretch_max <= stretch_min:
            return np.array([]), np.array([]), np.array([])

        n = int(round((stretch_max - stretch_min) / stretch_step)) + 1
        stretch_points = np.linspace(stretch_min, stretch_max, n)

        all_y = []
        for _, row in df_group.iterrows():
            x = parse_array(row["Stretch"])
            y = parse_array(row["Stress"])
            if x.size < 2:
                continue
            y_interp = np.interp(stretch_points, x, y, left=np.nan, right=np.nan)
            all_y.append(y_interp)

        if not all_y:
            return stretch_points, np.full_like(stretch_points, np.nan), np.full_like(stretch_points, np.nan)

        all_y = np.vstack(all_y)
        mean_y = np.nanmean(all_y, axis=0)
        std_y  = np.nanstd(all_y, axis=0)
        se_y   = std_y / np.sqrt(all_y.shape[0])
        return stretch_points, mean_y, se_y

    def fit_group_average_curve(model_name, stretch_points, stress_points, fit_range):
        """Fit model to mean group data within range."""
        mask = (stretch_points >= fit_range[0]) & (stretch_points <= fit_range[1])
        x_fit, y_fit = stretch_points[mask], stress_points[mask]
        if len(x_fit) < 3:
            return None
        try:
            res = fit_curve(x_fit, y_fit, model_name)
            theta = [res.params[k] for k in get_model(model_name)["params"]]
            return theta
        except Exception as e:
            print(f"[reporting] ⚠️ fit failed for {model_name}: {e}")
            return None

    # --- plotting ---
    sigma_fn = stress_fn(model_name)

    for region, df_region in specimens_df.groupby("Region"):
        # Compute control once using default range
        ctrl_x, ctrl_y, ctrl_se = group_average(
            df_region[df_region["GroupName"] == control],
            stretch_step=stretch_step,
            fit_range=default_fit_range,
        )
        ctrl_params = None
        if ctrl_x.size > 0 and not np.all(np.isnan(ctrl_y)):
            ctrl_params = fit_group_average_curve(model_name, ctrl_x, ctrl_y, default_fit_range)

        # Iterate through all defined fit ranges (skip "default" if redundant)
        for range_label, fit_range in fit_ranges.items():
            fine_stretch = np.arange(fit_range[0], fit_range[1] + stretch_step, stretch_step)
            alpha_val = 0.1 if range_label.lower() != "default" else exp_alpha

            fig, axes = plt.subplots(3, 3, figsize=(12, 10))
            axes = axes.flatten()
            p_idx = 0

            for g in GROUP_ORDER:
                if g not in df_region["GroupName"].unique():
                    continue
                if p_idx >= len(axes):
                    break

                sub = df_region[df_region["GroupName"] == g]

                # --- Full-range (unmasked) data for visualization ---
                gx_full, gy_full, gse_full = group_average(
                    sub, stretch_step=stretch_step, fit_range=default_fit_range
                )

                # --- Fit-range (masked) data for fitting ---
                gx_fit, gy_fit, gse_fit = group_average(
                    sub, stretch_step=stretch_step, fit_range=fit_range
                )

                if gx_full.size == 0 or np.all(np.isnan(gy_full)):
                    continue

                color = _group_color(g)
                ax = axes[p_idx]
                p_idx += 1

                # Experimental mean ± SE (full range, transparent)
                gx_ds_full = gx_full[::exp_skip]
                gy_ds_full = gy_full[::exp_skip]
                gse_ds_full = gse_full[::exp_skip]
                ax.errorbar(
                    gx_ds_full, gy_ds_full, yerr=gse_ds_full, fmt="s", color=color, ecolor=color,
                    capsize=3, alpha=0.25, label=f"{g} (full data)"
                )

                # Fitted curve (within fit range)
                params = fit_group_average_curve(model_name, gx_fit, gy_fit, fit_range)
                if params is not None:
                    y_fit = sigma_fn(fine_stretch, *params)
                    ax.plot(fine_stretch, y_fit, "--", color=color, alpha=0.8, label=f"{g} fit")
                else:
                    print(f"[reporting] ⚠️ No fit params for {g} in {region} over {range_label}")

                # Control overlay
                if ctrl_x.size > 0 and not np.all(np.isnan(ctrl_y)):
                    cx_ds, cy_ds, cse_ds = ctrl_x[::exp_skip], ctrl_y[::exp_skip], ctrl_se[::exp_skip]
                    ax.errorbar(cx_ds, cy_ds, yerr=cse_ds, fmt="o", color="black", ecolor="black",
                                capsize=3, label="Control data", alpha=0.3)
                    if ctrl_params is not None:
                        cf = sigma_fn(fine_stretch, *ctrl_params)
                        ax.plot(fine_stretch, cf, "--", color="black", alpha=0.8, label="Control fit")

                ax.set_ylim(0, 800)
                ax.set_xlabel("Stretch λ")
                ax.set_xlim(default_fit_range[0], default_fit_range[1])  # full-range x-axis
                ax.set_ylabel("Stress [kPa]")
                ax.set_title(_group_label(g))
                ax.legend(fontsize="x-small")


            for ax in axes[p_idx:]:
                ax.axis("off")

            plt.tight_layout()
            out_file = outdir / f"stress_stretch_summary_{region}_{model_name}_{range_label}.png"
            plt.savefig(out_file, dpi=300)
            plt.close(fig)
            print(f"[reporting] 📈 Wrote {out_file}")

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
        reg_df = se_df[se_df["Region"] == region].copy()   # <-- FIX 1

        # Map labels safely
        reg_df["GroupLabel"] = reg_df["GroupName"].map(GROUP_LABELS)

        # Build palette keyed by label, not raw name  <-- FIX 2
        palette = {
            GROUP_LABELS[g]: _group_color(g)
            for g in GROUP_ORDER if g in reg_df["GroupName"].unique()
        }

        order = [GROUP_LABELS[g] for g in GROUP_ORDER if g in reg_df["GroupName"].unique()]

        sns.boxplot(
            data=reg_df,
            x="GroupLabel", y="StrainEnergy",
            hue="GroupLabel", legend=False,   # <-- FIX 3 (future-proof)
            order=order,
            palette=palette
        )
        sns.stripplot(
            data=reg_df,
            x="GroupLabel", y="StrainEnergy",
            color="black", size=3, jitter=True,
            order=order
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
        reg_df = df[df["Region"] == region].copy()  # <-- avoid SettingWithCopyWarning
        for param in reg_df["Parameter"].unique():
            sub = reg_df[reg_df["Parameter"] == param].copy()  # <-- make full copy
            sub["GroupLabel"] = sub["GroupName"].map(GROUP_LABELS)

            # Build palette keyed by labels, not names
            palette = {
                GROUP_LABELS[g]: _group_color(g)
                for g in GROUP_ORDER if g in sub["GroupName"].unique()
            }
            order = [GROUP_LABELS[g] for g in GROUP_ORDER if g in sub["GroupName"].unique()]

            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=sub,
                x="GroupLabel", y="Value",
                hue="GroupLabel", legend=False,  # tie hue to x-variable for seaborn>=0.14
                order=order,
                palette=palette,
            )
            sns.stripplot(
                data=sub,
                x="GroupLabel", y="Value",
                color="black", size=3, jitter=True,
                order=order,
            )
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{model} parameter {param} ({region})")
            plt.tight_layout()

            out_file = Path(out_dir) / f"{model}_{param}_{region}.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"[reporting] Wrote {out_file}")
# -------------------------------------------------------------------

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
