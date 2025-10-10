import ast
import numpy as np
from pathlib import Path
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
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
    Each panel shows one CoA group vs the Control group (black baseline).
    """

    from ..models.core import get_model, stress_fn
    from ..fitting.core import fit_curve

    _ensure_dir(outdir)
    outdir = Path(outdir)

    # ------------------------------------------------------------------
    # Adjustable internal variables (user can tweak directly here)
    # ------------------------------------------------------------------
    AXIS_FONT_SIZE = 20          # font size for axis labels
    AXIS_TICK_FONT_SIZE = 16          # font size for axis tick labels
    LEGEND_FONT_SIZE = 14         # font size for legends
    SHOW_TITLES = False           # toggle panel titles on/off
    MODEL_LINE_WIDTH = [1.0,3.0]       # thickness of fitted model lines (default 2.0)
    # ------------------------------------------------------------------

    # Panel hiding rules
    hide_ylabel_indices = [1, 2, 4, 5, 7, 8]  # panels without y-label (0-based indexing)
    hide_xlabel_indices = [0, 1, 2, 3, 4, 5]  # panels without x-label

    # ------------------------------------------------------------------
    # Fit ranges
    # ------------------------------------------------------------------
    default_fit_range = tuple(STRAIN_ENERGY_RANGES["default"])
    fit_ranges = STRAIN_ENERGY_RANGES  # for labeled loops later

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
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

    def Yeoh_model_name_for_legend(name: str) -> str:
        """Return compact label for Yeoh models (e.g., 'yeoh2' → 'Y2', otherwise unchanged)."""
        n = name.lower()
        return f"Y{n[-1]}" if n.startswith("yeoh") and n[-1].isdigit() else name

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    sigma_fn = stress_fn(model_name)

    for region, df_region in specimens_df.groupby("Region"):
        # Compute control baseline
        ctrl_x, ctrl_y, ctrl_se = group_average(
            df_region[df_region["GroupName"] == control],
            stretch_step=stretch_step,
            fit_range=default_fit_range,
        )
        ctrl_params = None
        if ctrl_x.size > 0 and not np.all(np.isnan(ctrl_y)):
            ctrl_params = fit_group_average_curve(model_name, ctrl_x, ctrl_y, default_fit_range)

        for range_label, fit_range in fit_ranges.items():
            fine_stretch = np.arange(fit_range[0], fit_range[1] + stretch_step, stretch_step)
            alpha_val = 0.1 if range_label.lower() != "default" else exp_alpha

            fig, axes = plt.subplots(3, 3, figsize=(12, 10))
            axes = axes.flatten()
            p_idx = 0

            coa_groups = [g for g in GROUP_ORDER if g != control and g in df_region["GroupName"].unique()]

            for g in coa_groups:
                if p_idx >= len(axes):
                    break

                sub = df_region[df_region["GroupName"] == g]
                gx_full, gy_full, gse_full = group_average(sub, stretch_step, default_fit_range)
                gx_fit, gy_fit, gse_fit = group_average(sub, stretch_step, fit_range)

                if gx_full.size == 0 or np.all(np.isnan(gy_full)):
                    continue

                color = _group_color(g)
                ax = axes[p_idx]
                p_idx += 1

                # Experimental mean ± SE
                gx_ds_full, gy_ds_full, gse_ds_full = gx_full[::exp_skip], gy_full[::exp_skip], gse_full[::exp_skip]
                ax.errorbar(
                    gx_ds_full, gy_ds_full, yerr=gse_ds_full, fmt="s", color=color, ecolor=color,
                    capsize=3, alpha=alpha_val, label=f"{GROUP_LABELS.get(g, g)} (data)"
                )

                # Fitted curve
                params = fit_group_average_curve(model_name, gx_fit, gy_fit, fit_range)
                if params is not None:
                    y_fit = sigma_fn(fine_stretch, *params)
                    ax.plot(fine_stretch, y_fit, "--", color=color, 
                            linewidth=MODEL_LINE_WIDTH[0] if range_label=="default" else MODEL_LINE_WIDTH[1],
                            alpha=1.0, label=f"{GROUP_LABELS.get(g, g)} (fit)")

                # Control overlay
                if ctrl_x.size > 0 and not np.all(np.isnan(ctrl_y)):
                    cx_ds, cy_ds, cse_ds = ctrl_x[::exp_skip], ctrl_y[::exp_skip], ctrl_se[::exp_skip]
                    ax.errorbar(cx_ds, cy_ds, yerr=cse_ds, fmt="o", color="black", ecolor="black",
                                capsize=3, label=f"{GROUP_LABELS.get(control)} (data)", alpha=alpha_val)
                    if ctrl_params is not None:
                        cf = sigma_fn(fine_stretch, *ctrl_params)
                        ax.plot(fine_stretch, cf, "--", color="black", 
                                linewidth=MODEL_LINE_WIDTH[0] if range_label=="default" else MODEL_LINE_WIDTH[1],
                                alpha=1.0,
                                label=f"{GROUP_LABELS.get(control)} (fit)")

                # Axis & title settings
                ax.set_ylim(0, 800)
                ax.set_xlim(default_fit_range[0], default_fit_range[1])
                if SHOW_TITLES:
                    ax.set_title(GROUP_LABELS.get(g, g), fontsize=AXIS_FONT_SIZE)

                # Hide labels for middle/right/top panels
                if p_idx - 1 not in hide_ylabel_indices:
                    ax.set_ylabel("Stress [kPa]", fontsize=AXIS_FONT_SIZE)
                else:
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)

                if p_idx - 1 not in hide_xlabel_indices:
                    ax.set_xlabel("Stretch λ", fontsize=AXIS_FONT_SIZE)
                else:
                    ax.set_xlabel("")
                    ax.tick_params(labelbottom=False)

                ax.tick_params(axis="x", labelsize=AXIS_TICK_FONT_SIZE)
                ax.tick_params(axis="y", labelsize=AXIS_TICK_FONT_SIZE)
                ax.legend(
                    fontsize=LEGEND_FONT_SIZE,
                    frameon=True,                  # enable border
                    fancybox=True,                 # rounded corners
                    edgecolor="0.3",               # medium grey border
                    facecolor="#fffaf2",             # legend background color
                    framealpha=0.9,                # slightly transparent
                )

                '''
                ax.set_facecolor("#fafafa")   # ultra light gray (almost white)
                ax.set_facecolor("#f2f2f2")   # soft light gray (paper tone)
                ax.set_facecolor("#e6e6e6")   # medium-light gray
                ax.set_facecolor("#dcdcdc")   # gainsboro gray (standard matplotlib gray)
                ax.set_facecolor("#f5f5f5")   # whitesmoke (warm gray-white)
                ax.set_facecolor("#f0f0f0")   # very light neutral gray
                ax.set_facecolor("#f8f4f0")   # warm ivory-gray mix
                ax.set_facecolor("#fff5eb")   # very light orange / cream
                ax.set_facecolor("#ffe8cc")   # light peach
                ax.set_facecolor("#ffe0b3")   # soft pastel orange
                ax.set_facecolor("#ffd699")   # pale orange-yellow (subtle warm tone)
                ax.set_facecolor("#ffefdb")   # light beige / ivory (neutral warm background)
                ax.set_facecolor("#fffaf2")   # off-white with faint orange hue
                '''

            # Turn off remaining panels
            for ax in axes[p_idx:]:
                ax.axis("off")

            plt.tight_layout()
            out_file = outdir / f"stress_stretch_summary_{region}_{model_name}_{range_label}.png"
            plt.savefig(out_file, dpi=300)
            plt.close(fig)
            print(f"[reporting] 📈 Wrote {out_file}")
    
    # ------------------------------------------------------------------
    # Combined plots: diastolic + systolic (consistent with other plots)
    # ------------------------------------------------------------------
    combo_pairs = [("diastolic", "systolic")]

    for pair in combo_pairs:
        if not all(r in fit_ranges for r in pair):
            continue

        r1_label, r2_label = pair
        r1_range, r2_range = fit_ranges[r1_label], fit_ranges[r2_label]

        fine_stretch_r1 = np.arange(r1_range[0], r1_range[1] + stretch_step, stretch_step)
        fine_stretch_r2 = np.arange(r2_range[0], r2_range[1] + stretch_step, stretch_step)

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        p_idx = 0

        coa_groups = [g for g in GROUP_ORDER if g != control and g in df_region["GroupName"].unique()]

        for g in coa_groups:
            if p_idx >= len(axes):
                break

            sub = df_region[df_region["GroupName"] == g]
            gx_full, gy_full, gse_full = group_average(sub, stretch_step, default_fit_range)
            if gx_full.size == 0 or np.all(np.isnan(gy_full)):
                continue

            color = _group_color(g)
            ax = axes[p_idx]
            p_idx += 1

            # ----------------------------
            # Axis limits (set early so patches align correctly)
            # ----------------------------
            ax.set_ylim(0, 800)
            ax.set_xlim(default_fit_range[0], default_fit_range[1])
            y_min, y_max = ax.get_ylim()

            # ----------------------------
            # Vertical shaded regions (now subplot-specific)
            # ----------------------------
            # Soft, neutral tones distinct from GROUP_COLORS
            dia_color = (0.875, 0.905, 0.925)   # #dfe7ec cool gray-blue
            sys_color = (0.972, 0.913, 0.875)   # #f8e9df warm beige-peach

            ax.axvspan(r1_range[0], r1_range[1], ymin=0, ymax=1,
                    color=dia_color, alpha=0.25, zorder=0)
            ax.axvspan(r2_range[0], r2_range[1], ymin=0, ymax=1,
                    color=sys_color, alpha=0.25, zorder=0)

            # Vertical labels centered within patch bounds
            mid_r1 = 0.5 * (r1_range[0] + r1_range[1]) #- (r1_range[1] - r1_range[0])
            mid_r2 = 0.5 * (r2_range[0] + r2_range[1]) - (r2_range[1] - r2_range[0])
            y_center = y_min + 0.35 * (y_max - y_min)

            ax.text(mid_r1, y_center, "Diastolic", color="#1f3a5f",
                    rotation=90, fontsize=AXIS_TICK_FONT_SIZE-2, alpha=0.6,
                    ha="center", va="center", zorder=1)
            ax.text(mid_r2, y_center, "Systolic", color="#5a2a00",
                    rotation=90, fontsize=AXIS_TICK_FONT_SIZE-2, alpha=0.6,
                    ha="center", va="center", zorder=1)

            # ----------------------------
            # Experimental data (transparent)
            # ----------------------------
            gx_ds, gy_ds, gse_ds = gx_full[::exp_skip], gy_full[::exp_skip], gse_full[::exp_skip]
            ax.errorbar(
                gx_ds, gy_ds, yerr=gse_ds, fmt="s", color=color, ecolor=color,
                capsize=3, alpha=0.25, label=f"{GROUP_LABELS.get(g, g)} (data)"
            )

            # ----------------------------
            # Diastolic + systolic fits
            # ----------------------------
            p1 = fit_group_average_curve(model_name, gx_full, gy_full, r1_range)
            if p1 is not None:
                y1 = sigma_fn(fine_stretch_r1, *p1)
                ax.plot(fine_stretch_r1, y1, "--", color=color,
                        linewidth=MODEL_LINE_WIDTH[1], alpha=1.0,
                        label=f"{GROUP_LABELS.get(g, g)} (fit)")

            p2 = fit_group_average_curve(model_name, gx_full, gy_full, r2_range)
            if p2 is not None:
                y2 = sigma_fn(fine_stretch_r2, *p2)
                ax.plot(fine_stretch_r2, y2, "--", color=color,
                        linewidth=MODEL_LINE_WIDTH[1], alpha=1.0)

            # ----------------------------
            # Control overlay (same format)
            # ----------------------------
            if ctrl_x.size > 0 and not np.all(np.isnan(ctrl_y)):
                cx_ds, cy_ds, cse_ds = ctrl_x[::exp_skip], ctrl_y[::exp_skip], ctrl_se[::exp_skip]
                ax.errorbar(cx_ds, cy_ds, yerr=cse_ds, fmt="o", color="black", ecolor="black",
                            capsize=3, alpha=0.25, label=f"{GROUP_LABELS.get(control)} (data)")
                if ctrl_params is not None:
                    cf1 = sigma_fn(fine_stretch_r1, *ctrl_params)
                    cf2 = sigma_fn(fine_stretch_r2, *ctrl_params)
                    ax.plot(fine_stretch_r1, cf1, "--", color="black",
                            linewidth=MODEL_LINE_WIDTH[1], alpha=1.0, label="Control (fit)")
                    ax.plot(fine_stretch_r2, cf2, "--", color="black",
                            linewidth=MODEL_LINE_WIDTH[1], alpha=1.0)

            # ----------------------------
            # Labeling, ticks, legend (same style)
            # ----------------------------
            if SHOW_TITLES:
                ax.set_title(GROUP_LABELS.get(g, g), fontsize=AXIS_FONT_SIZE)
            if p_idx - 1 not in hide_ylabel_indices:
                ax.set_ylabel("Stress [kPa]", fontsize=AXIS_FONT_SIZE)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            if p_idx - 1 not in hide_xlabel_indices:
                ax.set_xlabel("Stretch λ", fontsize=AXIS_FONT_SIZE)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            ax.tick_params(axis="x", labelsize=AXIS_TICK_FONT_SIZE)
            ax.tick_params(axis="y", labelsize=AXIS_TICK_FONT_SIZE)
            ax.legend(
                fontsize=LEGEND_FONT_SIZE,
                frameon=True,
                fancybox=True,
                edgecolor="0.3",
                facecolor="#fffaf2",
                framealpha=0.9,
            )

        # Hide unused panels
        for ax in axes[p_idx:]:
            ax.axis("off")

        plt.tight_layout()
        out_file = outdir / f"stress_stretch_summary_{region}_{model_name}_combo_{r1_label}_{r2_label}.png"
        plt.savefig(out_file, dpi=300)
        plt.close(fig)
        print(f"[reporting] 📊 Wrote combined {r1_label}-{r2_label} → {out_file}")

    # ------------------------------------------------------------------
    # Combined plots: comparing two models (e.g., yeoh2 vs yeoh3)
    # ------------------------------------------------------------------
    model_pairs = [("yeoh2", "yeoh3")]   # you can extend this list
    target_range_label = "default"       # choose which fit range to show
    fit_range = fit_ranges[target_range_label]

    fine_stretch = np.arange(fit_range[0], fit_range[1] + stretch_step, stretch_step)

    for pair in model_pairs:
        m1, m2 = pair
        sigma_fn_m1 = stress_fn(m1)
        sigma_fn_m2 = stress_fn(m2)

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        p_idx = 0

        coa_groups = [g for g in GROUP_ORDER if g != control and g in df_region["GroupName"].unique()]

        for g in coa_groups:
            if p_idx >= len(axes):
                break

            sub = df_region[df_region["GroupName"] == g]
            gx_full, gy_full, gse_full = group_average(sub, stretch_step, default_fit_range)
            if gx_full.size == 0 or np.all(np.isnan(gy_full)):
                continue

            color = _group_color(g)
            ax = axes[p_idx]
            p_idx += 1

            # ----------------------------------------------------------
            # Transparent empirical data
            # ----------------------------------------------------------
            gx_ds, gy_ds, gse_ds = gx_full[::exp_skip], gy_full[::exp_skip], gse_full[::exp_skip]
            ax.errorbar(
                gx_ds, gy_ds, yerr=gse_ds, fmt="s", color=color, ecolor=color,
                capsize=3, alpha=0.25)#label=f"{GROUP_LABELS.get(g, g)} (data)"

            # ----------------------------------------------------------
            # Fit model 1 (e.g., yeoh2) → dashed line
            # ----------------------------------------------------------
            p1 = fit_group_average_curve(m1, gx_full, gy_full, fit_range)
            if p1 is not None:
                y1 = sigma_fn_m1(fine_stretch, *p1)
                ax.plot(
                    fine_stretch, y1, "--", color=color,
                    linewidth=MODEL_LINE_WIDTH[1], alpha=1.0,
                    label=f"{GROUP_LABELS.get(g, g)} ({Yeoh_model_name_for_legend(m1)})"
                )

            # ----------------------------------------------------------
            # Fit model 2 (e.g., yeoh3) → dotted line
            # ----------------------------------------------------------
            p2 = fit_group_average_curve(m2, gx_full, gy_full, fit_range)
            if p2 is not None:
                y2 = sigma_fn_m2(fine_stretch, *p2)
                ax.plot(
                    fine_stretch, y2, ":", color=color,
                    linewidth=MODEL_LINE_WIDTH[1], alpha=1.0,
                    label=f"{GROUP_LABELS.get(g, g)} ({Yeoh_model_name_for_legend(m2)})"
                )

            # ----------------------------------------------------------
            # Control overlay (same treatment)
            # ----------------------------------------------------------
            if ctrl_x.size > 0 and not np.all(np.isnan(ctrl_y)):
                cx_ds, cy_ds, cse_ds = ctrl_x[::exp_skip], ctrl_y[::exp_skip], ctrl_se[::exp_skip]
                ax.errorbar(cx_ds, cy_ds, yerr=cse_ds, fmt="o", color="black", ecolor="black",
                            capsize=3, alpha=0.25)#label=f"{GROUP_LABELS.get(control)} (data)"

                # Fit for each model on control
                if ctrl_params is not None:
                    # Control fits use the same model pair
                    try:
                        ctrl_p1 = fit_group_average_curve(m1, ctrl_x, ctrl_y, fit_range)
                        if ctrl_p1 is not None:
                            cf1 = sigma_fn_m1(fine_stretch, *ctrl_p1)
                            ax.plot(fine_stretch, cf1, "--", color="black",
                                    linewidth=MODEL_LINE_WIDTH[1], alpha=1.0, label=f"Control ({Yeoh_model_name_for_legend(m1)})")
                        ctrl_p2 = fit_group_average_curve(m2, ctrl_x, ctrl_y, fit_range)
                        if ctrl_p2 is not None:
                            cf2 = sigma_fn_m2(fine_stretch, *ctrl_p2)
                            ax.plot(fine_stretch, cf2, ":", color="black",
                                    linewidth=MODEL_LINE_WIDTH[1], alpha=1.0, label=f"Control ({Yeoh_model_name_for_legend(m2)})")
                    except Exception:
                        pass

            # ----------------------------------------------------------
            # Axes, ticks, legend formatting
            # ----------------------------------------------------------
            ax.set_ylim(0, 800)
            ax.set_xlim(default_fit_range[0], default_fit_range[1])
            if SHOW_TITLES:
                ax.set_title(GROUP_LABELS.get(g, g), fontsize=AXIS_FONT_SIZE)

            if p_idx - 1 not in hide_ylabel_indices:
                ax.set_ylabel("Stress [kPa]", fontsize=AXIS_FONT_SIZE)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            if p_idx - 1 not in hide_xlabel_indices:
                ax.set_xlabel("Stretch λ", fontsize=AXIS_FONT_SIZE)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            ax.tick_params(axis="x", labelsize=AXIS_TICK_FONT_SIZE)
            ax.tick_params(axis="y", labelsize=AXIS_TICK_FONT_SIZE)
            ax.legend(
                fontsize=LEGEND_FONT_SIZE,
                frameon=True,
                fancybox=True,
                edgecolor="0.3",
                facecolor="#fffaf2",
                framealpha=0.9,
            )

        # Hide unused subplots
        for ax in axes[p_idx:]:
            ax.axis("off")

        plt.tight_layout()
        out_file = outdir / f"stress_stretch_summary_{region}_{m1}_vs_{m2}_{target_range_label}.png"
        plt.savefig(out_file, dpi=300)
        plt.close(fig)
        print(f"[reporting] 📊 Wrote model comparison ({m1} vs {m2}) → {out_file}")

def tabulate_fitted_moduli(specimens_df, outdir, model_list=["linear_free", "yeoh2", "yeoh3"], fit_ranges=None, stretch_step=0.01):
    """
    Fit models per specimen and export parameter tables (raw + summary).
    """

    from ..models.core import get_model, stress_fn
    from ..fitting.core import fit_curve

    _ensure_dir(outdir)
    outdir = Path(outdir)

    if model_list is None:
        model_list = ["linear_free", "yeoh2", "yeoh3"]
    if fit_ranges is None:
        fit_ranges = STRAIN_ENERGY_RANGES

    raw_records = []

    for (region, group), df_sub in specimens_df.groupby(["Region", "GroupName"]):
        for model_name in model_list:
            model_info = get_model(model_name)
            param_names = model_info["params"]

            for fit_label, fit_range in fit_ranges.items():
                lo, hi = fit_range
                for sample_id, df_s in df_sub.groupby("SpecimenID"):
                    x = parse_array(df_s.iloc[0]["Stretch"])
                    y = parse_array(df_s.iloc[0]["Stress"])
                    if x.size < 3:
                        continue

                    # Fit the curve within the range
                    mask = (x >= lo) & (x <= hi)
                    x_fit, y_fit = x[mask], y[mask]
                    if len(x_fit) < 3:
                        continue
                    try:
                        res = fit_curve(x_fit, y_fit, model_name)
                        for pname in param_names:
                            raw_records.append({
                                "SpecimenID": sample_id,
                                "GroupName": group,
                                "Region": region,
                                "Model": model_name,
                                "FitRange": fit_label,
                                "ParamName": pname,
                                "ParamValue": res.params[pname],
                            })
                    except Exception as e:
                        print(f"[tabulate] ⚠️ Fit failed: {model_name} / {group} / {region} → {e}")

    # Convert to DataFrame
    raw_df = pd.DataFrame(raw_records)
    raw_file = outdir / "fitted_parameters_raw.csv"
    raw_df.to_csv(raw_file, index=False)
    print(f"[reporting] 💾 Wrote raw fits → {raw_file}")

    # ------------------------------------------------------------------
    # Summarize per Group, Region, Model, Param
    # ------------------------------------------------------------------
    summary_df = (
        raw_df.groupby(["Region", "Model", "ParamName", "FitRange", "GroupName"])
        .agg(Mean=("ParamValue", "mean"), SD=("ParamValue", "std"), N=("ParamValue", "count"))
        .reset_index()
    )

    sum_file = outdir / "fitted_parameters_summary.csv"
    summary_df.to_csv(sum_file, index=False)
    generate_transposed_latex_tables(sum_file, raw_csv=raw_file, outdir=outdir, p_show_sigstars=True)
    print(f"[reporting] 📊 Wrote summary table → {sum_file}")

    return raw_df, summary_df

def generate_latex_tables(summary_csv, outdir):
    """
    Generate LaTeX tables summarizing constitutive parameters (mean ± SD)
    and one-way ANOVA p-values per model, region, and parameter.
    """
    df = pd.read_csv(summary_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    latex_snippets = []

    # Unique combinations of region and model
    for (region, model), df_sub in df.groupby(["Region", "Model"]):
        tex_lines = []
        tex_lines.append("\\begin{table}[!ht]")
        tex_lines.append("\\centering")
        tex_lines.append(f"\\caption{{{model.replace('_', ' ').title()} constitutive parameters ({region.title()} region)}}")

        groups = df_sub["GroupName"].unique().tolist()
        groups_sorted = sorted(groups, key=lambda x: x.lower())

        # Header
        header_cols = " & ".join(groups_sorted + ["ANOVA $p$"])
        tex_lines.append("\\begin{tabular}{l" + "c" * (len(groups_sorted) + 1) + "}")
        tex_lines.append("\\hline")
        tex_lines.append("Parameter & " + header_cols + " \\\\")
        tex_lines.append("\\hline")

        for param, df_p in df_sub.groupby("ParamName"):
            means = []
            p_values = np.nan

            # Collect per-group data
            group_values = []
            for g in groups_sorted:
                sub = df_p[df_p["GroupName"] == g]
                if sub.empty:
                    means.append("--")
                    group_values.append([])
                    continue
                m, s = sub["Mean"].mean(), sub["SD"].mean()
                means.append(f"{m:.2f} ± {s:.2f}")
                group_values.append(sub["Mean"].dropna().values)

            # ANOVA across groups
            valid_groups = [arr for arr in group_values if len(arr) > 1]
            if len(valid_groups) > 1:
                try:
                    _, p = f_oneway(*valid_groups)
                    p_values = p
                except Exception:
                    p_values = np.nan
            p_str = f"{p_values:.3f}" if np.isfinite(p_values) else "--"

            tex_lines.append(f"${param}$ & " + " & ".join(means) + f" & {p_str} \\\\")

        tex_lines.append("\\hline")
        tex_lines.append("\\end{tabular}")
        tex_lines.append(f"\\label{{tab:{model}_{region}}}")
        tex_lines.append("\\end{table}")
        tex_lines.append("")

        # Write to file
        tex_out = outdir / f"{model}_{region}.tex"
        with open(tex_out, "w") as f:
            f.write("\n".join(tex_lines))
        print(f"[latex] 🧾 Wrote {tex_out}")

        latex_snippets.append("\n".join(tex_lines))

    # Optional: master concatenation
    master_file = outdir / "constitutive_tables.tex"
    with open(master_file, "w") as f:
        f.write("\n\n".join(latex_snippets))
    print(f"[latex] 📚 Wrote combined master file → {master_file}")

def generate_transposed_latex_tables(
    summary_csv,
    outdir,
    raw_csv=None,                # <-- NEW: raw per-sample parameters for proper ANOVA
    fit_order=("diastolic","physiological","systolic","default"),
    p_show_sigstars=False        # optional: add *, **, *** markers
):
    """
    Make LaTeX tables (transposed):
      • One table per (Model, Region, ParamName)
      • Rows: study groups (in GROUP_ORDER, labeled via GROUP_LABELS)
      • Columns: fit ranges (diastolic, physiological, systolic, default)
      • Bottom row: 'ANOVA $p$' with a p-value per fit range (computed across groups).
    """
    df_sum = pd.read_csv(summary_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # restrict to available fit ranges in the data, but keep requested order
    present = set(df_sum["FitRange"].unique())
    fit_cols = [f for f in fit_order if f in present]

    # (optional) raw per-sample values for ANOVA
    df_raw = pd.read_csv(raw_csv) if raw_csv else None

    def fmt_mu_sd(mu, sd):
        return f"{mu:.2f} ± {sd:.2f}"

    def fmt_p(p):
        if not np.isfinite(p):
            return "--"
        if p < 0.001:
            s = "<0.001"
        else:
            s = f"{p:.3f}"
        if p_show_sigstars:
            if p < 0.001: s += "***"
            elif p < 0.01: s += "**"
            elif p < 0.05: s += "*"
        return s

    all_tex = []

    # Loop per Region → Model → Parameter
    for (region, model, param), df_sub in df_sum.groupby(["Region", "Model", "ParamName"]):
        lines = []
        lines.append(r"\begin{table}[!ht]")
        lines.append(r"\centering")
        lines.append(
            f"\\caption{{{model.replace('_',' ').title()}: {region.title()} region – Constitutive parameter ${param}$}}"
        )

        # column spec: 1 for 'Group' + one per fit range
        lines.append("\\begin{tabular}{l" + "c" * len(fit_cols) + "}")
        lines.append("\\hline")
        lines.append("Group & " + " & ".join(fit_cols) + " \\\\")
        lines.append("\\hline")

        # group order and labels
        groups = [g for g in GROUP_ORDER if g in df_sub["GroupName"].unique()]

        # table body: one row per group
        for g in groups:
            label = GROUP_LABELS.get(g, g)
            cell_vals = []
            for f in fit_cols:
                sub = df_sub[(df_sub["GroupName"] == g) & (df_sub["FitRange"] == f)]
                if sub.empty:
                    cell_vals.append("--")
                else:
                    mu = sub["Mean"].mean()
                    sd = sub["SD"].mean()
                    cell_vals.append(fmt_mu_sd(mu, sd))
            lines.append(f"{label} & " + " & ".join(cell_vals) + r" \\")
        
        # --- ANOVA row per fit range (columns) ---
        p_cells = []
        for f in fit_cols:
            if df_raw is None:
                p_cells.append("--")
                continue
            # collect per-group sample arrays for this fit range
            arrs = []
            for g in groups:
                vals = df_raw[
                    (df_raw["Region"] == region) &
                    (df_raw["Model"] == model) &
                    (df_raw["ParamName"] == param) &
                    (df_raw["FitRange"] == f) &
                    (df_raw["GroupName"] == g)
                ]["ParamValue"].dropna().values
                # include only groups with >= 2 samples to avoid degenerate ANOVA
                if len(vals) >= 2:
                    arrs.append(vals)
            if len(arrs) >= 2:
                try:
                    _, p = f_oneway(*arrs)
                except Exception:
                    p = np.nan
            else:
                p = np.nan
            p_cells.append(fmt_p(p))

        # add the ANOVA row at the end
        lines.append("\\hline")
        lines.append("ANOVA $p$ & " + " & ".join(p_cells) + r" \\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append(f"\\label{{tab:{model}_{param}_{region}}}")
        lines.append("\\end{table}")
        lines.append("")

        # write .tex
        tex_out = outdir / f"{model}_{param}_{region}.tex"
        with open(tex_out, "w") as f:
            f.write("\n".join(lines))
        print(f"[latex] 🧮 Wrote {tex_out}")

        all_tex.append("\n".join(lines))

    # combined master file
    master = outdir / "constitutive_tables_transposed.tex"
    with open(master, "w") as f:
        f.write("\n\n".join(all_tex))
    print(f"[latex] 📚 Combined all tables → {master}")


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
# 4. Constitutive parameter plots + ANOVA/Tukey + (optional) bootstrap restatement
# -------------------------------------------------------------------
def plot_model_params(param_long_df, out_dir, model="holz_iso"):
    """
    Boxplot of model parameters across groups for a given model.
    Additionally collects the exact plotting dataframe for downstream stats.
    """
    _ensure_dir(out_dir)
    out_dir = Path(out_dir)

    df = param_long_df[param_long_df["Model"] == model].copy()
    # normalize names up front
    df["GroupName"] = (
        df["GroupName"].astype(str).str.strip().replace(
            {"Control1": "Control", "control1": "Control", "CTRL": "Control"}
        )
    )

    if df.empty:
        print(f"[reporting] No parameters found for model {model}")
        return

    # ---------- style knobs ----------
    AXIS_FONT_SIZE   = 20
    TICK_FONT_SIZE   = 16
    TITLE_FONT_SIZE  = 22
    SHOW_TITLES      = True
    STRIP_DOT_SIZE   = 3
    STRIP_JITTER     = 0.25
    FIG_SIZE         = (10, 6)
    GRID_ALPHA       = 0.15
    BOX_LINE_WIDTH   = 1.5

    # collector for stats
    collected_rows = []

    regions = df["Region"].unique()
    for region in regions:
        reg_df = df[df["Region"] == region].copy()

        for param in reg_df["Parameter"].unique():
            sub = reg_df[reg_df["Parameter"] == param].copy()

            # normalize again in subset just to be bulletproof
            sub["GroupName"] = (
                sub["GroupName"].astype(str).str.strip().replace(
                    {"Control1": "Control", "control1": "Control", "CTRL": "Control"}
                )
            )
            sub["GroupLabel"] = sub["GroupName"].map(GROUP_LABELS)

            # keep only rows with label + value
            sub = sub.dropna(subset=["GroupLabel", "Value"]).copy()

            # --- collect rows for stats (exactly what we plot) ---
            if not sub.empty:
                collected_rows.extend(
                    dict(
                        Region=region,
                        Parameter=str(param),
                        GroupName=gn,
                        GroupLabel=gl,
                        Value=float(v),
                        Model=model,
                    )
                    for gn, gl, v in zip(sub["GroupName"], sub["GroupLabel"], sub["Value"])
                )

            # palette + order only among present groups
            present_groups = [g for g in GROUP_ORDER if g in sub["GroupName"].unique()]
            palette = {GROUP_LABELS[g]: _group_color(g) for g in present_groups}
            order   = [GROUP_LABELS[g] for g in present_groups]

            # plot
            plt.figure(figsize=FIG_SIZE)
            sns.boxplot(
                data=sub, x="GroupLabel", y="Value",
                hue="GroupLabel", legend=False,
                order=order, palette=palette,
                linewidth=BOX_LINE_WIDTH, width=0.6,
            )
            sns.stripplot(
                data=sub, x="GroupLabel", y="Value",
                color="black", size=STRIP_DOT_SIZE, jitter=STRIP_JITTER,
                order=order,
            )
            plt.xticks(rotation=45, ha="right", fontsize=TICK_FONT_SIZE)
            plt.yticks(fontsize=TICK_FONT_SIZE)
            plt.xlabel("", fontsize=AXIS_FONT_SIZE)
            # Yeoh parameters are in stress units
            ylab = "Parameter Value [kPa]" if str(param).lower().startswith(("c", "yeoh", "mr", "mu")) else "Parameter Value"
            plt.ylabel(ylab, fontsize=AXIS_FONT_SIZE)
            plt.grid(alpha=GRID_ALPHA, zorder=0)

            if SHOW_TITLES:
                plt.title(f"{model} parameter: {param} ({region})", fontsize=TITLE_FONT_SIZE)

            plt.tight_layout()
            out_file = out_dir / f"{model}_{param}_{region}.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"[reporting] 📦 Wrote {out_file}")

    # ------ after plotting: build the stats dataframe and save ------
    if collected_rows:
        param_plot_df = pd.DataFrame(collected_rows)
        stats_src_csv = out_dir / f"{model}_param_plot_dataframe.csv"
        param_plot_df.to_csv(stats_src_csv, index=False)
        print(f"[reporting] 🧾 Saved stats source DF → {stats_src_csv}")
        # run stats now
        run_param_stats(param_plot_df, out_dir)
    else:
        print("[reporting] No rows collected for stats (nothing to analyze).")

def run_param_stats(param_plot_df: pd.DataFrame, out_dir: Path):
    """
    Runs one-way ANOVA and Tukey HSD per (Region, Parameter) on the collected plotting dataframe.
    Writes ANOVA text and Tukey CSVs under out_dir/'stats'.
    """
    out_dir = Path(out_dir)
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # ensure clean columns
    df = param_plot_df.copy()
    df["GroupLabel"] = df["GroupLabel"].astype(str).str.strip()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["GroupLabel", "Value"])

    for (region, param), sub in df.groupby(["Region", "Parameter"]):
        stat_df = sub[["GroupLabel", "Value"]].rename(columns={"GroupLabel": "group", "Value": "value"}).copy()

        # group counts
        counts = stat_df.groupby("group").size().sort_values(ascending=False)

        # need ≥2 groups and ≥2 samples per involved group for Tukey
        if stat_df["group"].nunique() < 2:
            _write_anova_note(stats_dir, region, param, msg="Not enough groups for ANOVA.")
            continue

        # ANOVA
        anova_p = np.nan
        try:
            m = ols("value ~ C(group)", data=stat_df).fit()
            tbl = anova_lm(m, typ=2)
            if "C(group)" in tbl.index:
                anova_p = float(tbl.loc["C(group)", "PR(>F)"])
        except Exception as e:
            _write_anova_note(stats_dir, region, param, msg=f"ANOVA failed: {e}")
            continue

        # Tukey (only when every group has at least 2 observations)
        tukey_path = None
        if (counts >= 2).all() and np.isfinite(anova_p):
            try:
                tuk = pairwise_tukeyhsd(
                    endog=stat_df["value"].values,
                    groups=stat_df["group"].values,
                    alpha=0.05,
                )
                # build DataFrame robustly from summary() (avoids shape issues)
                summ = tuk.summary()
                header = summ.data[0]
                rows = summ.data[1:]
                tuk_df = pd.DataFrame(rows, columns=header)
                tukey_path = stats_dir / f"tukey_{param}_{region}.csv"
                tuk_df.to_csv(tukey_path, index=False)
            except Exception as e:
                _write_anova_note(stats_dir, region, param, msg=f"Tukey failed: {e}")

        # write ANOVA summary
        with open(stats_dir / f"anova_{param}_{region}.txt", "w") as fh:
            fh.write(f"ANOVA for {param} ({region})\n")
            fh.write(f"group sizes:\n{counts.to_string()}\n\n")
            fh.write(f"p_value: {anova_p:.6g}\n")
            if tukey_path is not None:
                fh.write(f"Tukey CSV: {tukey_path}\n")

def _write_anova_note(stats_dir: Path, region: str, param: str, msg: str):
    with open(stats_dir / f"anova_{param}_{region}.txt", "w") as fh:
        fh.write(f"{msg}\n")

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
