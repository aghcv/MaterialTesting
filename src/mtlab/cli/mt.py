import argparse, json
from pathlib import Path
import os
import pandas as pd
from ..data.io import load_material_data
from ..analysis.strain_energy import compute_strain_energy
from ..analysis.constitutive_summary import specimen_level_param_table, per_group_stats, \
    expand_param_column, build_master_table, enrich_master_with_specimens, summarize_coupling_results
from ..fitting.core import fit_curve
from ..verification.validate import kfold_verify, trustworthy_models
from ..data.schema import PATHS, ensure_dirs, STRAIN_ENERGY_RANGES

def main():
    p = argparse.ArgumentParser(prog="mt")
    sub = p.add_subparsers(dest="cmd", required=True)
    
    # ----------------------------------------------------------------------
    # Ingest
    # ----------------------------------------------------------------------
    p_ing = sub.add_parser("ingest", help="ingest proximal/distal/thickness xlsx into standardized CSVs")
    p_ing.add_argument("--proximal", required=True)
    p_ing.add_argument("--distal", required=True)
    p_ing.add_argument("--thickness", required=True)
    p_ing.add_argument("--out", default=PATHS["data"])

    # ----------------------------------------------------------------------
    # Fit
    # ----------------------------------------------------------------------
    p_fit = sub.add_parser("fit", help="fit a model per specimen")
    p_fit.add_argument("--specimens", default=os.path.join(PATHS["data"], "specimens_master.csv"))   # raw inputs must be given
    p_fit.add_argument("--model", default="all", help="model name or 'all'")
    p_fit.add_argument("--out", default=PATHS["model"])
    p_fit.add_argument("--ranges", default=",".join(STRAIN_ENERGY_RANGES.keys()), help="Comma-separated list of stretch ranges to fit")

    # ----------------------------------------------------------------------
    # Summarize (includes augmented master table)
    # ----------------------------------------------------------------------
    p_sum = sub.add_parser("summarize", help="summarize fitted parameters and augment with strain energy + metrics")
    p_sum.add_argument("--fits", default=os.path.join(PATHS["model"], "fits.csv"))
    p_sum.add_argument("--strain", default=os.path.join(PATHS["strain_energy"], "strain_energy_stats.csv"))
    p_sum.add_argument("--metrics", default=os.path.join(PATHS["verify"], "metrics.csv"))
    p_sum.add_argument("--specimens", default=os.path.join(PATHS["data"], "specimens_master.csv"))
    p_sum.add_argument("--out", default=PATHS["model"])

    # ----------------------------------------------------------------------
    # Strain energy
    # ----------------------------------------------------------------------
    p_se = sub.add_parser("strain-energy", help="compute strain energy per specimen")
    p_se.add_argument("--specimens", default=os.path.join(PATHS["data"], "specimens_master.csv"))
    p_se.add_argument("--out", default=PATHS["strain_energy"])

    # ----------------------------------------------------------------------
    # Verification
    # ----------------------------------------------------------------------
    p_verify = sub.add_parser("verify", help="k-fold verification of models")
    p_verify.add_argument("--specimens", default=os.path.join(PATHS["data"], "specimens_master.csv"))
    p_verify.add_argument("--models", default="all", help="comma-separated or 'all'")
    p_verify.add_argument("--kfold", type=int, default=5)
    p_verify.add_argument("--out", default=PATHS["verify"])

    # ----------------------------------------------------------------------
    # Analyze
    # ----------------------------------------------------------------------
    p_an = sub.add_parser("analyze", help="generate analysis plots and reports")
    p_an.add_argument("--specimens", default=os.path.join(PATHS["data"], "specimens_master.csv"))
    p_an.add_argument("--fits", default=os.path.join(PATHS["model"], "fits.csv"))
    p_an.add_argument("--strain", default=os.path.join(PATHS["strain_energy"], "strain_energy_stats.csv"))
    p_an.add_argument("--metrics", default=os.path.join(PATHS["verify"], "metrics.csv"))
    p_an.add_argument("--params", default=os.path.join(PATHS["model"], "specimen_params_long.csv"))
    p_an.add_argument("--out", default=PATHS["report"])

    # ----------------------------------------------------------------------
    # Dispatch
    # ----------------------------------------------------------------------
    args = p.parse_args()
    ensure_dirs()

    if args.cmd == "fit":
        from ..models.core import available_models
        df = load_material_data(args.specimens)
        models_to_fit = (
            available_models() if args.model.lower() == "all" else [args.model]
        )
        ranges = [r.strip() for r in args.ranges.split(",")]

        rows = []
        for idx, r in df.iterrows():
            x_full, y_full = r["Stretch"], r["Stress"]
            for range_name in ranges:
                lo, hi = STRAIN_ENERGY_RANGES[range_name]
                mask = (x_full >= lo) & (x_full <= hi)
                x, y = x_full[mask], y_full[mask]
                for model_name in models_to_fit:
                    res = fit_curve(x, y, model_name)
                    rows.append(dict(
                        SpecimenID=r.get("SpecimenID", idx),
                        GroupName=r.get("GroupName", None),
                        Region=r.get("Region", None),
                        Range=range_name,   # 🔑 NEW COLUMN
                        Model=res.model,
                        Params=res.params,
                        RMSE=res.rmse,
                        NRMSE=res.nrmse,
                        MSE = res.mse,
                        R2=res.r2,
                    ))

        out_df = pd.DataFrame(rows)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        out_file = Path(args.out) / "fits.csv"
        out_df.to_csv(out_file, index=False)
        print(f"Wrote {out_file}")

        # --- Compute average goodness-of-fit summary ---
        summary_df = (
            out_df.groupby(["Model", "Region", "Range"])
                .agg(
                    RMSE_mean=("RMSE", "mean"),
                    RMSE_sd=("RMSE", "std"),
                    NRMSE_mean=("NRMSE", "mean"),
                    NRMSE_sd=("NRMSE", "std"),
                    MSE_mean=("MSE", "mean"),
                    MSE_sd=("MSE", "std"),
                    R2_mean=("R2", "mean"),
                    R2_sd=("R2", "std"),
                    n=("SpecimenID", "count"),
                )
                .reset_index()
        )

        # Save one file per Region × Range
        for (region, rng), sub_df in summary_df.groupby(["Region", "Range"]):
            fname = f"goodness_of_fit_{region}_{rng}.csv"
            fpath = Path(args.out) / fname
            sub_df.to_csv(fpath, index=False)
            print(f"Wrote {fpath}")

    elif args.cmd == "ingest":
        from ..data.io import ingest_xlsx_to_csvs
        path = ingest_xlsx_to_csvs(args.proximal, args.distal, args.thickness, args.out)
        print(f"Wrote master CSV and per-specimen curves under {args.out}: {path}")

    elif args.cmd == "summarize":
        fits = pd.read_csv(args.fits)
        strain_df = pd.read_csv(args.strain)
        metrics_df = pd.read_csv(args.metrics)
        specimens_df = pd.read_csv(args.specimens)

        # --- Parse Params column ---
        if "Params" in fits.columns:
            import ast
            def parse(v):
                if isinstance(v, str):
                    try:
                        return ast.literal_eval(v)
                    except Exception:
                        return v
                return v
            fits["Params"] = fits["Params"].apply(parse)

        # --- Per specimen/group summaries ---
        long_tbl = specimen_level_param_table(fits)
        stats_tbl = per_group_stats(long_tbl)

        # --- Build base master ---
        master = build_master_table(strain_df, fits, metrics_df)

        # --- Enrich with specimens metadata ---
        master = enrich_master_with_specimens(master, specimens_df)

        # --- Write augmented outputs ---
        Path(args.out).mkdir(parents=True, exist_ok=True)
        long_tbl.to_csv(Path(args.out) / "specimen_params_long.csv", index=False)
        stats_tbl.to_csv(Path(args.out) / "group_param_stats.csv", index=False)
        master_file = Path(args.out) / "master_augmented.csv"
        master.to_csv(master_file, index=False)

        print(f"Wrote specimen_params_long.csv, group_param_stats.csv, and master_augmented.csv to {args.out}")

        # === NEW: run statistical analysis on master_augmented ===
        try:
            from ..analysis import stats as mtstats   # <-- place the functions I sketched in analysis/stats.py

            # filter to deterministic models
            deterministic = master[master["NRMSE"].notna() & (master["NRMSE"] <= 0.10)]

            for region in deterministic["Region"].dropna().unique():
                for rng in deterministic["Range"].dropna().unique():
                    # --- A) SE vs covariates ---
                    se_eff = mtstats.analyze_se(deterministic, region, rng)
                    if not se_eff.empty:
                        se_eff.to_csv(Path(args.out)/f"se_effects_{region}_{rng}.csv", index=False)
                    se_boot = mtstats.bootstrap_stability(deterministic, mtstats.analyze_se, 
                                                        n_boot=500, region=region, range_name=rng)
                    if not se_boot.empty:
                        se_boot.to_csv(Path(args.out)/f"se_effects_boot_{region}_{rng}.csv", index=False)

                    # --- B,C) parameters & coupling, per model ---
                    for model in deterministic["Model"].dropna().unique():
                        par_eff = mtstats.analyze_params(deterministic, region, rng, model)
                        if not par_eff.empty:
                            par_eff.to_csv(Path(args.out)/f"param_effects_{region}_{rng}_{model}.csv", index=False)
                        couple = mtstats.analyze_se_param_coupling(deterministic, region, rng, model)
                        if not couple.empty:
                            couple.to_csv(Path(args.out)/f"se_param_coupling_{region}_{rng}_{model}.csv", index=False)

                        # bootstraps for stability
                        par_boot = mtstats.bootstrap_stability(deterministic, mtstats.analyze_params, 
                                                            n_boot=500, region=region, range_name=rng, model=model)
                        if not par_boot.empty:
                            par_boot.to_csv(Path(args.out)/f"param_effects_boot_{region}_{rng}_{model}.csv", index=False)
                        couple_boot = mtstats.bootstrap_stability(deterministic, mtstats.analyze_se_param_coupling, 
                                                                n_boot=500, region=region, range_name=rng, model=model)
                        if not couple_boot.empty:
                            couple_boot.to_csv(Path(args.out)/f"se_param_coupling_boot_{region}_{rng}_{model}.csv", index=False)

            print(f"Statistical analyses written to {args.out}")
        except Exception as e:
            print(f"[WARN] Statistical analysis skipped due to error: {e}")
        
        summarize_coupling_results(input_dir=args.out)

    elif args.cmd == "strain-energy":
        df = load_material_data(args.specimens)
        se = compute_strain_energy(df)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        out_file = Path(args.out) / "strain_energy_stats.csv"
        se.to_csv(out_file, index=False)
        print(f"Wrote {out_file}")

    elif args.cmd == "verify":
        import mtlab.models.core as core
        df = load_material_data(args.specimens)
        if args.models.lower() == "all":
            model_list = core.available_models()
        else:
            model_list = [m.strip() for m in args.models.split(",")]
        metrics = kfold_verify(df, model_list, k=args.kfold)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        metrics.to_csv(Path(args.out) / "metrics.csv", index=False)
        trust = trustworthy_models(metrics)
        trust.to_csv(Path(args.out) / "trustworthy_models.csv", index=False)
        print(f"Wrote verification metrics and trustworthy_models to {args.out}")

    elif args.cmd == "analyze":
        from ..analysis.reporting import (
            plot_stress_stretch,
            plot_strain_energy,
            summarize_model_metrics,
            plot_model_params,
            overlay_model_fits,
            plot_stress_stretch_summary,
            # plot_fit_metrics
        )

        specimens = pd.read_csv(args.specimens)
        # --- Normalize control naming ---
        if "GroupName" in specimens.columns:
            specimens["GroupName"] = (
                specimens["GroupName"]
                .astype(str)
                .apply(lambda x: "Control" if str(x).strip().lower().startswith("control") else x)
            )
        fits = pd.read_csv(args.fits)
        strain = pd.read_csv(args.strain)
        metrics = pd.read_csv(args.metrics)
        params = pd.read_csv(args.params)

        # Run the plots
        plot_stress_stretch(specimens, args.out)
        plot_stress_stretch_summary(
            specimens,
            args.out,
            model_name="holz_iso",
            )
        overlay_model_fits(specimens, fits, args.out)
        plot_strain_energy(strain, args.out)
        # plot_fit_metrics(metrics, args.out)
        plot_model_params(params, args.out, model="holz_iso")
