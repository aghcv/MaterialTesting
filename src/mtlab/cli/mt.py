import argparse, json
from pathlib import Path
import pandas as pd
from ..data.io import load_material_data
from ..analysis.strain_energy import compute_strain_energy
from ..analysis.constitutive_summary import specimen_level_param_table, per_group_stats, expand_param_column
from ..fitting.core import fit_curve
from ..verification.validate import kfold_verify, trustworthy_models

def main():
    p = argparse.ArgumentParser(prog="mt")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="fit a model per specimen")
    p_fit.add_argument("--data", required=True)
    p_fit.add_argument("--model", required=True)
    p_fit.add_argument("--out", required=True)

    p_sum = sub.add_parser("summarize", help="summarize fitted parameters")
    p_sum.add_argument("--fits", required=True)
    p_sum.add_argument("--out", required=True)

    p_se = sub.add_parser("strain-energy", help="compute strain energy per specimen")
    p_se.add_argument("--data", required=True)
    p_se.add_argument("--out", required=True)

    p_verify = sub.add_parser("verify", help="k-fold verification of models")
    p_verify.add_argument("--data", required=True)
    p_verify.add_argument("--models", default="all", help="comma-separated or 'all'")
    p_verify.add_argument("--kfold", type=int, default=5)
    p_verify.add_argument("--out", required=True)

    p_ing = sub.add_parser("ingest", help="ingest proximal/distal/thickness xlsx into standardized CSVs")
    p_ing.add_argument("--proximal", required=True)
    p_ing.add_argument("--distal", required=True)
    p_ing.add_argument("--thickness", required=True)
    p_ing.add_argument("--out", required=True)

    p_an = sub.add_parser("analyze", help="generate analysis plots and reports")
    p_an.add_argument("--specimens", required=True, help="path to specimens_master.csv")
    p_an.add_argument("--fits", required=True, help="path to fits.csv (model results)")
    p_an.add_argument("--strain", required=True, help="path to strain_energy_stats.csv")
    p_an.add_argument("--metrics", required=True, help="path to metrics.csv from verify")
    p_an.add_argument("--params", required=True, help="path to specimen_params_long.csv")
    p_an.add_argument("--out", required=True)

    args = p.parse_args()

    if args.cmd == "fit":
        from ..models.core import available_models  # ← add this import here
        df = load_material_data(args.data)
        models_to_fit = (
            available_models() if args.model.lower() == "all" else [args.model]
        )
        rows = []
        for idx, r in df.iterrows():
            x, y = r["Stretch"], r["Stress"]  # unchanged pipeline: Stress only
            for model_name in models_to_fit:
                res = fit_curve(x, y, model_name)
                rows.append(dict(
                    SpecimenID=r.get("SpecimenID", idx),
                    GroupName=r.get("GroupName", None),
                    Region=r.get("Region", None),
                    Model=res.model,
                    Params=res.params,
                    RMSE=res.rmse,
                    NRMSE=res.nrmse
                ))
        out_df = pd.DataFrame(rows)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        out_file = Path(args.out)/"fits.csv"
        out_df.to_csv(out_file, index=False)
        print(f"Wrote {out_file}")
    
    elif args.cmd == "ingest":
        from ..data.io import ingest_xlsx_to_csvs
        path = ingest_xlsx_to_csvs(args.proximal, args.distal, args.thickness, args.out)
        print(f"Wrote master CSV and per-specimen curves under {args.out}: {path}")

    elif args.cmd == "summarize":
        fits = pd.read_csv(args.fits)
        # Try to parse Params dict
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

        long_tbl = specimen_level_param_table(fits)
        stats_tbl = per_group_stats(long_tbl)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        long_tbl.to_csv(Path(args.out)/"specimen_params_long.csv", index=False)
        stats_tbl.to_csv(Path(args.out)/"group_param_stats.csv", index=False)
        print(f"Wrote specimen_params_long.csv and group_param_stats.csv to {args.out}")

    elif args.cmd == "strain-energy":
        df = load_material_data(args.data)
        se = compute_strain_energy(df)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        se.to_csv(Path(args.out)/"strain_energy_stats.csv", index=False)
        print(f"Wrote strain_energy_stats.csv to {args.out}")

    elif args.cmd == "verify":
        import mtlab.models.core as core
        df = load_material_data(args.data)
        if args.models.lower() == "all":
            model_list = core.available_models()
        else:
            model_list = [m.strip() for m in args.models.split(",")]
        metrics = kfold_verify(df, model_list, k=args.kfold)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        metrics.to_csv(Path(args.out)/"metrics.csv", index=False)
        trust = trustworthy_models(metrics)
        trust.to_csv(Path(args.out)/"trustworthy_models.csv", index=False)
        print(f"Wrote verification metrics and trustworthy_models to {args.out}")
    
    elif args.cmd == "analyze":
        from ..analysis.reporting import (
            plot_stress_stretch,
            plot_strain_energy,
            summarize_model_metrics,
            plot_model_params,
            overlay_model_fits,
            #plot_fit_metrics
        )

        specimens = pd.read_csv(args.specimens)
        fits = pd.read_csv(args.fits)
        strain = pd.read_csv(args.strain)
        metrics = pd.read_csv(args.metrics)
        params = pd.read_csv(args.params)

        # Run the plots
        plot_stress_stretch(specimens, args.out)          # raw curves
        overlay_model_fits(specimens, fits, args.out)     # raw + fit overlay
        plot_strain_energy(strain, args.out)              # boxplots
        #plot_fit_metrics(metrics, args.out)               # model performance
        plot_model_params(params, args.out, model="holz_iso")