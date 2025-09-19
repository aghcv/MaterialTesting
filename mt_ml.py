# mt-ml.py
# ----------------------------------------------------------------------
# Machine Learning analysis of constitutive parameters
# ----------------------------------------------------------------------

import os
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys

from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------------------
# Suppress noisy warnings & model logs
# ----------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# LightGBM & XGBoost log suppression
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)

# CatBoost log suppression
# By default, CatBoost prints lots of messages unless verbose=0 is set.
# We set verbose=0 in model creation already.

# Silence stdout for specific libs if needed
class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass

def silence_stdout_stderr():
    sys.stdout = DummyFile()
    sys.stderr = DummyFile()

def restore_stdout_stderr():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# ----------------------------------------------------------------------
# Core ML analysis
# ----------------------------------------------------------------------
def run_ml_analysis(df, features, target, out_file_prefix, drop_control=True):
    """
    Train/test ML regressors to predict target from features.
    Optionally drops Control group.
    Saves plots, scores, and feature contributions.
    """
    df_clean = df.dropna(subset=features+[target])
    if drop_control and "GroupName" in df_clean.columns:
        df_clean = df_clean[df_clean["GroupName"] != "Control"]

    if df_clean.empty:
        print(f"⚠️ No data available for target={target} after filtering")
        return None

    # --- Debug/summary feedback ---
    print(f"\n🔎 Running standard ML analysis for target: {target}")
    print(f"   Features: {features}")
    print(f"   Clean dataset shape: {df_clean.shape}")
    if drop_control:
        print("   Control group excluded.")
    print("   Sample rows:\n", df_clean[features+[target]].head(), "\n")

    X, y = df_clean[features], df_clean[target]

    # Skip if target has no variance
    if y.nunique() < 2:
        print(f"⚠️ Skipping target={target} because all values are the same.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Training set size: {X_train.shape[0]} rows")
    print(f"   Test set size: {X_test.shape[0]} rows")
    print("   Train target head:\n", y_train.head())
    print("   Test target head:\n", y_test.head(), "\n")

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), features)
    ])

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "KNN": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
        "SVR": SVR(),
        "RF": RandomForestRegressor(),
        "GB": GradientBoostingRegressor(),
        "XGB": XGBRegressor(verbosity=0, use_label_encoder=False),
        "LGBM": LGBMRegressor(verbose=-1),
        "CatBoost": CatBoostRegressor(verbose=0),
    }

    scores = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for name, model in models.items():
        try:
            pipe = Pipeline([("pre", preproc), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            contrib = np.zeros(len(features))
            if hasattr(pipe.named_steps["model"], "coef_"):
                contrib = pipe.named_steps["model"].coef_
            elif hasattr(pipe.named_steps["model"], "feature_importances_"):
                contrib = pipe.named_steps["model"].feature_importances_

            scores.append({
                "Model": name,
                "MSE": mse,
                "R2": r2,
                **{f"{f}_importance": c for f, c in zip(features, contrib)}
            })

            axes[0].scatter(y_test, y_pred, label=name, alpha=0.6, s=20)

        except Exception as e:
            print(f"   ⚠️ Skipping model {name} due to error: {e}")

    if not scores:
        print(f"⚠️ No models successfully trained for target={target}")
        return None

    lims = [min(y_test), max(y_test)]
    axes[0].plot(lims, lims, 'k--')
    axes[0].set_title("(A) Predictions")
    axes[0].legend(fontsize="x-small")

    scores_df = pd.DataFrame(scores).sort_values("MSE")
    axes[1].barh(scores_df["Model"], scores_df["MSE"])
    axes[1].set_title("(B) Errors")

    plt.tight_layout()
    fig.savefig(f"{out_file_prefix}_ml.png", dpi=300)
    print(f"📊 Saved ML comparison plot: {out_file_prefix}_ml.png")

    scores_df.to_csv(f"{out_file_prefix}_ml_scores.csv", index=False)
    print(f"📊 Saved ML scores: {out_file_prefix}_ml_scores.csv")

    return scores_df

def run_ml_analysis_loocv(df, features, target, out_file_prefix, drop_controls=True):
    """
    Leave-One-Out CV analysis for predicting target from features.
    Returns aggregated performance across LOOCV folds, including feature importances.
    """
    df_clean = df.dropna(subset=features+[target]).copy()

    # Optionally drop controls
    if drop_controls and "GroupName" in df_clean.columns:
        df_clean = df_clean[df_clean["GroupName"].str.lower() != "control"]

    if df_clean.empty or df_clean[target].nunique() < 2:
        print(f"⚠️ Skipping LOOCV for target={target} (not enough variance in target)")
        return None

    # --- Debug/summary feedback ---
    print(f"\n🔎 Running LOOCV ML analysis for target: {target}")
    print(f"   Features: {features}")
    print(f"   Clean dataset shape: {df_clean.shape}")
    print("   Sample rows:\n", df_clean[features+[target]].head(), "\n")

    X, y = df_clean[features].values, df_clean[target].values
    loo = LeaveOneOut()

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "KNN": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
        "SVR": SVR(),
        "RF": RandomForestRegressor(),
        "GB": GradientBoostingRegressor(),
        "XGB": XGBRegressor(verbosity=0, use_label_encoder=False),
        "LGBM": LGBMRegressor(verbose=-1),
        "CatBoost": CatBoostRegressor(verbose=0),
    }

    scores, importances = [], []

    for name, model in models.items():
        y_true, y_pred = [], []
        skipped_folds = 0
        fold_importances = []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Skip folds with no variance in training target
            if len(np.unique(y_train)) < 2:
                skipped_folds += 1
                continue

            try:
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)
                y_true.append(y_test[0])
                y_pred.append(y_hat[0])

                # Collect feature importances/coeffs if available
                if hasattr(model, "feature_importances_"):
                    fold_importances.append(model.feature_importances_)
                elif hasattr(model, "coef_"):
                    coefs = np.ravel(model.coef_)
                    if coefs.shape[0] == len(features):
                        fold_importances.append(coefs)

            except Exception as e:
                skipped_folds += 1
                continue

        # Evaluate metrics
        if len(y_true) > 1:
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
        else:
            mse, r2 = np.nan, np.nan

        scores.append({
            "Model": name,
            "MSE": mse,
            "R2": r2,
            "SkippedFolds": skipped_folds
        })

        # Average feature importances if available
        if fold_importances:
            avg_importance = np.mean(fold_importances, axis=0)
            for feat, val in zip(features, avg_importance):
                importances.append({
                    "Model": name,
                    "Feature": feat,
                    "Importance": val
                })

    # Save results
    scores_df = pd.DataFrame(scores).sort_values("MSE")
    scores_df.to_csv(f"{out_file_prefix}_ml_loocv_scores.csv", index=False)
    print(f"📊 Saved LOOCV ML scores: {out_file_prefix}_ml_loocv_scores.csv")

    if importances:
        importances_df = pd.DataFrame(importances)
        importances_df.to_csv(f"{out_file_prefix}_ml_loocv_importances.csv", index=False)
        print(f"📊 Saved LOOCV feature importances: {out_file_prefix}_ml_loocv_importances.csv")
    else:
        importances_df = pd.DataFrame()

    return scores_df, importances_df

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main(fits, outdir, regions):
    """
    Run ML pipeline for given fits and regions.
    - fits: list of pickle files (one per region)
    - outdir: output directory
    - regions: list of region names (e.g., ["proximal", "distal"])
    """
    os.makedirs(outdir, exist_ok=True)

    for fit_file, region in zip(fits, regions):
        print(f"\n🔍 Loading fit results for region={region} from {fit_file}")
        with open(fit_file, "rb") as f:
            df_region = pickle.load(f)

        # Build feature matrix
        features = ["Severity", "Duration", "Weight", "Sex"]

        # ---------------------------------------------------
        # 1) Run ML for constitutive parameters
        # ---------------------------------------------------
        for col in df_region.columns:
            if col.endswith("_params"):
                model_name = col.replace("_params", "")
                params_series = df_region[col].dropna()
                if params_series.empty:
                    continue

                max_len = max(len(np.atleast_1d(p)) for p in params_series)
                for i in range(max_len):
                    param_name = f"{model_name}_p{i+1}"
                    df_region[param_name] = df_region[col].apply(
                        lambda x: np.atleast_1d(x)[i] if x is not None and len(np.atleast_1d(x)) > i else np.nan
                    )

                    # Standard ML CV
                    out_prefix = os.path.join(outdir, f"{region}_{param_name}")
                    run_ml_analysis(df_region, features, param_name, out_prefix)

                    # Leave-One-Out CV
                    run_ml_analysis_loocv(df_region, features, param_name, out_prefix)

        # ---------------------------------------------------
        # 2) Run ML for Thickness (original analysis)
        # ---------------------------------------------------
        if "Thickness" in df_region.columns:
            out_prefix = os.path.join(outdir, f"{region}_Thickness")
            run_ml_analysis(df_region, features, "Thickness", out_prefix)
            run_ml_analysis_loocv(df_region, features, "Thickness", out_prefix)

# ----------------------------------------------------------------------
# CLI wrapper
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML analysis on constitutive parameter fits")
    parser.add_argument("--fits", nargs="+", required=True, help="List of fit cache pickle files")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--regions", nargs="+", required=True, help="List of regions corresponding to fits")
    args = parser.parse_args()

    main(args.fits, args.outdir, args.regions)
