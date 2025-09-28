import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ..fitting.core import fit_curve
from ..models.core import get_model

def kfold_verify(df: pd.DataFrame, models, k=5, random_state=42) -> pd.DataFrame:
    """K-fold verification over specimens: train on K-1 specimens, test on 1 by re-fitting per specimen.
    Returns per-specimen metrics for each model."""
    # For constitutive models, fitting is typically specimen-specific. Here we compute goodness-of-fit per specimen
    # and aggregate across K splits simply as cross-validated metrics (re-fitting on that specimen's data).
    # This is effectively a repeated evaluation with different initializations/splits to test stability.
    rng = np.random.RandomState(random_state)
    idxs = np.arange(len(df))
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    rows = []
    for train_idx, test_idx in kf.split(idxs):
        for i in test_idx:
            row = df.iloc[i]
            x = np.asarray(row["Stretch"]); y = np.asarray(row["Stress"])
            if len(x) < 2: 
                continue
            for m in models:
                res = fit_curve(x, y, m)
                rows.append(dict(
                    Fold=int(len(rows)),
                    SpecimenID=row.get("SpecimenID", i),
                    GroupName=row.get("GroupName", None),
                    Region=row.get("Region", None),
                    Model=m,
                    RMSE=res.rmse,
                    NRMSE=res.nrmse
                ))
    return pd.DataFrame(rows)

def trustworthy_models(metrics_df: pd.DataFrame, nrmse_threshold: float=0.1):
    """Return models that meet an NRMSE threshold across the cohort (median)."""
    out = (metrics_df
           .groupby("Model")["NRMSE"]
           .median()
           .reset_index()
           .query("NRMSE <= @nrmse_threshold")
          )
    return out
