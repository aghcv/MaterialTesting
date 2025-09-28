import pandas as pd
from ..models.core import MODEL_REGISTRY

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
