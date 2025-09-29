import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ..data.schema import PARAM_COLUMNS

def fdr_bh(p):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, float)
    _, q, _, _ = multipletests(p, alpha=0.10, method="fdr_bh")
    return q

def _robust_fit(formula, df):
    # OLS with HC3 robust SE
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    return model

def dedupe_se(master):
    # one SE per SpecimenID×Range to avoid pseudo-replication
    keep_cols = [c for c in master.columns if c != "Model"]  # Model duplicates SE
    return (master
            .sort_values(["SpecimenID","Range"])  # stable
            .drop_duplicates(subset=["SpecimenID","Range"])
            [keep_cols])

def analyze_se(master, region, range_name):
    m = master[(master["Region"]==region) & (master["Range"]==range_name)]
    m = m.dropna(subset=["StrainEnergy","Severity","Sex","BW","Duration"])
    if m.empty:
        return pd.DataFrame()

    fit = _robust_fit("StrainEnergy ~ C(Severity) + Duration + C(Sex) + BW", m)
    rows = []
    for term in fit.params.index:
        if term == "Intercept":
            continue
        rows.append({
            "term": term,
            "beta": fit.params[term],
            "p": fit.pvalues[term],   # ensure we always return "p"
            "region": region,
            "range": range_name,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = fdr_bh(out["p"].values)
    return out

def _maybe_log(x):
    # log1p if strictly nonnegative and skewed
    if np.nanmin(x) >= 0 and (np.nanmax(x) / max(np.nanmedian(x),1e-9) > 10):
        return np.log1p(x), "log1p"
    return x, "raw"

def analyze_params(master, region, range_name, model):
    m = master[(master["Region"]==region) &
               (master["Range"]==range_name) &
               (master["Model"]==model)]
    m = m.dropna(subset=["Severity","Sex","BW","Duration"])
    if m.empty:
        return pd.DataFrame()

    outs = []
    for pcol in PARAM_COLUMNS:
        if pcol not in m.columns or m[pcol].notna().mean() < 0.3:
            continue
        x, tr = _maybe_log(m[pcol].values)
        df = m.assign(**{f"{pcol}__trans": x})
        fit = _robust_fit(f"{pcol}__trans ~ C(Severity) + Duration + C(Sex) + BW", df)

        term = f"{pcol}__trans"
        if term in fit.params.index:
            outs.append({
                "term": term,
                "beta": fit.params[term],
                "p": fit.pvalues[term],  # ensure "p" is present
                "region": region,
                "range": range_name,
                "model": model,
                "param": pcol,
                "transform": tr,
            })
    if not outs:
        return pd.DataFrame()

    out = pd.DataFrame(outs)
    out["q"] = fdr_bh(out["p"].values)
    return out

def analyze_se_param_coupling(master, region, range_name, model):
    m = master[(master["Region"] == region) &
               (master["Range"] == range_name) &
               (master["Model"] == model)]
    m = m.dropna(subset=["StrainEnergy", "Severity", "Duration"])
    if m.empty:
        return pd.DataFrame()

    rows = []
    for pcol in [c for c in PARAM_COLUMNS if c in m.columns]:
        # skip if too sparse
        if m[pcol].notna().mean() < 0.30:
            continue

        x, tr = _maybe_log(m[pcol].values)
        df = m.assign(**{f"{pcol}__trans": x})

        fit = _robust_fit(
            f"StrainEnergy ~ {pcol}__trans + C(Severity) + C(Duration)",
            df
        )

        term = f"{pcol}__trans"
        beta = fit.params[term] if term in fit.params.index else np.nan
        pval = fit.pvalues[term] if term in fit.pvalues.index else np.nan

        rows.append({
            "term": term,
            "beta": beta,
            "p": pval,
            "region": region,
            "range": range_name,
            "model": model,
            "param": pcol,
            "transform": tr,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # FDR across params for this (region, range, model)
    mask = out["p"].notna()
    if mask.any():
        out.loc[mask, "q"] = fdr_bh(out.loc[mask, "p"].values)
    else:
        out["q"] = np.nan

    return out

def bootstrap_stability(master, fn, n_boot=1000, random_state=0, **kwargs):
    rng = np.random.default_rng(random_state)

    # slice once here for speed/consistency
    m = master.copy()
    if "region" in kwargs:
        m = m[m["Region"] == kwargs["region"]]
    if "range_name" in kwargs:
        m = m[m["Range"] == kwargs["range_name"]]
    if "model" in kwargs:
        m = m[m["Model"] == kwargs["model"]]

    sids = m["SpecimenID"].dropna().unique()
    if len(sids) < 5:
        return pd.DataFrame()

    records = []
    for b in range(n_boot):
        boot_ids = rng.choice(sids, size=len(sids), replace=True)
        boot_df = pd.concat([m[m["SpecimenID"] == sid] for sid in boot_ids], ignore_index=True)

        try:
            res = fn(boot_df, **kwargs)
        except Exception:
            continue
        if res is None or res.empty:
            continue

        res = res.copy()
        res["boot"] = b

        # add sign if present
        if "beta" in res.columns:
            res["sign"] = np.sign(res["beta"])

        # robust significance flag (prefer q, else p, else 0)
        if "q" in res.columns and res["q"].notna().any():
            res["sig_q10"] = (res["q"] <= 0.10).astype(int)
        elif "p" in res.columns and res["p"].notna().any():
            res["sig_q10"] = (res["p"] <= 0.10).astype(int)
        else:
            res["sig_q10"] = 0

        records.append(res)

    if not records:
        return pd.DataFrame()

    boot = pd.concat(records, ignore_index=True)

    keys = [c for c in ["region", "range", "model", "param", "term"] if c in boot.columns]
    grp = boot.groupby(keys, dropna=False)
    summary = grp.agg(
        n_boots=("boot", "nunique"),
        support=("sig_q10", "mean"),
        sign_consistency=("sign", lambda s: s.eq(s.mode().iloc[0]).mean() if not s.empty else np.nan),
        beta_median=("beta", "median"),
        beta_lo=("beta", lambda x: np.nanpercentile(x, 2.5)),
        beta_hi=("beta", lambda x: np.nanpercentile(x, 97.5)),
    ).reset_index()

    summary["support"] = summary["support"] * 100.0
    summary["sign_consistency"] = summary["sign_consistency"] * 100.0
    return summary
