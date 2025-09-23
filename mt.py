# -*- coding: utf-8 -*-
"""
Material Testing Analysis with Hyperelastic Fits, Group Summaries, and Regression
Outputs are saved into ./mt_out/
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, approx_fprime
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import r2_score
from numdifftools import Hessian

# ----------------------------------------------------------------------
# Suppress warnings
# ----------------------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------
# Constitutive Models
# ----------------------------------------------------------------------

# ---- Invariant helpers (uniaxial, incompressible) --------------------
def I1_uniaxial(lmbda):
    """
    First invariant for incompressible uniaxial extension:
    λ1 = λ, λ2 = λ3 = λ^(-1/2)
    """
    return lmbda**2 + 2.0 / lmbda

def I2_uniaxial(lmbda):
    """
    Second invariant for incompressible uniaxial extension:
    I2 = λ^2 λ2^2 + λ2^2 λ3^2 + λ3^2 λ1^2
    With λ2 = λ3 = λ^(-1/2):
    I2 = 2λ + 1/λ^2
    """
    return 2.0 * lmbda + 1.0 / (lmbda**2)

def I4_uniaxial(lmbda, alpha_rad):
    """
    Fourth invariant for a single fiber family a0 making angle alpha
    with the loading axis.
    For incompressible uniaxial: C = diag(λ^2, λ^-1, λ^-1).
    """
    lam_t = lmbda**(-0.5)  # transverse stretch
    c, s = np.cos(alpha_rad), np.sin(alpha_rad)
    return (lmbda**2) * (c**2) + (lam_t**2) * (s**2)

# ---- Constitutive models ---------------------------------------------
def neo_hookean(lmbda, G):
    I1 = I1_uniaxial(lmbda)
    return 0.5 * G * (I1 - 3.0)

def yeoh(lmbda, *C):
    I1 = I1_uniaxial(lmbda)
    return sum(C[i] * (I1 - 3.0)**(i + 1) for i in range(len(C)))

def mooney_rivlin_2(lmbda, C10, C01):
    """
    2-parameter Mooney–Rivlin:
    W = C10*(I1 - 3) + C01*(I2 - 3)
    """
    I1, I2 = I1_uniaxial(lmbda), I2_uniaxial(lmbda)
    return C10 * (I1 - 3.0) + C01 * (I2 - 3.0)

def mooney_rivlin_3(lmbda, C1, C2, C3):
    """
    3-term polynomial Mooney–Rivlin extension
    (often written as sum of higher powers of I1 - 3).
    """
    I1 = I1_uniaxial(lmbda)
    return C1 * (I1 - 3.0) + C2 * (I1 - 3.0)**2 + C3 * (I1 - 3.0)**3

def mooney_rivlin_5(lmbda, C10, C01, C11, C20, C02):
    """
    5-parameter Mooney–Rivlin (commonly used in biomechanics):
    W = C10*(I1-3) + C01*(I2-3) + C11*(I1-3)(I2-3)
        + C20*(I1-3)^2 + C02*(I2-3)^2
    """
    I1, I2 = I1_uniaxial(lmbda), I2_uniaxial(lmbda)
    return (C10 * (I1 - 3.0) +
            C01 * (I2 - 3.0) +
            C11 * (I1 - 3.0) * (I2 - 3.0) +
            C20 * (I1 - 3.0)**2 +
            C02 * (I2 - 3.0)**2)

def holzapfel_isotropic(lmbda, mu, k1, k2):
    """
    Holzapfel isotropic part:
    W = (μ/2)*(I1-3) + (k1/(2k2))*(exp(k2*(I1-3)^2) - 1)
    """
    I1 = I1_uniaxial(lmbda)
    term = I1 - 3.0
    return 0.5 * mu * term + (k1 / (2.0 * k2)) * (np.exp(k2 * term**2) - 1.0)

def holzapfel_anisotropic(lmbda, mu, k1, k2, alpha_rad, kappa):
    """
    Holzapfel–Gasser–Ogden (HGO) model with fiber dispersion:
    W = (μ/2)*(I1-3) + (k1/(2k2))*(exp(k2*Ef^2) - 1)
    Ef = κ*(I1-3) + (1-3κ)*(I4-1)
    """
    I1 = I1_uniaxial(lmbda)
    I4 = I4_uniaxial(lmbda, alpha_rad)
    Ef = kappa * (I1 - 3.0) + (1.0 - 3.0 * kappa) * (I4 - 1.0)
    W_iso = 0.5 * mu * (I1 - 3.0)
    W_fib = (k1 / (2.0 * k2)) * (np.exp(k2 * Ef**2) - 1.0)
    return W_iso + W_fib

def linear_elastic(lmbda, E):
    """
    Simple linear elastic stress law:
    σ = E (λ - 1)
    """
    return E * (lmbda - 1.0)

def linear_elastic_free(lmbda, E, sigma0):
    """
    Linear elastic with free intercept: σ = E*(λ - 1) + σ0
    """
    return E * (lmbda - 1.0) + sigma0

# ----------------------------------------------------------------------
# Derivative helpers
# ----------------------------------------------------------------------
def numerical_derivative(func, x, *params, eps=1e-6):
    return approx_fprime([x], lambda z: func(z[0], *params), [eps])[0]

def stress_from_W(W_func, lmbda, *params):
    dW_dlambda = numerical_derivative(W_func, lmbda, *params)
    return (1.0 / lmbda) * dW_dlambda

# ----------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------
MODEL_FUNCS = {
    "Neo-Hookean": {
        "W": neo_hookean,
        "guess": [10.0],
        "bounds": [(1e-6, 1e4)]
    },

    "Yeoh": {
        "W": yeoh,
        "guess": [1.0, 0.1],
        "bounds": [(1e-6, 1e3), (1e-6, 1e3)]
    },

    "Mooney-Rivlin-2": {
        "W": mooney_rivlin_2,
        "guess": [1.0, 0.1],                  # [C10, C01]
        "bounds": [(1e-6, 1e3), (1e-6, 1e3)]
    },

    "Mooney-Rivlin-3": {
        "W": mooney_rivlin_3,
        "guess": [1.0, 0.1, 0.01],            # [C1, C2, C3]
        "bounds": [(1e-6, 1e3), (1e-6, 1e3), (1e-6, 1e3)]
    },

    "Mooney-Rivlin-5": {
        "W": mooney_rivlin_5,
        "guess": [1.0, 0.1, 0.05, 0.01, 0.01], # [C10, C01, C11, C20, C02]
        "bounds": [
            (1e-6, 1e3),  # C10
            (1e-6, 1e3),  # C01
            (1e-6, 1e3),  # C11
            (1e-6, 1e3),  # C20
            (1e-6, 1e3)   # C02
        ]
    },

    "Holzapfel-Isotropic": {
        "W": holzapfel_isotropic,
        "guess": [1.0, 1.0, 1.0],             # [mu, k1, k2]
        "bounds": [(1e-6, 1e3), (1e-6, 1e3), (1e-6, 100.0)]
    },

    "Holzapfel-Anisotropic": {
        "W": holzapfel_anisotropic,
        "guess": [1.0, 1.0, 1.0, np.deg2rad(30.0), 0.05], # [mu, k1, k2, alpha_rad, kappa]
        "bounds": [
            (1e-6, 1e3),    # mu
            (1e-6, 1e3),    # k1
            (1e-6, 100.0),  # k2
            (0.0, np.pi/2), # alpha in radians
            (0.0, 1.0/3.0)  # kappa dispersion
        ]
    },

    "Linear-Elastic": {
        "stress": linear_elastic,
        "guess": [100.0],
        "bounds": [(1e-3, 1e5)]
    },

    "Linear-Elastic-Free": {
        "stress": linear_elastic_free,
        "guess": [100.0, 0],
        "bounds": [(1e-3, 1e5), (-1e3, 1e3)]
    }
}

# ----------------------------------------------------------------------
# Optimization
# ----------------------------------------------------------------------
def objective(params, model_name, x, y, obj_type="NRMSE"):
    model = MODEL_FUNCS[model_name]
    if model_name in ["Linear-Elastic", "Linear-Elastic-Free"]:
        stress_model = model["stress"](x,*params)
    else:
        W_func = model["W"]
        stress_model = np.array([stress_from_W(W_func,l,*params) for l in x])
    residuals = stress_model - y
    if obj_type=="SSE": return np.sum(residuals**2)
    if obj_type=="MSE": return np.mean(residuals**2)
    if obj_type=="NRMSE": return np.sqrt(np.mean(residuals**2))/(np.max(y)-np.min(y)+1e-12)
    if obj_type=="Relative": return np.sum(((residuals)/(y+1e-12))**2)
    raise ValueError(f"Unknown obj {obj_type}")

def fit_model(model_name, x, y, method="local", obj_type="NRMSE"):
    guess,bounds = MODEL_FUNCS[model_name]["guess"], MODEL_FUNCS[model_name]["bounds"]
    if method=="genetic":
        res = differential_evolution(objective,bounds,args=(model_name,x,y,obj_type),
                                     strategy="best1bin",maxiter=500,tol=1e-6)
    else:
        res = minimize(objective,guess,args=(model_name,x,y,obj_type),
                       method="L-BFGS-B",bounds=bounds)
    return res.x, res.fun

def evaluate_fits(df_region, region="proximal", out_file="fit_metrics.csv"):
    """
    Collect fitted parameters and fit quality metrics for all models.
    Loads from df_region (already has *_params columns).
    Saves a CSV with one row per specimen per model.
    """
    rows = []
    for idx, row in df_region.iterrows():
        x, y = row["Stretch"], row["Stress"]
        if x is None or y is None or len(x) < 3:
            continue
        for m in MODEL_FUNCS:
            params = row.get(f"{m}_params", None)
            if params is None or isinstance(params, float):
                continue
            params = np.atleast_1d(params)

            # Predict stress
            if m in ["Linear-Elastic", "Linear-Elastic-Free"]:
                y_pred = MODEL_FUNCS[m]["stress"](x, *params)
            else:
                Wf = MODEL_FUNCS[m]["W"]
                y_pred = np.array([stress_from_W(Wf, l, *params) for l in x])

            residuals = y_pred - y
            mse = np.mean(residuals**2)
            nrmse = np.sqrt(mse) / (np.max(y) - np.min(y) + 1e-12)
            r2 = r2_score(y, y_pred)

            # Confidence intervals (approximate if Hessian is available)
            ci_lower, ci_upper = [], []
            try:
                # Numerical Hessian via finite differences
                H = Hessian(lambda p: objective(p, m, x, y))(params)
                cov = np.linalg.inv(H)
                se = np.sqrt(np.diag(cov))
                ci_lower = params - 1.96 * se
                ci_upper = params + 1.96 * se
            except Exception:
                ci_lower = [np.nan] * len(params)
                ci_upper = [np.nan] * len(params)

            rows.append({
            "Region": region,
            "SpecimenID": idx,
            "GroupName": row["GroupName"],
            "RabbitNumber": row["RabbitNumber"],
            "Model": m,
            "Params": params.tolist(),
            "MSE": mse,
            "NRMSE": nrmse,
            "R2": r2,
            "CI_lower": ci_lower,
            "CI_upper": ci_upper
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_file, index=False)
    print(f"📊 Saved fit metrics to {out_file}")
    return metrics

# ----------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------
def load_material_data(proximal_file, distal_file, thickness_file):
    files={"proximal":proximal_file,"distal":distal_file}
    cols=['GroupName','GroupNumber','RabbitNumber','Region','Stretch','Stress']
    df=pd.DataFrame(columns=cols)
    for region,file in files.items():
        xls=pd.ExcelFile(file)
        for sheet in xls.sheet_names:
            data=xls.parse(sheet)
            gnum,gname,rnum=sheet.split('_')
            new_row={"GroupName":gname,"GroupNumber":int(gnum),
                     "RabbitNumber":int(rnum[1:].split('.')[0]),
                     "Region":region,"Stretch":data.iloc[:,0].values,
                     "Stress":data.iloc[:,1].values}
            df=pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)
    tdf=pd.read_excel(thickness_file)
    tdf[['GroupNumber','GroupName','RabbitNumber']] = tdf['Name'].str.split('_',expand=True)
    tdf['GroupNumber']=tdf['GroupNumber'].str[1:].astype(int)
    tdf['RabbitNumber']=tdf['RabbitNumber'].str[1:].astype(int)
    merged=pd.merge(df,tdf,on=["GroupNumber","RabbitNumber"])
    df["Thickness"]=merged.apply(lambda r:r["NPT"] if r["Region"]=="proximal" else r["NDT"],axis=1)
    df["Weight"]=merged["BW"]
    df["Sex"]=merged["Sex"].map({"M":0,"F":1})
    return df

def add_severity_duration(df):
    sev,dur=[],[]
    for name in df["GroupName"]:
        if "CoA" in name:
            severity=5 if name.endswith("5") else 10 if name.endswith("10") else 20
            duration=1 if name.startswith("RdCoA") else 3 if name.startswith("dCoA") else 22
        else: severity,duration=0,22
        sev.append(severity); dur.append(duration)
    df["Severity"],df["Duration"]=sev,dur
    return df

# ----------------------------------------------------------------------
# Fitting wrapper
# ----------------------------------------------------------------------
def fit_all_models_for_df(df, region="proximal", fit_range=None,
                          method="local", verbose=True, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        print(f"🔄 Loading cached fits from {cache_file}")
        return pd.read_pickle(cache_file)
    df_region=df[df["Region"]==region].copy()
    for m in MODEL_FUNCS: df_region[f"{m}_params"]=None
    for idx,row in df_region.iterrows():
        x,y=row["Stretch"],row["Stress"]
        if x is None or y is None: continue
        if fit_range is not None:
            mask=(x>=fit_range[0])&(x<=fit_range[1]); x,y=x[mask],y[mask]
        if len(x)<3: continue
        for m in MODEL_FUNCS:
            try:
                if verbose: print(f"Row {idx} → fitting {m}")
                t0=time.time()
                p,loss=fit_model(m,x,y,method=method)
                df_region.at[idx,f"{m}_params"]=p
                if verbose: print(f"   done {m} in {time.time()-t0:.2f}s | loss={loss:.3e}")
            except Exception as e:
                if verbose: print(f"   ⚠️ {m} fail: {e}")
                df_region.at[idx,f"{m}_params"]=None
    if cache_file:
        df_region.to_pickle(cache_file); print(f"💾 Saved fits to {cache_file}")
    return df_region

# ----------------------------------------------------------------------
# Group averages + plots
# ----------------------------------------------------------------------
def group_average(df_group, stretch_points=None):
    if stretch_points is None: stretch_points=np.linspace(1.0,2.0,20)
    all_y=[]
    for _,row in df_group.iterrows():
        x,y=row["Stretch"],row["Stress"]
        if x is None or y is None or len(x)<2: continue
        y_interp=np.interp(stretch_points,x,y)
        all_y.append(y_interp)
    if not all_y: return stretch_points,np.zeros_like(stretch_points),np.zeros_like(stretch_points)
    all_y=np.array(all_y)
    mean_y=np.nanmean(all_y,axis=0); std_y=np.nanstd(all_y,axis=0)
    se_y=std_y/np.sqrt(all_y.shape[0])
    return stretch_points,mean_y,se_y

def fit_group_average_curve(model_name, stretch_points, stress_points, fit_range=None):
    if fit_range is not None:
        mask=(stretch_points>=fit_range[0])&(stretch_points<=fit_range[1])
        stretch_points,stress_points=stretch_points[mask],stress_points[mask]
    try:
        params,_=fit_model(model_name,stretch_points,stress_points)
        return params
    except Exception as e:
        print(f"⚠️ group fit fail {model_name}: {e}"); return None

def plot_stress_stretch_summary(df, control="Control", model_type="Neo-Hookean",
                                region="proximal", fit_range=None, out_file=None, 
                                exp_alpha=1.0):
    df_region = df[df["Region"]==region]
    if df_region.empty:
        return

    if out_file is None:
        safe_model=model_type.replace(" ","_").replace("-","")
        safe_region=region.lower()
        out_file=f"plot_{safe_model}_{safe_region}.png"

    group_colors = {
        "Control": (0, 0, 0),          # black
        "RdCoA5": (0.1, 0.6, 1),     # light blue
        "RdCoA10": (0.1, 0.4, 0.8),    # medium blue
        "RdCoA20": (0.1, 0.3, 0.5),    # dark blue
        "dCoA5": (0.1, 0.9, 0.1),        # light green
        "dCoA10": (0.1, 0.7, 0.1),     # medium green
        "dCoA20": (0.1, 0.5, 0.1),     # dark green
        "CoA5": (1, 0.1, 0.1),         # light red
        "CoA10": (0.8, 0.1, 0.1),      # medium red
        "CoA20": (0.5, 0.1, 0.1)       # dark red
    }

    group_order=["RdCoA5","RdCoA10","RdCoA20",
                 "dCoA5","dCoA10","dCoA20",
                 "CoA5","CoA10","CoA20"]

    fig,axes=plt.subplots(3,3,figsize=(12,10))
    # inside plot_stress_stretch_summary
    if fit_range is not None:
        fine_stretch = np.linspace(fit_range[0], fit_range[1], 50)
    else:
        fine_stretch = np.linspace(1.0, 2.1, 110)
    
    if fit_range is not None:
        exp_alpha = 0.2

    # Control curve
    ctrl_x,ctrl_y,ctrl_se=group_average(df_region[df_region["GroupName"]==control])
    ctrl_params=fit_group_average_curve(model_type,ctrl_x,ctrl_y,fit_range)

    for i,g in enumerate(group_order):
        if g not in df_region["GroupName"].unique():
            continue
        r,c=divmod(i,3)
        ax=axes[r,c]

        gx,gy,gse=group_average(df_region[df_region["GroupName"]==g])
        color=group_colors.get(g,"gray")

        # group data + fit
        ax.errorbar(gx,gy,yerr=gse,fmt="s",color=color,ecolor=color,
                    capsize=3,label=f"{g} data", alpha = exp_alpha)

        gp=fit_group_average_curve(model_type,gx,gy,fit_range)
        if gp is not None:
            if model_type in ["Linear-Elastic", "Linear-Elastic-Free"]:
                gf=MODEL_FUNCS[model_type]["stress"](fine_stretch,*gp)
            else:
                Wf=MODEL_FUNCS[model_type]["W"]
                gf=[stress_from_W(Wf,l,*gp) for l in fine_stretch]
            ax.plot(fine_stretch,gf,"--",color=color,alpha=0.8,label=f"{g} fit")

        # control data + fit in black
        ax.errorbar(ctrl_x,ctrl_y,yerr=ctrl_se,fmt="o",color="black",
                    ecolor="black",capsize=3,label="Control data", alpha=exp_alpha)
        if ctrl_params is not None:
            if model_type in ["Linear-Elastic", "Linear-Elastic-Free"]:
                cf=MODEL_FUNCS[model_type]["stress"](fine_stretch,*ctrl_params)
            else:
                Wf=MODEL_FUNCS[model_type]["W"]
                cf=[stress_from_W(Wf,l,*ctrl_params) for l in fine_stretch]
            ax.plot(fine_stretch,cf,"--",color="black",alpha=0.8,label="Control fit")

        ax.set_ylim(0,800)
        if r==2: ax.set_xlabel("Stretch λ")
        if c==0: ax.set_ylabel("Stress [kPa]")
        ax.set_title(g)
        ax.legend(fontsize="x-small")

    plt.tight_layout()
    plt.savefig(out_file,dpi=300)
    plt.close(fig)
    print(f"📈 Saved {out_file}")

# ----------------------------------------------------------------------
# Parameter summary
# ----------------------------------------------------------------------
def summarize_all_parameters(df, region="proximal", group_col="GroupName", out_file="param_summary.csv"):
    df_region=df[df["Region"]==region]
    rows=[]
    for m in MODEL_FUNCS:
        pc=f"{m}_params"
        if pc not in df_region: continue
        d=df_region.dropna(subset=[pc]).copy()
        if d.empty: continue
        non_null=d[pc].dropna()
        if len(non_null)==0: continue
        max_len=max(len(np.atleast_1d(p)) for p in non_null)
        for i in range(max_len):
            d[f"param_{i+1}"]=d[pc].apply(lambda x: np.atleast_1d(x)[i] if x is not None and len(np.atleast_1d(x))>i else np.nan)
            groups=[g.dropna().values for _,g in d.groupby(group_col)[f"param_{i+1}"]]
            try: _,pval=f_oneway(*groups)
            except: pval=np.nan
            try:
                tukey=pairwise_tukeyhsd(endog=d[f"param_{i+1}"],groups=d[group_col],alpha=0.05)
                tukey_df=pd.DataFrame(tukey._results_table.data[1:],columns=tukey._results_table.data[0])
            except: tukey_df=pd.DataFrame()
            gs=d.groupby(group_col)[f"param_{i+1}"].agg(['mean','std','count'])
            gs['se']=gs['std']/np.sqrt(gs['count'])
            for grp,v in gs.iterrows():
                rows.append({"Model":m,"Parameter":f"p{i+1}","Group":grp,
                             "Mean±SE":f"{v['mean']:.3f} ± {v['se']:.3f}",
                             "ANOVA p":f"{pval:.3e}" if grp==gs.index[0] else ""})
            if not tukey_df.empty:
                for _,r in tukey_df.iterrows():
                    rows.append({"Model":m,"Parameter":f"p{i+1}",
                                 "Group":f"{r['group1']} vs {r['group2']}",
                                 "Mean±SE":"","ANOVA p":f"Tukey p={r['p-adj']:.3e}"})
    summary=pd.DataFrame(rows); summary.to_csv(out_file,index=False)
    print(f"📊 Saved param stats to {out_file}")
    return summary

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__=="__main__":
    outdir="mt_out"
    os.makedirs(outdir,exist_ok=True)

    df=load_material_data("proximal.xlsx","distal.xlsx","Thickness_Normalized.xlsx")
    df=add_severity_duration(df)
    physiological_range=[1.41,1.75]
    diastolic_range = [1.34, 1.44]
    systolic_range = [1.70, 1.8]
    regions=["proximal","distal"]
    models = [
        "Linear-Elastic",
        "Linear-Elastic-Free",
        "Neo-Hookean",
        "Yeoh",
        "Mooney-Rivlin-2",
        "Mooney-Rivlin-3",
        "Mooney-Rivlin-5",         
        "Holzapfel-Isotropic",
        "Holzapfel-Anisotropic"    
    ]

    for region in regions:
        cache_file=os.path.join(outdir,f"fits_{region}.pkl")
        df_region=fit_all_models_for_df(df,region=region,cache_file=cache_file)
        summarize_all_parameters(df_region,region=region,
                                 out_file=os.path.join(outdir,f"param_summary_{region}.csv"))
        evaluate_fits(df_region, region=region,
                    out_file=os.path.join(outdir, f"fit_metrics_{region}.csv"))

        for m in models:
            # Full-range fit/plot
            plot_stress_stretch_summary(df_region,model_type=m,region=region,
                                        out_file=os.path.join(outdir,f"plot_{m}_{region}.png"))
            # Physiological-range fit/plot
            plot_stress_stretch_summary(df_region,model_type=m,region=region,
                                        fit_range=physiological_range,
                                        out_file=os.path.join(outdir,f"plot_{m}_{region}_phys.png"))
            plot_stress_stretch_summary(df_region,model_type=m,region=region,
                                        fit_range=diastolic_range,
                                        out_file=os.path.join(outdir,f"plot_{m}_{region}_diastol.png"))
            plot_stress_stretch_summary(df_region,model_type=m,region=region,
                                        fit_range=systolic_range,
                                        out_file=os.path.join(outdir,f"plot_{m}_{region}_systol.png"))
