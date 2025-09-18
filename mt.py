# -*- coding: utf-8 -*-
"""
Material Testing Analysis & Visualization
Refactored for clarity and compactness.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import mean_squared_error, roc_curve

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def export_figure(fig, filename, dpi=600):
    fig.savefig(filename, format='png', dpi=dpi)
    print(f"Saved: {filename}")

# ----------------------------------------------------------------------
# Material Models
# ----------------------------------------------------------------------
def neo_hookean(lmbda, G): return (G / 2) * (lmbda**2 - 1 / lmbda**2)

def yeoh(lmbda, *mu): 
    return sum((mu[i] / len(mu)) * (lmbda**2 - 3)**((i+1)/2) for i in range(len(mu)))

def mooney_rivlin(lmbda, *c):
    I1 = lmbda**2 + lmbda**-2 - 2
    return sum(ci * I1**(i+1) for i, ci in enumerate(c))

def holzapfel(lmbda, kappa, beta1, delta, beta2):
    return (kappa/2)*(np.exp(beta1*(lmbda**2-3))-1) + (delta/2)*(np.exp(beta2*(lmbda**2-3))-1)

MODEL_FUNCS = {
    "Neo-Hookean": (neo_hookean, [1.0]),
    "Yeoh": (yeoh, [1.0, 1.0]),
    "Mooney-Rivlin-2": (lambda l,c1,c2: mooney_rivlin(l,c1,c2), [1.0,1.0]),
    "Mooney-Rivlin-3": (lambda l,c1,c2,c3: mooney_rivlin(l,c1,c2,c3), [1.0,1.0,1.0]),
    "Holzapfel": (holzapfel, [1.0,1.0,1.0,1.0])
}

def fit_models(stretch, stress):
    """Fit multiple hyperelastic models."""
    results = {}
    for name, (func, guess) in MODEL_FUNCS.items():
        try:
            params, _ = curve_fit(func, stretch, stress, p0=guess)
            results[name] = (params, func(stretch, *params))
        except Exception as e:
            print(f"Fit failed for {name}: {e}")
    return results

# ----------------------------------------------------------------------
# Data Processing
# ----------------------------------------------------------------------
def smooth_and_energy(stretch, stress, n_points=30):
    """Smooth stress and compute strain energy curve."""
    stress_smooth = savgol_filter(stress, 15, 2, mode="mirror")
    x_new = np.linspace(1, 2, n_points)
    y_new = np.interp(x_new, stretch, stress_smooth)
    energy = np.array([trapezoid(y_new[:i], x_new[:i]) for i in range(len(y_new))])
    return x_new, y_new, energy

# ----------------------------------------------------------------------
# Machine Learning Comparison
# ----------------------------------------------------------------------
def compare_regressors(df, features, target, threshold=None):
    """Compare regression models on given features/target."""
    X = df[features]; y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    preproc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), features)])
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(), "Lasso": Lasso(), "ElasticNet": ElasticNet(),
        "KNN": KNeighborsRegressor(), "DecisionTree": DecisionTreeRegressor(),
        "SVR": SVR(), "RF": RandomForestRegressor(), "GB": GradientBoostingRegressor(),
        "XGB": XGBRegressor(), "LGBM": LGBMRegressor(), "CatBoost": CatBoostRegressor(verbose=0)
    }

    scores = {}
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)

    for i,(name, model) in enumerate(models.items()):
        pipe = Pipeline([("pre", preproc),("model", model)])
        pipe.fit(X_train,y_train); y_pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test,y_pred); scores[name]=mse
        plt.scatter(y_test,y_pred,label=name,alpha=0.6)
        if threshold is not None:
            roc_curve((y_test>=threshold).astype(int),(y_pred>=threshold).astype(int))

    lims=[min(y_test),max(y_test)]
    plt.plot(lims,lims,'k--'); plt.xlabel("Observed"); plt.ylabel("Predicted"); plt.title("(A) Predictions")
    plt.legend(fontsize="small")

    plt.subplot(1,2,2)
    plt.barh(list(scores.keys()),list(scores.values())); plt.xlabel("MSE"); plt.title("(B) Errors")
    plt.tight_layout(); plt.show()
    return scores

# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example fake data (replace with real import/merge step)
    stretch = np.linspace(1,2,50)
    stress = 100*(stretch-1)**2 + np.random.normal(0,5,50)

    # Fit material models
    results = fit_models(stretch, stress)
    for name,(params,fit) in results.items():
        print(f"{name}: {params}")

    # ML comparison (dummy dataframe)
    df = pd.DataFrame({
        "Severity": np.random.randint(0,3,100),
        "Duration": np.random.randint(0,3,100),
        "Weight": np.random.rand(100)*5,
        "Sex": np.random.randint(0,2,100),
        "neo": np.random.rand(100)*200
    })
    compare_regressors(df, ["Severity","Duration","Weight","Sex"], "neo", threshold=80)
