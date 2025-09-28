# src/mtlab/fitting/core.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Sequence
from scipy.optimize import minimize
from .utils import rmse, nrmse
from ..models.core import get_model, stress_fn

@dataclass
class FitResult:
    model: str
    params: Dict[str, float]
    rmse: float
    nrmse: float

def fit_curve(stretch, stress, model: str, x0: Sequence[float]=None) -> FitResult:
    stretch = np.asarray(stretch, dtype=float).reshape(-1)
    stress  = np.asarray(stress,  dtype=float).reshape(-1)
    if stretch.shape != stress.shape:
        raise ValueError(f"stretch/stress shape mismatch: {stretch.shape} vs {stress.shape}")
    m = np.isfinite(stretch) & np.isfinite(stress)
    if not np.all(m):
        stretch, stress = stretch[m], stress[m]

    mod = get_model(model)
    fn   = stress_fn(model)              # ← derive σ(λ;θ) from W(λ;θ)
    bounds = mod.get("bounds", None)
    pnames = mod["params"]

    if x0 is None:
        x0 = [1.0 for _ in pnames]

    def loss(theta):
        yhat = fn(stretch, *theta)
        return np.mean((yhat - stress)**2)

    res = minimize(loss, x0, bounds=bounds, method="L-BFGS-B")
    theta = res.x
    yhat = fn(stretch, *theta)
    return FitResult(
        model=model,
        params={k: float(v) for k, v in zip(pnames, theta)},
        rmse=float(rmse(stress, yhat)),
        nrmse=float(nrmse(stress, yhat)),
    )
