


import numpy as np
from typing import Dict, List

# ---------------------------
# Uniaxial, incompressible kinematics
# principal stretches: (λ, λ^{-1/2}, λ^{-1/2})
# ---------------------------
def _arr(x): return np.asarray(x, dtype=float)

def I1_uniaxial(lmbda):
    l = _arr(lmbda)
    return l**2 + 2.0 / l

def I2_uniaxial(lmbda):
    l = _arr(lmbda)
    return (1.0 / l**2) + 2.0 * l

def I4_uniaxial(lmbda, theta_rad):
    """I4 for a fiber family at angle theta (radians) to the loading axis (in-plane)."""
    l = np.asarray(lmbda, dtype=float)
    c2 = np.cos(theta_rad)**2
    s2 = np.sin(theta_rad)**2
    return (l**2) * c2 + (l**-1) * s2

# -----------------------------------
# Strain-energy density models: W(λ)
# (Unit-agnostic; return W with same units as input stress scale.)
# -----------------------------------

def neo_hookean_W(lmbda, G):
    I1 = I1_uniaxial(lmbda)
    return 0.5 * G * (I1 - 3.0)

def yeoh_W(lmbda, *C):
    I1 = I1_uniaxial(lmbda)
    t = (I1 - 3.0)
    # Your original style: sum C[i] * t**(i+1)
    return sum(C[i] * t**(i + 1) for i in range(len(C)))

def mooney_rivlin_2_W(lmbda, C10, C01):
    I1 = I1_uniaxial(lmbda)
    I2 = I2_uniaxial(lmbda)
    return C10 * (I1 - 3.0) + C01 * (I2 - 3.0)

def mooney_rivlin_3_W(lmbda, C1, C2, C3):
    I1 = I1_uniaxial(lmbda)
    t = (I1 - 3.0)
    return C1 * t + C2 * t**2 + C3 * t**3

def mooney_rivlin_5_W(lmbda, C10, C01, C11, C20, C02):
    I1 = I1_uniaxial(lmbda)
    I2 = I2_uniaxial(lmbda)
    t1 = (I1 - 3.0)
    t2 = (I2 - 3.0)
    return (C10 * t1 +
            C01 * t2 +
            C11 * t1 * t2 +
            C20 * t1**2 +
            C02 * t2**2)

def holzapfel_isotropic_W(lmbda, mu, k1, k2):
    I1 = I1_uniaxial(lmbda)
    t = (I1 - 3.0)
    # numeric guard: avoid overflow in exp
    return 0.5 * mu * t + (k1 / (2.0 * k2)) * (safe_exp(k2 * t**2) - 1.0)

def holzapfel_anisotropic_W(lmbda, mu, k1, k2, theta_rad):
    """
    W = 0.5*mu*(I1-3) + 2 * [ k1/(2*k2) * (exp(k2*(I4-1)^2) - 1) ]
    - '2 * [...]' accounts for ±theta symmetric families with same parameters.
    - theta_rad is in radians.
    """
    I1 = I1_uniaxial(lmbda)
    I4 = I4_uniaxial(lmbda, theta_rad)
    iso = 0.5 * mu * (I1 - 3.0)
    fib = (k1 / (2.0 * k2)) * (safe_exp(k2 * (I4 - 1.0)**2) - 1.0)
    return iso + 2.0 * fib

def holzapfel_GOH_W(lmbda, mu, k1, k2, theta_rad, kappa):
    """
    W = 0.5*mu*(I1-3) + 2 * [ k1/(2*k2) * (exp(k2 * Ef^2) - 1) ]
    Ef = kappa*(I1-3) + (1-3*kappa)*(I4-1)
    - Two symmetric families at ±theta → factor 2.
    - theta_rad in radians, kappa in [0, 1/3).
    """
    I1 = I1_uniaxial(lmbda)
    I4 = I4_uniaxial(lmbda, theta_rad)
    Ef = kappa * (I1 - 3.0) + (1.0 - 3.0*kappa) * (I4 - 1.0)
    iso = 0.5 * mu * (I1 - 3.0)
    fib = (k1 / (2.0 * k2)) * (safe_exp(k2 * (Ef**2)) - 1.0)
    return iso + 2.0 * fib

def reduced_ogden_W(lmbda, mu, alpha=2.0):
    l = _arr(lmbda)
    # W = (2μ/α^2)(λ^α + 2 λ^{-α/2} - 3)
    return (2.0 * mu / (alpha**2)) * (l**alpha + 2.0 * l**(-alpha / 2.0) - 3.0)

def fung_exponential_W(lmbda, c):
    I1 = I1_uniaxial(lmbda)
    t = (I1 - 3.0)
    return 0.5 * c * (safe_exp(t) - 1.0)

def safe_exp(x, cap: float = 50.0):
    """Numerically safe exp: clip argument to avoid overflow."""
    return np.exp(np.clip(x, -cap, cap))

# -----------------------------------
# Generic stress builder from W
# -----------------------------------
def sigma_from_W(W_fn, kind: str = "cauchy", eps: float = 1e-6):
    """
    Turn any W(lmbda, *theta) into a stress function sigma(lmbda, *theta).
    Uses central differences: dW/dλ ≈ [W(λ+h)-W(λ-h)]/(2h), with relative step h=eps*max(1,|λ|).
    Uniaxial, incompressible:
        nominal P = dW/dλ
        Cauchy σ  = λ * P
    kind: "cauchy" or "nominal"
    """
    def sigma(lmbda, *theta):
        l = _arr(lmbda)
        h = eps * np.maximum(1.0, np.abs(l))
        Wp = W_fn(l + h, *theta)
        Wm = W_fn(l - h, *theta)
        dWdl = (Wp - Wm) / (2.0 * h)
        P = dWdl
        return l * P if kind == "cauchy" else P
    return sigma


# ---------------------------
# Registry (W-only)
# ---------------------------
MODEL_REGISTRY: Dict[str, Dict] = {
    "neo":      {"W": neo_hookean_W,       "params": ["G"],                            "bounds": [(1e-9, 1e9)]},
    "yeoh2":    {"W": lambda l,C10,C20: yeoh_W(l, C10, C20),
                 "params": ["C10","C20"],                                             "bounds": [(1e-9,1e9)]*2},
    "yeoh3":    {"W": lambda l,C10,C20,C30: yeoh_W(l, C10, C20, C30),
                 "params": ["C10","C20","C30"],                                       "bounds": [(1e-9,1e9)]*3},
    "mr2":      {"W": mooney_rivlin_2_W,   "params": ["C10","C01"],                    "bounds": [(1e-9,1e9)]*2},
    "mr3":      {"W": mooney_rivlin_3_W,   "params": ["C1","C2","C3"],                 "bounds": [(1e-9,1e9)]*3},
    "mr5":      {"W": mooney_rivlin_5_W,   "params": ["C10","C01","C11","C20","C02"],  "bounds": [(1e-9,1e9)]*5},
    # keep k2 modest to prevent exp overflow; adjust if your units demand larger
    "holz_iso": {"W": holzapfel_isotropic_W,"params": ["mu","k1","k2"],                "bounds": [(1e-9,1e9),(1e-9,1e9),(1e-6,2.0)]},
    "ogden1":   {"W": reduced_ogden_W,     "params": ["mu","alpha"],                   "bounds": [(1e-9,1e9),(0.2,10.0)]},
}

MODEL_REGISTRY.update({
    "fung": {
        "params": ["c"],
        "bounds": [(1e-9, 1e4)],
        "W": fung_exponential_W,
    },
    "holz_aniso4": {  # θ as a fitted parameter (radians)
        "params": ["mu", "k1", "k2", "theta"],
        "bounds": [(1e-9, 1e3), (1e-9, 1e4), (1e-6, 50.0), (0.0, np.pi/2)],
        "W": holzapfel_anisotropic_W,
    },
    "holz_goh5": {   # includes dispersion kappa
        "params": ["mu", "k1", "k2", "theta", "kappa"],
        "bounds": [(1e-9, 1e3), (1e-9, 1e4), (1e-6, 50.0), (0.0, np.pi/2), (0.0, 0.3333)],
        "W": holzapfel_GOH_W,
    },
})

def get_model(name: str) -> Dict:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]

def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())

def stress_fn(name: str, kind: str = "cauchy", eps: float = 1e-4):
    """
    Prefer analytic σ(λ) if available; otherwise derive from W via central differences.
    """
    # fallback to generic numeric derivative from W
    W = get_model(name)["W"]
    return sigma_from_W(W, kind=kind, eps=eps)
