# mtlab/data/schema.py
import os

BASE_OUT = "outputs"

PATHS = {
    "data": os.path.join(BASE_OUT, "data"),
    "derived": os.path.join(BASE_OUT, "data", "derived"),
    "strain_energy": os.path.join(BASE_OUT, "data", "strain-energy"),
    "model": os.path.join(BASE_OUT, "model", "fits"),
    "verify": os.path.join(BASE_OUT, "verify"),
    "report": os.path.join(BASE_OUT, "report"),
}

# optional helpers
def ensure_dirs():
    for p in PATHS.values():
        os.makedirs(p, exist_ok=True)


PARAM_COLUMNS = [f"p{i}" for i in range(1, 7)]  # fixed width for all models

# Map model → which param goes into which pX
MODEL_PARAM_MAP = {
    # Simple models
    "neo": ["G"],
    "linear": ["E"],  # if you keep linear elastic fits

    # Mooney-Rivlin & Yeoh families
    "mr2": ["C10", "C01"],
    "mr3": ["C1", "C2", "C3"],  # matches your W definition
    "mr5": ["C10", "C01", "C11", "C20", "C02"],
    "yeoh2": ["C10", "C20"],
    "yeoh3": ["C10", "C20", "C30"],

    # Ogden reduced model
    "ogden1": ["mu", "alpha"],

    # Holzapfel variants
    "holz_iso": ["mu", "k1", "k2"],               # isotropic
    "holz_aniso4": ["mu", "k1", "k2", "theta"],   # 1 fiber family, 1 angle
    "holz_goh5": ["mu", "k1", "k2", "kappa", "theta"],  # Goh variant with dispersion

    # Fung model
    "fung": ["C", "b1", "b2", "b3"],  # common 4-parameter Fung form
}


# RGB tuples in [0,1] (reused from your existing palette)
GROUP_COLORS = {
    'Control': (0.0, 0.0, 0.0),
    'RdCoA5': (0.1, 0.6, 1.0),
    'RdCoA10': (0.1, 0.4, 0.8),
    'RdCoA20': (0.1, 0.3, 0.5),
    'dCoA5': (0.1, 0.9, 0.1),
    'dCoA10': (0.1, 0.7, 0.1),
    'dCoA20': (0.1, 0.5, 0.1),
    'CoA5': (1.0, 0.1, 0.1),
    'CoA10': (0.8, 0.1, 0.1),
    'CoA20': (0.5, 0.1, 0.1),
}

# Human-friendly labels (edit these to your preferred wording, e.g., “Mild-Short-CoA”)
GROUP_LABELS = {
    'Control': 'Control',
    'RdCoA5': 'Short-Mild-CoA',   # e.g., "Mild-Short-CoA"
    'RdCoA10': 'Short-Intermediate-CoA',
    'RdCoA20': 'Short-Severe-CoA',
    'dCoA5': 'Long-Mild-CoA',
    'dCoA10': 'Long-Intermediate-CoA',
    'dCoA20': 'Long-Severe-CoA',
    'CoA5': 'Prolong-Mild-CoA',
    'CoA10': 'Prolong-Intermediate-CoA',
    'CoA20': 'Prolong-Severe-CoA',
}

GROUP_ORDER = ['Control','RdCoA5','RdCoA10','RdCoA20','dCoA5','dCoA10','dCoA20','CoA5','CoA10','CoA20']

# Canonical column order for the master CSV
MASTER_COLUMNS = [
  'SpecimenID','Region','GroupName','GroupLabel','GroupColor',
  'RabbitNumber','Sex','Duration','BW','MaxDopplerGradient','EstimatedCatheter',
  'PT','DT','PL','DL','PW','DW','NPT','NDT',
  'Stretch','Stress'
]

# Standard stretch ranges for strain energy integration
STRAIN_ENERGY_RANGES = {
    "default": [1.0, 2.0],          # broad range, trimmed in code by specimen data
    "physiological": [1.41, 1.75],  # typical physiological range
    "diastolic": [1.34, 1.44],      # diastolic stretch window
    "systolic": [1.70, 1.80],       # systolic stretch window
}
