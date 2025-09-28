import numpy as np
import pandas as pd
from ..data.schema import STRAIN_ENERGY_RANGES

def compute_strain_energy(
    df: pd.DataFrame,
    stretch_points=None,
    ranges: dict = None
) -> pd.DataFrame:
    """
    Compute stored strain energy via trapezoid rule per specimen.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must include columns 'Stretch' and 'Stress'.
    stretch_points : array-like, optional
        Common grid to interpolate onto. If None, defaults to np.linspace(1.0, xmax, 300),
        where xmax = min(2.0, global max stretch across all specimens).
    ranges : dict, optional
        Override for stretch ranges. If None, defaults to STRAIN_ENERGY_RANGES in schema.
    
    Returns
    -------
    pd.DataFrame
        Columns: SpecimenID, GroupName, Region, RabbitNumber, Range, StrainEnergy
    """
    # Determine global xmax if not given
    if stretch_points is None:
        xmax_all = min(
            2.0,
            np.min([np.max(np.asarray(x)) for x in df["Stretch"] if isinstance(x, (list, np.ndarray))])
        )
        stretch_points = np.linspace(1.0, xmax_all, 300)

    # Use schema ranges unless overridden
    if ranges is None:
        ranges = STRAIN_ENERGY_RANGES

    rows = []
    for idx, row in df.iterrows():
        x, y = row.get("Stretch"), row.get("Stress")
        if x is None or y is None:
            continue
        x = np.asarray(x); y = np.asarray(y)
        if len(x) < 2 or len(y) < 2:
            continue

        y_interp = np.interp(stretch_points, x, y)

        for rname, (rmin, rmax) in ranges.items():
            xmin = max(np.min(x), rmin)
            xmax = min(np.max(x), rmax)
            mask = (stretch_points >= xmin) & (stretch_points <= xmax)
            if mask.sum() < 2:
                continue
            auc = np.trapz(y_interp[mask], stretch_points[mask])
            rows.append(dict(
                SpecimenID=row.get("SpecimenID", idx),
                GroupName=row.get("GroupName", None),
                Region=row.get("Region", None),
                RabbitNumber=row.get("RabbitNumber", None),
                Range=rname,
                StrainEnergy=float(auc)
            ))
    return pd.DataFrame(rows)
