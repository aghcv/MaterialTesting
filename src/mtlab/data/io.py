# src/mtlab/data/io.py
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, Optional
from .schema import GROUP_COLORS, GROUP_LABELS, GROUP_ORDER, MASTER_COLUMNS


# ---------------------------
# CSV loader (unit-agnostic)
# ---------------------------
def load_material_data(csv_path: str) -> pd.DataFrame:
    """
    Load a master CSV (specimen-level) that contains at least:
      - Stretch (list-like)
      - Stress  (list-like, unit-agnostic)
    Other recommended columns: SpecimenID, Region, GroupName, RabbitNumber, etc.
    """
    df = pd.read_csv(csv_path)
    # Coerce list-like strings to Python lists
    for col in ["Stretch", "Stress"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(_maybe_parse_list)
    return df


def _maybe_parse_list(x):
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast
                return list(ast.literal_eval(s))
            except Exception:
                return x
    return x

# ---------------------------
# XLSX ingestion pipeline
# ---------------------------

# Examples of supported sheet names:
#   "10_RdCoA_R02"  -> prefix=10, base=RdCoA, dur=None, rabbit=2  => GroupName 'RdCoA10'
#   "RdCoA10_R03"   -> prefix=None, base=RdCoA, dur=10, rabbit=3  => GroupName 'RdCoA10'
#   "Control_R01"   -> GroupName 'Control'
SHEET_RE = re.compile(
    r"^(?:(?P<prefix>\d+)_)?(?P<base>[A-Za-z]+(?:CoA)?)?(?P<dur>\d+)?_?R(?P<rabbit>\d+)$"
)


def parse_sheet_name(name: str):
    """
    Parse a sheet/tab name into (prefix_duration, base, dur_suffix, rabbit).
    Returns: (prefix: Optional[int], base: str, dur: Optional[int], rabbit: Optional[int])
    """
    name = name.strip()
    m = SHEET_RE.match(name)
    if not m:
        # Fallback parser for odd cases
        parts = name.split("_")
        prefix = None
        base = None
        dur = None
        rabbit = None
        for p in parts:
            if p.startswith("R") and p[1:].isdigit():
                rabbit = int(p[1:])
            elif p.isdigit():
                if prefix is None:
                    prefix = int(p)
                else:
                    dur = int(p)
            else:
                base = p
        return prefix, (base or ""), dur, rabbit

    gd = m.groupdict()
    prefix = int(gd["prefix"]) if gd["prefix"] else None
    base = gd["base"] or ""
    dur = int(gd["dur"]) if gd["dur"] else None
    rabbit = int(gd["rabbit"]) if gd["rabbit"] else None
    return prefix, base, dur, rabbit


def compose_group_name(prefix: Optional[int], base: str, dur: Optional[int]) -> str:
    """
    Build canonical GroupName from parsed parts.
    If base already ends with a number (e.g., 'RdCoA10'), return as-is.
    Else, append dur or prefix if available.
    """
    base = (base or "").strip()
    m = re.match(r"^(.*?)(\d+)$", base)
    if m:
        return base  # already has trailing duration
    if dur is not None:
        return f"{base}{dur}"
    if prefix is not None and base:
        return f"{base}{prefix}"
    return base or "Control"


def _sheet_to_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expect two columns (no headers) in each sheet:
      col0 = Stretch, col1 = Stress  (unit-agnostic)
    """
    if df.shape[1] < 2:
        raise ValueError("Sheet must have at least two columns: stretch, stress")
    a = df.iloc[:, 0].astype(float).to_numpy()
    b = df.iloc[:, 1].astype(float).to_numpy()
    return a, b


def read_region_workbook(xlsx_path: str, region: str) -> pd.DataFrame:
    """
    Read an XLSX with multiple tabs; each tab is one specimen's stress–stretch curve for a given region.
    Returns columns:
      SpecimenID, Region, GroupName, RabbitNumber, Stretch, Stress
    """
    all_sheets = pd.read_excel(xlsx_path, sheet_name=None, header=None)
    rows = []
    for sheet, df in all_sheets.items():
        try:
            prefix, base, dur, rabbit = parse_sheet_name(str(sheet))
            group = compose_group_name(prefix, base, dur)
            x, y = _sheet_to_xy(df)
            specimen = f"{group}_R{rabbit:02d}_{region}"
            rows.append(
                dict(
                    SpecimenID=specimen,
                    Region=region,
                    GroupName=group,
                    RabbitNumber=rabbit,
                    Stretch=list(map(float, x.tolist())),
                    Stress=list(map(float, y.tolist())),  # ← unit-agnostic
                )
            )
        except Exception:
            # Skip malformed sheet names or non-numeric rows
            continue
    return pd.DataFrame(rows)


def read_thickness_xlsx(xlsx_path: str) -> pd.DataFrame:
    """
    Read thickness/metadata table with a 'Name' column like:
      '01_Control_R01' or '03_RdCoA10_R01'
    Adds GroupName, RabbitNumber (and Duration if absent).
    """
    df = pd.read_excel(xlsx_path, header=0)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    group = []
    rabbit = []
    duration = []
    for v in df.get("Name", []):
        if isinstance(v, str):
            parts = v.split("_")
            base = None
            dur = None
            rbt = None
            for p in parts:
                if p.startswith("R") and p[1:].isdigit():
                    rbt = int(p[1:])
                elif p.isdigit():
                    # leading index like "01_", ignore
                    pass
                else:
                    base = p
                    m = re.match(r"^(.*?)(\d+)$", base)
                    if m:
                        base, dur = m.group(1), int(m.group(2))
            gname = base + (str(dur) if dur is not None and not re.search(r"\d+$", base) else "")
            group.append(gname)
            rabbit.append(rbt)
            duration.append(dur)
        else:
            group.append(None)
            rabbit.append(None)
            duration.append(None)

    df["GroupName"] = group
    df["RabbitNumber"] = rabbit
    if "Duration" not in df.columns:
        df["Duration"] = duration

    # Normalize 'Estimated Catheter...' → 'EstimatedCatheter'
    for c in list(df.columns):
        if c.lower().startswith("estimated") and "catheter" in c.lower():
            df = df.rename(columns={c: "EstimatedCatheter"})
            break

    return df

def merge_specimens_with_thickness(spec_df: pd.DataFrame, thick_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge proximal/distal specimens with thickness/meta by GroupName + RabbitNumber.
    Adds GroupLabel, GroupColor, and a Region-specific 'Thickness' column:
      - proximal → PT
      - distal   → DT
    """
    merged = pd.merge(
        spec_df, thick_df, on=["GroupName", "RabbitNumber"], how="left", suffixes=("", "_meta")
    )

    # Labels & colors for plotting
    merged["GroupLabel"] = merged["GroupName"].map(GROUP_LABELS).fillna(merged["GroupName"])
    merged["GroupColor"] = merged["GroupName"].map(GROUP_COLORS).apply(
        lambda c: list(c) if isinstance(c, tuple) else c
    )

    # Region-specific thickness
    def pick_thickness(row):
        region = str(row.get("Region", "")).lower()
        if region.startswith("prox"):
            return row.get("PT", np.nan)
        if region.startswith("dist"):
            return row.get("DT", np.nan)
        return np.nan

    merged["Thickness"] = merged.apply(pick_thickness, axis=1)

    # Ensure canonical columns exist (unit-agnostic: 'Stress', not 'Stress_kPa')
    for c in MASTER_COLUMNS:
        if c not in merged.columns:
            merged[c] = np.nan

    # Reorder columns (keep canonical first)
    cols = [c for c in MASTER_COLUMNS if c in merged.columns] + [
        c for c in merged.columns if c not in MASTER_COLUMNS
    ]
    return merged[cols]

def write_master_and_per_specimen_csvs(master_df: pd.DataFrame, out_dir: str):
    """
    Write a master CSV and individual per-specimen curve CSVs (Stretch, Stress).
    """
    p = Path(out_dir)
    (p / "specimens").mkdir(parents=True, exist_ok=True)

    master_file = p / "specimens_master.csv"
    master_df.to_csv(master_file, index=False)

    for _, r in master_df.iterrows():
        sid = r["SpecimenID"]
        if not isinstance(sid, str):
            continue
        df = pd.DataFrame({"Stretch": r["Stretch"], "Stress": r["Stress"]})
        (p / "specimens" / f"{sid}.csv").parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p / "specimens" / f"{sid}.csv", index=False)

def ingest_xlsx_to_csvs(proximal_xlsx: str, distal_xlsx: str, thickness_xlsx: str, out_dir: str) -> str:
    """
    High-level: read proximal/distal workbooks + thickness sheet, merge, and emit CSVs.
    Returns the path to the master CSV.
    """
    prox = read_region_workbook(proximal_xlsx, "proximal")
    dist = read_region_workbook(distal_xlsx, "distal")
    spec = pd.concat([prox, dist], ignore_index=True)
    thick = read_thickness_xlsx(thickness_xlsx)
    master = merge_specimens_with_thickness(spec, thick)
    write_master_and_per_specimen_csvs(master, out_dir)
    return str(Path(out_dir) / "specimens_master.csv")
