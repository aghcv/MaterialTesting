import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_strain_energy(stats_csv: str, out_dir: str):
    df = pd.read_csv(stats_csv)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Simple boxplot per group (customize as needed)
    if "GroupName" in df.columns:
        ax = df.boxplot(column="StrainEnergy", by="GroupName", grid=False, rot=45)
        plt.title("Strain Energy by Group")
        plt.suptitle("")
        plt.ylabel("Strain Energy (AUC)")
        plt.tight_layout()
        plt.savefig(str(Path(out_dir)/"strain_energy_by_group.png"), dpi=200)
        plt.close()
