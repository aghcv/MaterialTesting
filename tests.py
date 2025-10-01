from pathlib import Path
import pandas as pd
from src.mtlab.analysis.constitutive_summary import summarize_coupling_results

# Define the input folder that contains all your
# `se_param_coupling_boot_<region>_<range>_<model>.csv` files
input_dir = Path("outputs/model/fits")   # adjust to your actual path

# Define where you want the summary written
output_file = "constitutive_coupling_reliability.csv"

# Run the summarizer
summary_df = summarize_coupling_results(input_dir, output_file)

# Inspect first few lines
print(summary_df.head())
