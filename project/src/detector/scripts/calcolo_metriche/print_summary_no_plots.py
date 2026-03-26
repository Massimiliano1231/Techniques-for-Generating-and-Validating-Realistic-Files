import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))


import pandas as pd
from config.constants import metricsForStampa

CSV_PATH = str(PROJECT_ROOT / "data" / "generator" / "csv" / "pairwise_real_vs_generated.csv")

df = pd.read_csv(CSV_PATH)


summary = df.groupby("format")[metricsForStampa].agg(["mean", "std"]).round(4) 

print("\n=== STATISTICHE PER FORMATO ===")
print(summary)
