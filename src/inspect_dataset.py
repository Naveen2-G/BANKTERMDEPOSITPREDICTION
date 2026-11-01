"""Inspect the downloaded UCI CSV before importing it."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "data" / "bank-full.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Place bank-full.csv at: {csv_path}")

df = pd.read_csv(csv_path, sep=";", keep_default_na=False)
print("Rows, columns:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst five records:\n", df.head().to_string(index=False))
print("\nDuplicate rows:", int(df.duplicated().sum()))
print("\nTarget distribution:\n", df["y"].value_counts(dropna=False))
