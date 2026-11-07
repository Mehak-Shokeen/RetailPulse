"""
Writes a small StockCode -> Description lookup so the dashboard can run
without the 114 MB data/processed/cleaned_transactions.csv (which is not on GitHub).

Run once from the project root:
    python scripts/make_product_mapping.py
Output: artifacts/association/product_mapping.csv  (commit this file)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/cleaned_transactions.csv", usecols=["StockCode", "Description"])
df = df.dropna(subset=["StockCode", "Description"])
df["StockCode"] = df["StockCode"].astype(str).str.strip()
df["Description"] = df["Description"].astype(str).str.strip()
m = df.drop_duplicates("StockCode", keep="first")          # same rule the app used before
out = ROOT / "artifacts/association/product_mapping.csv"
m.to_csv(out, index=False)
print(f"wrote {len(m):,} products to {out}")
