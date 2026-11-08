"""Subgroup ROC-AUC of HGB-Lookback on the locked origins (Jun-Aug 2011), recomputed from the revision predictions.
Run after revision_analyses.py:  python src/evaluation/revision_subgroups.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.evaluation.revision_analyses import build_table  # noqa: E402

t = build_table(); t["Origin"] = t["SnapPeriod"].astype(str)
p = pd.read_csv(ROOT / "artifacts/revision/revision_all_origin_predictions.csv")
p = p[p["Model"] == "HGB-Lookback"].merge(t[["Origin", "Customer ID", "Frequency", "ObservedTenureDays", "PriorWindowInvoices"]],
                                          on=["Origin", "Customer ID"], validate="one_to_one")
lk = p[p["Origin"].isin(["2011-06", "2011-07", "2011-08"])].copy()
med = lk.groupby("Origin")["ObservedTenureDays"].transform("median")
groups = {"Tenure at or below origin median": lk["ObservedTenureDays"] <= med,
          "Tenure above origin median": lk["ObservedTenureDays"] > med,
          "1 invoice in window": lk["Frequency"] == 1, "2-3 invoices in window": lk["Frequency"].between(2, 3),
          "4+ invoices in window": lk["Frequency"] >= 4,
          "No invoice in prior window (true zero)": lk["PriorWindowInvoices"] == 0,
          "1+ invoice in prior window": lk["PriorWindowInvoices"] >= 1}
rows = []
for n, m in groups.items():
    a = [roc_auc_score(g["y"], g["p"]) for _, g in lk[m].groupby("Origin") if g["y"].nunique() == 2]
    rows.append({"Subgroup": n, "Rows": int(m.sum()), "MeanAUC_locked": float(np.mean(a))})
out = pd.DataFrame(rows); out.to_csv(ROOT / "artifacts/revision/revision_subgroups_locked.csv", index=False)
print(out.round(4).to_string(index=False))
