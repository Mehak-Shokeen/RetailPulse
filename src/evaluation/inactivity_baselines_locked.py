"""
Extra baselines (inactivity): simple baselines on the SAME locked origins as the real models.

Why: the notebook already compares single features on the development folds, but the
locked-origin table (the one that goes in the paper) has no simple baseline next to it.

Run from the project root:
    python src/evaluation/inactivity_baselines_locked.py

Needs: data/processed/cleaned_transactions.csv and
       artifacts/inactivity/inactivity_v2_locked_predictions.csv
Writes: artifacts/inactivity/inactivity_v2_locked_baselines.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.inactivity_features import (  # noqa: E402
    build_snapshot_features, build_labels, prepare_transactions)

LOCKED = ["2011-06", "2011-07", "2011-08"]
# Orientation fixed in advance (no fitting): rarely/long-ago buyers are more likely inactive.
BASELINES = {"Recency only": ("Recency", +1),
             "Frequency only": ("Frequency", -1),
             "Monetary only": ("Monetary", -1),
             "ActiveMonths only": ("ActiveMonths", -1)}
N_BOOT = 500


def main():
    raw = pd.read_csv(ROOT / "data/processed/cleaned_transactions.csv",
                      parse_dates=["InvoiceDate"],
                      dtype={"Invoice": "string", "StockCode": "string"})
    tx = prepare_transactions(raw)
    data_start = tx["InvoiceDate"].min().normalize().replace(day=1)

    parts = []
    for o in LOCKED:
        p = pd.Period(o, "M")
        f = build_snapshot_features(tx, p.end_time, data_start=data_start)
        f["Inactive"] = build_labels(tx, p.end_time, f["Customer ID"])
        f["Origin"] = o
        parts.append(f)
    feats = pd.concat(parts, ignore_index=True)

    preds = pd.read_csv(ROOT / "artifacts/inactivity/inactivity_v2_locked_predictions.csv")
    preds["Origin"] = preds["Origin"].astype(str)
    models = preds["Model"].unique()

    # one row per (Origin, Customer) with every score side by side
    wide = preds.pivot_table(index=["Origin", "Customer ID", "y"], columns="Model",
                             values="p").reset_index()
    df = wide.merge(feats[["Origin", "Customer ID", "Inactive"] +
                          [c for c, _ in BASELINES.values()]],
                    on=["Origin", "Customer ID"], how="left", validate="one_to_one")
    assert (df["y"] == df["Inactive"]).all(), "labels do not line up with saved predictions"

    scores = {m: df[m].values for m in models}
    for name, (col, sign) in BASELINES.items():
        scores[name] = sign * df[col].values

    y = df["y"].values
    origins = df["Origin"].values
    groups = [np.where(origins == o)[0] for o in LOCKED]
    codes, uniq = pd.factorize(df["Customer ID"])
    n = len(uniq)

    def mean_auc(s, w=None):
        out = []
        for g in groups:
            if w is None:
                out.append(roc_auc_score(y[g], s[g]))
            else:
                keep = g[w[g] > 0]
                out.append(roc_auc_score(y[keep], s[keep], sample_weight=w[keep]))
        return float(np.mean(out))

    rng = np.random.default_rng(0)
    weights = [np.bincount(rng.integers(0, n, n), minlength=n)[codes] for _ in range(N_BOOT)]

    rows = []
    for name, s in scores.items():
        boots = [mean_auc(s, w) for w in weights]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"Model": name, "ROC_AUC_locked_mean": mean_auc(s), "CI_lo": lo, "CI_hi": hi})
    out = pd.DataFrame(rows).sort_values("ROC_AUC_locked_mean", ascending=False)

    # paired difference: best model vs best simple baseline (same resamples)
    best_model = "B: 12-month look-back"
    best_base = out[out["Model"].isin(BASELINES)].iloc[0]["Model"]
    d = [mean_auc(scores[best_model], w) - mean_auc(scores[best_base], w) for w in weights]
    point = mean_auc(scores[best_model]) - mean_auc(scores[best_base])
    lo, hi = np.percentile(d, [2.5, 97.5])

    print(out.round(4).to_string(index=False))
    print(f"\nPaired difference  {best_model}  minus  {best_base}: "
          f"{point:+.4f}  (95% CI {lo:+.4f} to {hi:+.4f})")
    out.to_csv(ROOT / "artifacts/inactivity/inactivity_v2_locked_baselines.csv", index=False)


if __name__ == "__main__":
    main()
