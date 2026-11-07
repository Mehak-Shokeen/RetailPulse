"""
Makes the three paper figures from files that already exist in the project.

Run from the project root:
    python src/evaluation/make_figures.py

Reads
    artifacts/research_improvements/inactivity_development_oof_predictions.csv
    artifacts/research_improvements/inactivity_locked_model_comparison.csv
    artifacts/inactivity/inactivity_v2_locked_predictions.csv
Writes (300 dpi PNG)
    paper_figures/fig1_auc_by_origin.png
    paper_figures/fig2_drift_observed_vs_predicted.png
    paper_figures/fig3_calibration_before_after.png

It also re-fits Platt scaling on the April-May 2011 development predictions and prints
whether the result matches the number saved by your friend's script (it should).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)

DEV = ROOT / "artifacts/research_improvements/inactivity_development_oof_predictions.csv"
CMP = ROOT / "artifacts/research_improvements/inactivity_locked_model_comparison.csv"
LOCKED = ROOT / "artifacts/inactivity/inactivity_v2_locked_predictions.csv"
MODEL_B = "B: 12-month look-back"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120})


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece_equal_frequency(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if (idx == b).any())


def ece_equal_width(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if (idx == b).any())


def load():
    dev_all = pd.read_csv(DEV)
    dev_all["Origin"] = dev_all["Origin"].astype(str)
    dev = dev_all[dev_all["PredictionRole"] == "frozen development OOF"]
    locked = pd.read_csv(LOCKED)
    locked["Origin"] = locked["Origin"].astype(str)
    locked = locked[locked["Model"] == MODEL_B]
    return dev, locked, dev_all


# ---------------------------------------------------------------- Figure 1
def fig1():
    d = pd.read_csv(CMP)
    d = d[d["Scope"].isin(["2011-06", "2011-07", "2011-08"])]
    order = ["GB-7 (before)", "GB-17 (before, repaired notebook)", "A: in-window", MODEL_B]
    labels = {"GB-7 (before)": "GB-7", "GB-17 (before, repaired notebook)": "GB-17",
              "A: in-window": "A: in-window", MODEL_B: "B: 12-month look-back"}
    marks = ["s", "^", "D", "o"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    months = ["Jun 2011", "Jul 2011", "Aug 2011"]
    for m, mk in zip(order, marks):
        s = d[d["Model"] == m].sort_values("Scope")
        ax.plot(months, s["ROC_AUC"].values, marker=mk, lw=2 if m == MODEL_B else 1.3,
                label=labels[m], zorder=3 if m == MODEL_B else 2)
    base = ROOT / "artifacts/inactivity/inactivity_v2_locked_baselines.csv"
    if base.exists():
        b = pd.read_csv(base)
        mon = b[b["Model"] == "Monetary only"]["ROC_AUC_locked_mean"]
        if len(mon):
            ax.axhline(float(mon.iloc[0]), ls="--", color="gray", lw=1)
            ax.text(2.02, float(mon.iloc[0]), "Monetary only\n(3-month mean)", fontsize=8,
                    va="center", color="gray")
    ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_auc_by_origin.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig2(dev, locked, dev_all):
    rows = []
    for o, g in dev.groupby("Origin"):
        rows.append((o, g["y"].mean(), g["p"].mean(), "development"))
    for o, g in locked.groupby("Origin"):
        rows.append((o, g["y"].mean(), g["p"].mean(), "locked"))
    t = pd.DataFrame(rows, columns=["Origin", "obs", "pred", "role"]).sort_values("Origin")
    print("\nFigure 2 data (observed vs mean predicted by origin):")
    print(t.round(3).to_string(index=False))

    # How far behind does the mean prediction lag the observed rate?
    # Uses every origin that has predictions (including the exploratory Sep-Nov 2010 ones)
    # so that observed rates are available for the look-back months.
    allrows = pd.concat([dev_all[["Origin", "y", "p"]], locked[["Origin", "y", "p"]]])
    ta = (allrows.groupby("Origin").agg(obs=("y", "mean"), pred=("p", "mean"))
          .reset_index().sort_values("Origin").reset_index(drop=True))
    obs = dict(zip(ta["Origin"], ta["obs"]))
    months = list(ta["Origin"])
    for start_origin, note in [("2011-02", "origins Feb-Aug 2011 (first two dev origins left out: very few matured training snapshots)"),
                               ("2010-09", "ALL origins Sep 2010-Aug 2011 (sensitivity check)")]:
        print(f"\nLag check: mean |mean prediction - observed rate k months earlier|, {note}")
        for lag in range(0, 6):
            gaps = []
            for _, r in ta[ta["Origin"] >= start_origin].iterrows():
                i = months.index(r["Origin"]) - lag
                if i >= 0:
                    gaps.append(abs(r["pred"] - obs[months[i]]))
            print(f"  lag {lag} months: MAE {np.mean(gaps):.3f}  (n={len(gaps)})")

    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7, 3.9))
    ax.plot(x, t["obs"], marker="o", lw=2, label="Observed inactivity rate")
    ax.plot(x, t["pred"], marker="s", lw=2, ls="--", label="Mean predicted probability (Model B)")
    first_locked = int((t["role"] == "locked").values.argmax())
    ax.axvspan(first_locked - 0.5, len(t) - 0.5, color="gray", alpha=0.12)
    ax.text(first_locked, ax.get_ylim()[1] * 0.995, "locked origins", fontsize=8, va="top")
    ax.set_xticks(x)
    ax.set_xticklabels(pd.to_datetime(t["Origin"]).dt.strftime("%b\n%Y"), fontsize=8)
    ax.set_ylabel("Rate / probability")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_drift_observed_vs_predicted.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig3(dev, locked):
    fit = dev[dev["Origin"].isin(["2011-04", "2011-05"])]
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(fit["p"].values).reshape(-1, 1),
                                                         fit["y"].values)
    test = locked[locked["Origin"].isin(["2011-06", "2011-07"])]
    y = test["y"].values
    p0 = test["p"].values
    p1 = platt.predict_proba(logit(p0).reshape(-1, 1))[:, 1]

    saved = pd.read_csv(ROOT / "artifacts/research_improvements/inactivity_calibration_before_after.csv")
    print("\nReproduction check on Jun-Jul 2011 (your numbers vs this script):")
    for name, p in [("Uncalibrated", p0), ("Platt", p1)]:
        ref = saved[(saved["Model"] == name) &
                    (saved["Scope"].str.contains("June-July"))].iloc[0]
        print(f"  {name:12s} saved ECE {ref['ECE_10_bins']:.4f} | equal-frequency {ece_equal_frequency(y, p):.4f}"
              f" | equal-width {ece_equal_width(y, p):.4f} | saved mean {ref['Mean_probability']:.4f}"
              f" vs here {p.mean():.4f}")

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":", label="Perfect calibration")
    for p, name, mk in [(p0, "Uncalibrated", "o"), (p1, "After Platt scaling", "s")]:
        edges = np.quantile(p, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
        mx = [p[idx == b].mean() for b in range(10)]
        my = [y[idx == b].mean() for b in range(10)]
        ax.plot(mx, my, marker=mk, lw=1.6, label=name)
    ax.set_xlabel("Mean predicted probability (decile bin)")
    ax.set_ylabel("Observed inactivity rate")
    ax.set_xlim(0.0, 0.95)
    ax.set_ylim(0.0, 0.95)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_calibration_before_after.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    dev_df, locked_df, dev_all_df = load()
    fig1()
    fig2(dev_df, locked_df, dev_all_df)
    fig3(dev_df, locked_df)
    print(f"\nSaved figures to {OUT}")
