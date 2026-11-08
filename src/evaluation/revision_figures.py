"""Figures for the revised paper (vector PDF + 300-dpi PNG, readable in greyscale).
Run after revision_analyses.py:  python src/evaluation/revision_figures.py
Writes paper_figures/v2/*.pdf and *.png"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "artifacts/revision"; OUT = ROOT / "paper_figures/v2"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "pdf.fonttype": 42, "font.family": "serif"})
STY = {  # colour + marker + linestyle so that every series differs in greyscale too
    "obs": dict(color="black", marker="o", ls="-", lw=2), "gb": dict(color="#555555", marker="s", ls="--", lw=1.5),
    "hgb": dict(color="#1f4e99", marker="^", ls="-.", lw=1.5), "base": dict(color="#999999", marker="x", ls=":", lw=1.5),
    "sl": dict(color="#b35900", marker="D", ls=(0, (5, 1)), lw=1.3)}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---- Fig 1: ROC-AUC by locked origin
lp = pd.read_csv(ROOT / "artifacts/inactivity/inactivity_v2_locked_predictions.csv"); lp["Origin"] = lp["Origin"].astype(str)
names = {"GB-7 (before)": "GB-Base", "GB-17 (before, repaired notebook)": "GB-Window", "A: in-window": "LR-Base",
         "B: 12-month look-back": "HGB-Lookback"}
sty = {"GB-Base": STY["gb"], "GB-Window": STY["sl"], "LR-Base": STY["base"], "HGB-Lookback": STY["hgb"]}
fig, ax = plt.subplots(figsize=(5.2, 3.3))
for m, n in names.items():
    d = lp[lp["Model"] == m]
    v = [roc_auc_score(g["y"], g["p"]) for _, g in d.groupby("Origin")]
    ax.plot(["Jun 2011", "Jul 2011", "Aug 2011"], v, label=n, ms=6, **sty[n])
ax.axhline(0.7270, color="black", lw=1, ls=":"); ax.text(2.05, 0.7275, "Monetary only\n(3-origin mean)", fontsize=7, va="bottom", ha="right")
ax.set_ylabel("ROC-AUC"); ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right"); ax.set_xlim(-0.2, 2.2)
save(fig, "fig1_auc_by_origin")

# ---- Fig 2: drift with all 12 origins
o = pd.read_csv(R / "revision_origin_table.csv")
x = np.arange(len(o)); lab = [pd.Period(s, "M").strftime("%b\n%Y") for s in o["Origin"]]
fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.axvspan(-0.5, 2.5, color="0.93"); ax.axvspan(8.5, 11.5, color="0.85")
ax.text(1, 0.675, "exploratory", ha="center", fontsize=7); ax.text(5.5, 0.675, "development", ha="center", fontsize=7)
ax.text(10, 0.675, "locked", ha="center", fontsize=7)
ax.plot(x, o["Observed rate"], label="Observed inactivity rate", ms=5, **STY["obs"])
ax.plot(x, o["MeanPred HGB-Lookback"], label="Mean prediction, HGB-Lookback", ms=5, **STY["hgb"])
ax.plot(x, o["MeanPred GB-7"], label="Mean prediction, GB-Base", ms=5, **STY["gb"])
ax.plot(x, o["Train base rate"], label="Training base rate (constant predictor)", ms=5, **STY["base"])
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7); ax.set_ylabel("Rate / mean probability"); ax.set_ylim(0.34, 0.70)
ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
save(fig, "fig2_drift_all_origins")

# ---- Fig 3: lag function
lag = pd.read_csv(R / "revision_lag_function.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, key, ttl in ((axs[0], "G", "All 12 origins"), (axs[1], "G_FebAug", "Feb–Aug 2011 origins (7)")):
    for s, st, lb in (("GB-7", "gb", "GB-Base"), ("GB-7 sliding(3)", "sl", "GB-Base, sliding window"),
                      ("HGB-Lookback", "hgb", "HGB-Lookback"), ("Training base rate", "base", "Training base rate")):
        d = lag[lag["Series"] == s]; ax.plot(d["k"], d[key], label=lb, ms=5, **STY[st])
    ax.set_xlabel("k (months back)"); ax.set_xticks(range(5)); ax.text(0.97, 0.03, ttl, transform=ax.transAxes, fontsize=8, va="bottom", ha="right")
axs[0].set_ylabel("Mean |mean prediction − rate k months earlier|"); axs[1].legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1, 0.88))
save(fig, "fig3_lag_function")

# ---- Fig 4: reliability diagrams
rb = pd.read_csv(R / "revision_reliability_bins.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
for ax, sc in zip(axs, ["Confirmatory Jun-Jul 2011", "Previously inspected Aug 2011"]):
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls=":")
    for m, st, lb in (("Raw (uncalibrated)", "obs", "Uncalibrated"), ("Platt, last 2", "hgb", "Platt, last 2 matured origins"),
                      ("Isotonic, last 2", "gb", "Isotonic, last 2 matured origins")):
        d = rb[(rb["Scope"] == sc) & (rb["Method"] == m)]
        ax.plot(d["MeanPred"], d["Observed"], label=lb, ms=4.5, **{**STY[st], "lw": 1.3})
    ax.set_xlabel("Mean predicted probability (decile bin)"); ax.text(0.03, 0.95, sc.replace("Previously inspected", "Holdout"), transform=ax.transAxes, fontsize=8, va="top")
axs[0].set_ylabel("Observed inactivity rate"); axs[0].legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "fig4_reliability")

# ---- Fig 5: per-origin ECE for recalibration strategies
po = pd.read_csv(R / "revision_recalibration_per_origin.csv")
fig, ax = plt.subplots(figsize=(6.6, 3.2))
for m, st, lb in (("Raw (uncalibrated)", "obs", "Uncalibrated"), ("Platt, last 2", "hgb", "Platt, last 2 matured"),
                  ("Platt, all matured", "sl", "Platt, all matured"), ("Intercept shift to last matured rate", "base", "Intercept shift"),
                  ("Saerens EM prior shift", "gb", "Saerens EM")):
    d = po[po["Method"] == m]
    ax.plot([pd.Period(s, "M").strftime("%b %y") for s in d["Origin"]], d["ECE10"], label=lb, ms=4.5, **{**STY[st], "lw": 1.3})
ax.set_ylabel("ECE (10 equal-frequency bins)"); ax.legend(frameon=False, fontsize=7, ncol=2); plt.setp(ax.get_xticklabels(), fontsize=7)
save(fig, "fig5_ece_by_origin")
print("done", sorted(p.name for p in OUT.iterdir()))
