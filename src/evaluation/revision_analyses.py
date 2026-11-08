"""
Revision analyses for the churn-calibration paper (responds to the pre-submission review).

Run from the project root:
    python src/evaluation/revision_analyses.py

Needs : data/processed/cleaned_transactions.csv
Writes: artifacts/revision/*.csv, artifacts/revision/revision_summary.json

What it does (review item in brackets)
  M1  rolling, maturity-respecting recalibration (Platt / isotonic, last-2 and all matured origins),
      with the old Apr-May fit relabelled "oracle"; plus label-shift baselines (M6)
  M2  training window type, snapshots per origin, training base rate per origin;
      the drift/lag analysis repeated with the 7-feature GB-7 and a sliding-window GB-7
  M4  structural-missing vs true-zero counts per snapshot; zero-fill / complete-case / indicator sensitivity
  M5  lag function G(k) over all 12 origins with block-bootstrap CIs and permutation null
  M6  Saerens EM, naive intercept shift (mean matching to last matured base rate), BBSE
  misc  5-seed stability, log-loss, intercept/slope, ECE sensitivity, contact-cost illustration
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.inactivity_features import (  # noqa: E402
    PREDICTION_MONTHS, add_derived_features, build_labels, build_snapshot_features,
    columns_for, fit_derived_params, prepare_transactions)

OUT = ROOT / "artifacts" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
ORIGINS = list(pd.period_range("2010-09", "2011-08", freq="M"))   # 12 forecast origins
SNAPS = list(pd.period_range("2010-05", "2011-08", freq="M"))     # 16 snapshots
CONFIRM = ["2011-06", "2011-07"]
HOLDOUT = ["2011-08"]
CAL_TARGETS = [o for o in ORIGINS if str(o) >= "2010-12"]        # need >=1 matured origin to calibrate
PRIOR_COLS = ["PriorWindowInvoices", "PriorWindowSpend", "PriorWindowActiveMonths",
              "PriorWindowRecencyDays", "TwelveMonthTrend"]
BASE = columns_for(["BASE"])
B13 = columns_for(["BASE", "PRIOR", "PRIOR_EXTRA", "LIFETIME"])
LOG1P = ["Frequency", "Monetary", "AverageOrderValue", "TotalQuantity", "UniqueProducts", "Recency"]
HGB_B = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=200, l2_regularization=5.0)
GB7 = dict(n_estimators=150, learning_rate=0.05, max_depth=2, min_samples_leaf=20, subsample=0.8)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ data
def build_table():
    raw = pd.read_csv(ROOT / "data/processed/cleaned_transactions.csv", parse_dates=["InvoiceDate"],
                      dtype={"Invoice": "string", "StockCode": "string"})
    tx = prepare_transactions(raw)
    data_start = tx["InvoiceDate"].min().normalize().replace(day=1)
    parts = []
    for p in SNAPS:
        f = build_snapshot_features(tx, p.end_time, data_start=data_start)
        f["Inactive"] = build_labels(tx, p.end_time, f["Customer ID"])
        parts.append(f)
    t = pd.concat(parts, ignore_index=True)
    t["SnapPeriod"] = t["Snapshot"].dt.to_period("M")
    return t


def make(kind, seed=SEED, C=0.01):
    if kind == "LR":
        pre = ColumnTransformer([("l", FunctionTransformer(np.log1p, validate=False), LOG1P)], remainder="passthrough")
        return Pipeline([("pre", pre), ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("sc", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=3000, random_state=seed))])
    if kind == "GB":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", GradientBoostingClassifier(random_state=seed, **GB7))])
    if kind == "HGB":
        return HistGradientBoostingClassifier(random_state=seed, **HGB_B)
    raise ValueError(kind)


class Folds:
    def __init__(self, table):
        self.t = table
        self.cache = {}

    def get(self, o, window=None):
        """Expanding window by default (all snapshots whose labels matured by o).
        window=w keeps only the w most recent matured snapshots (sliding)."""
        key = (str(o), window)
        if key not in self.cache:
            tr = self.t[self.t["SnapPeriod"] + PREDICTION_MONTHS <= o]
            if window:
                tr = tr[tr["SnapPeriod"] > tr["SnapPeriod"].max() - window]
            te = self.t[self.t["SnapPeriod"] == o]
            assert (tr["SnapPeriod"].max() + PREDICTION_MONTHS).end_time <= te["Snapshot"].iloc[0]
            par = fit_derived_params(tr)
            self.cache[key] = (add_derived_features(tr, par), add_derived_features(te, par))
        return self.cache[key]


def fit_predict(folds, o, kind, cols, window=None, seed=SEED, transform=None, **kw):
    tr, te = folds.get(o, window)
    if transform:
        tr, te = transform(tr), transform(te)
    cols = [c for c in cols if tr[c].notna().any()]   # a column that is missing for every training row carries no signal
    m = make(kind, seed, **kw).fit(tr[cols], tr["Inactive"])
    return pd.DataFrame({"Origin": str(o), "Customer ID": te["Customer ID"].values,
                         "y": te["Inactive"].values, "p": m.predict_proba(te[cols])[:, 1]})


# ------------------------------------------------------------------ metrics
def ece_eqfreq(y, p, bins=10):
    p = np.asarray(p, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def ece_eqwidth(y, p, bins=15):
    idx = np.clip((np.asarray(p) * bins).astype(int), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def cal_line(y, p):
    y = np.asarray(y); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return {"N": len(y), "Observed": float(y.mean()), "MeanPred": float(p.mean()),
            "ECE10": ece_eqfreq(y, p, 10), "ECE15_eqwidth": ece_eqwidth(y, p, 15),
            "Brier": float(brier_score_loss(y, p)), "LogLoss": float(log_loss(y, p)),
            "CalIntercept": float(lr.intercept_[0]), "CalSlope": float(lr.coef_[0, 0]),
            "AUC": float(roc_auc_score(y, p))}


# ------------------------------------------------------------------ recalibrators
def fit_platt(y, p):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return lambda q: lr.predict_proba(logit(q).reshape(-1, 1))[:, 1]     # p' = sigmoid(a*logit(p)+b)


def fit_iso(y, p):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    return lambda q: np.clip(iso.predict(q), 1e-4, 1 - 1e-4)


def prior_shift(p, pi_old, pi_new):
    a, b = pi_new / pi_old, (1 - pi_new) / (1 - pi_old)
    return a * p / (a * p + b * (1 - p))


def saerens(p, pi_train, iters=200):
    q = pi_train
    for _ in range(iters):
        q_new = float(np.mean(prior_shift(p, pi_train, q)))
        if abs(q_new - q) < 1e-7:
            break
        q = q_new
    return prior_shift(p, pi_train, q), q


def mean_match(p, target):
    f = lambda d: expit(logit(p) + d).mean() - target
    d = brentq(f, -8, 8)
    return expit(logit(p) + d)


def bbse(p_test, prev_y, prev_p, pi_train, thr=0.5):
    """Black-box shift estimation with hard predictions; confusion matrix from the last matured origin."""
    tpr = float((prev_p[prev_y == 1] >= thr).mean())
    fpr = float((prev_p[prev_y == 0] >= thr).mean())
    mu1 = float((p_test >= thr).mean())
    q = float(np.clip((mu1 - fpr) / max(tpr - fpr, 1e-3), 0.02, 0.98))
    return prior_shift(p_test, pi_train, q), q


def main():
    table = build_table()
    folds = Folds(table)
    rate = table.groupby("SnapPeriod")["Inactive"].mean()
    summary = {}

    # ===== 1. predictions at all 12 origins ============================
    preds = {}
    for name, (kind, cols, kw) in {"GB-7": ("GB", BASE, {}), "LR-Base": ("LR", BASE, {}),
                                   "HGB-Lookback": ("HGB", B13, {})}.items():
        preds[name] = pd.concat([fit_predict(folds, o, kind, cols, **kw) for o in ORIGINS], ignore_index=True)
        print("fitted", name, flush=True)
    preds["GB-7 sliding(3)"] = pd.concat(
        [fit_predict(folds, o, "GB", BASE, window=3) for o in ORIGINS], ignore_index=True)
    allp = pd.concat([d.assign(Model=k) for k, d in preds.items()], ignore_index=True)
    allp.to_csv(OUT / "revision_all_origin_predictions.csv", index=False)

    # ===== 2. origin table: training design, base rates, drift =========
    rows = []
    for o in ORIGINS:
        tr, te = folds.get(o)
        rows.append({"Origin": str(o), "Train snapshots": tr["SnapPeriod"].nunique(), "Train rows": len(tr),
                     "Train window": "expanding (all matured snapshots)",
                     "Train base rate": tr["Inactive"].mean(),
                     "Last matured snapshot rate (t-3)": rate[o - 3],
                     "Observed rate": te["Inactive"].mean(), "Test rows": len(te),
                     "Train rows with prior window": int(tr["PriorWindowInvoices"].notna().sum()),
                     **{f"MeanPred {m}": d.loc[d["Origin"] == str(o), "p"].mean() for m, d in preds.items()}})
    origin_tbl = pd.DataFrame(rows)
    origin_tbl.to_csv(OUT / "revision_origin_table.csv", index=False)

    # ===== 3. lag function G(k) =========================================
    pmean = lambda name, o: preds[name].loc[preds[name]["Origin"] == str(o), "p"].mean()

    def G(series, k):                       # series: dict origin-> predicted mean
        return np.array([abs(series[o] - rate[o - k]) for o in ORIGINS])

    series = {"GB-7": {o: pmean("GB-7", o) for o in ORIGINS},
              "GB-7 sliding(3)": {o: pmean("GB-7 sliding(3)", o) for o in ORIGINS},
              "HGB-Lookback": {o: pmean("HGB-Lookback", o) for o in ORIGINS},
              "LR-Base": {o: pmean("LR-Base", o) for o in ORIGINS},
              "Training base rate": {o: folds.get(o)[0]["Inactive"].mean() for o in ORIGINS}}
    rng = np.random.default_rng(0)
    B, L = 4000, 3
    lag_rows = []
    for name, s in series.items():
        for k in range(0, 5):
            g = G(s, k)
            boots = []
            for _ in range(B):                       # circular moving-block bootstrap, block length 3
                starts = rng.integers(0, len(g), int(np.ceil(len(g) / L)))
                idx = np.concatenate([(st + np.arange(L)) % len(g) for st in starts])[:len(g)]
                boots.append(g[idx].mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            lag_rows.append({"Series": name, "k": k, "G": g.mean(), "CI_lo": lo, "CI_hi": hi,
                             "G_FebAug": g[[i for i, o in enumerate(ORIGINS) if str(o) >= "2011-02"]].mean()})
    lag = pd.DataFrame(lag_rows)
    # permutation null: shuffle the 16 snapshot rates, ask how often G(3) is as small by chance
    rv = rate.values.copy()
    obs3 = {n: G(s, 3).mean() for n, s in series.items()}
    idx0 = {p: i for i, p in enumerate(SNAPS)}
    null = {n: [] for n in series}
    for _ in range(5000):
        perm = rng.permutation(rv)
        for n, s in series.items():
            null[n].append(np.mean([abs(s[o] - perm[idx0[o - 3]]) for o in ORIGINS]))
    perm_p = {n: float(np.mean(np.array(v) <= obs3[n])) for n, v in null.items()}
    # model vs trivial constant predictor at k=3: difference in G with block-bootstrap CI
    diffs = {}
    for n in ["GB-7", "GB-7 sliding(3)", "HGB-Lookback"]:
        d = G(series[n], 3) - G(series["Training base rate"], 3)
        bs = []
        for _ in range(B):
            starts = rng.integers(0, len(d), int(np.ceil(len(d) / L)))
            idx = np.concatenate([(st + np.arange(L)) % len(d) for st in starts])[:len(d)]
            bs.append(d[idx].mean())
        diffs[n] = {"G_model_minus_G_trainbase_k3": float(d.mean()), "CI_lo": float(np.percentile(bs, 2.5)),
                    "CI_hi": float(np.percentile(bs, 97.5))}
    lag.to_csv(OUT / "revision_lag_function.csv", index=False)
    summary["lag_permutation_p_G3"] = perm_p
    summary["lag_model_vs_trainbase"] = diffs
    # correlation between model mean prediction and training base rate / matured rate
    summary["corr_pbar_trainbase_GB7"] = float(np.corrcoef([series["GB-7"][o] for o in ORIGINS],
                                                          [series["Training base rate"][o] for o in ORIGINS])[0, 1])

    # ===== 4. missing vs zero ============================================
    miss = table.groupby("SnapPeriod").agg(Rows=("Inactive", "size"),
                                           PriorMissing=("PriorWindowInvoices", lambda s: int(s.isna().sum())))
    miss["Share structurally missing"] = miss["PriorMissing"] / miss["Rows"]
    zero_true = table[table["PriorWindowInvoices"].notna()].groupby("SnapPeriod").apply(
        lambda d: int((d["PriorWindowInvoices"] == 0).sum()))
    miss["Observable, true zero prior invoices"] = zero_true.reindex(miss.index).fillna(0).astype(int)
    miss.reset_index().to_csv(OUT / "revision_prior_window_missingness_by_snapshot.csv", index=False)

    def zero_fill(df):
        df = df.copy()
        m = df["PriorWindowInvoices"].isna()
        for c in ["PriorWindowInvoices", "PriorWindowSpend", "PriorWindowActiveMonths"]:
            df.loc[m, c] = 0.0
        df.loc[m, "TwelveMonthTrend"] = np.log1p(df.loc[m, "Frequency"])
        return df

    def with_indicator(df):
        df = df.copy()
        df["PriorObservable"] = df["PriorWindowInvoices"].notna().astype(float)
        return df

    def complete_case_note():
        return None

    sens = []
    var = {"HGB-Lookback (NaN, as in paper)": (lambda: preds["HGB-Lookback"]),
           "zero-fill structural missing": (lambda: pd.concat(
               [fit_predict(folds, o, "HGB", B13, transform=zero_fill) for o in ORIGINS], ignore_index=True)),
           "NaN + observability indicator": (lambda: pd.concat(
               [fit_predict(folds, o, "HGB", B13 + ["PriorObservable"], transform=with_indicator) for o in ORIGINS],
               ignore_index=True))}
    cc_preds = []
    for o in ORIGINS:
        tr, te = folds.get(o)
        trc = tr[tr["PriorWindowInvoices"].notna()]
        if len(trc) < 1000 or trc["Inactive"].nunique() < 2:
            continue
        m = make("HGB").fit(trc[B13], trc["Inactive"])
        cc_preds.append(pd.DataFrame({"Origin": str(o), "Customer ID": te["Customer ID"].values,
                                      "y": te["Inactive"].values, "p": m.predict_proba(te[B13])[:, 1]}))
    var["complete-case training (needs observable prior window)"] = lambda: pd.concat(cc_preds, ignore_index=True)
    for name, fn in var.items():
        d = fn()
        for scope, os_ in {"Dev Feb-May 2011": ["2011-02", "2011-03", "2011-04", "2011-05"],
                           "Locked Jun-Aug 2011": CONFIRM + HOLDOUT}.items():
            aucs = [roc_auc_score(g["y"], g["p"]) for o, g in d.groupby("Origin") if o in os_]
            if aucs:
                sens.append({"Variant": name, "Scope": scope, "MeanAUC": float(np.mean(aucs)), "Origins": len(aucs)})
    pd.DataFrame(sens).to_csv(OUT / "revision_missingness_sensitivity.csv", index=False)

    # ===== 5. recalibration + label-shift baselines (M1, M6) =============
    hb = preds["HGB-Lookback"]
    by_o = {o: hb[hb["Origin"] == str(o)] for o in ORIGINS}

    def matured(t, last=None):
        os_ = [o for o in ORIGINS if o + PREDICTION_MONTHS <= t]
        return os_[-last:] if last else os_

    methods_out = []          # per target origin: dict method -> calibrated p
    cal_preds = {}
    for t in CAL_TARGETS:
        te = by_o[t]; p = te["p"].values
        pi_tr = folds.get(t)[0]["Inactive"].mean()
        r3 = rate[t - 3]
        out = {"Raw (uncalibrated)": p}
        for tag, last in (("last 2", 2), ("all matured", None)):
            ms = matured(t, last)
            fit = pd.concat([by_o[o] for o in ms])
            out[f"Platt, {tag}"] = fit_platt(fit["y"].values, fit["p"].values)(p)
            out[f"Isotonic, {tag}"] = fit_iso(fit["y"].values, fit["p"].values)(p)
        out["Intercept shift to last matured rate"] = mean_match(p, r3)
        out["Saerens EM prior shift"], _ = saerens(p, pi_tr)
        prev = by_o[t - 3]
        out["BBSE prior shift"], _ = bbse(p, prev["y"].values, prev["p"].values, pi_tr)
        out["Constant: training base rate"] = np.full_like(p, pi_tr)
        out["Constant: last matured rate"] = np.full_like(p, r3)
        if str(t) in CONFIRM + HOLDOUT:       # the old Apr-May fit: oracle for Jun/Jul, legitimate only for Aug
            fit = pd.concat([by_o[pd.Period("2011-04", "M")], by_o[pd.Period("2011-05", "M")]])
            tag = "Platt, Apr-May (oracle)" if str(t) in CONFIRM else "Platt, Apr-May"
            out[tag] = fit_platt(fit["y"].values, fit["p"].values)(p)
            tag = "Isotonic, Apr-May (oracle)" if str(t) in CONFIRM else "Isotonic, Apr-May"
            out[tag] = fit_iso(fit["y"].values, fit["p"].values)(p)
        cal_preds[t] = (te, out)

    def pooled(targets, method):
        ys, ps, cs = [], [], []
        for t in targets:
            te, out = cal_preds[pd.Period(t, "M")]
            if method not in out:
                return None
            ys.append(te["y"].values); ps.append(out[method]); cs.append(te["Customer ID"].values)
        return np.concatenate(ys), np.concatenate(ps), np.concatenate(cs)

    def boot_ci(y, p, c, n=300):
        codes, uniq = pd.factorize(c)
        rg = np.random.default_rng(1)
        eb, bb = [], []
        for _ in range(n):
            w = np.bincount(rg.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]
            rep = np.repeat(np.arange(len(y)), w)
            eb.append(ece_eqfreq(y[rep], p[rep])); bb.append(brier_score_loss(y[rep], p[rep]))
        return np.percentile(eb, [2.5, 97.5]), np.percentile(bb, [2.5, 97.5])

    cal_rows = []
    scopes = {"Confirmatory Jun-Jul 2011": CONFIRM, "Previously inspected Aug 2011": HOLDOUT}
    all_methods = list(cal_preds[pd.Period("2011-06", "M")][1].keys())
    for scope, tg in scopes.items():
        for m in dict.fromkeys(all_methods + list(cal_preds[pd.Period("2011-08", "M")][1].keys())):
            r = pooled(tg, m)
            if r is None:
                continue
            y, p, c = r
            if m.startswith("Constant"):
                p = np.clip(p, 1e-4, 1 - 1e-4)
                row = {"Scope": scope, "Method": m, "N": len(y), "Observed": float(y.mean()), "MeanPred": float(p.mean()),
                       "ECE10": abs(float(y.mean()) - float(p.mean())), "Brier": float(brier_score_loss(y, p)),
                       "LogLoss": float(log_loss(y, p))}
            else:
                row = {"Scope": scope, "Method": m, **cal_line(y, p)}
                (elo, ehi), (blo, bhi) = boot_ci(y, p, c)
                row.update({"ECE10_lo": elo, "ECE10_hi": ehi, "Brier_lo": blo, "Brier_hi": bhi})
            cal_rows.append(row)
    cal_tbl = pd.DataFrame(cal_rows)
    cal_tbl.to_csv(OUT / "revision_recalibration_locked.csv", index=False)

    per_o = []
    for t, (te, out) in cal_preds.items():
        for m, p in out.items():
            if m.startswith("Constant"):
                continue
            per_o.append({"Origin": str(t), "Method": m, "ECE10": ece_eqfreq(te["y"].values, p),
                          "Brier": brier_score_loss(te["y"].values, p), "MeanPred": float(np.mean(p)),
                          "Observed": float(te["y"].mean())})
    pd.DataFrame(per_o).to_csv(OUT / "revision_recalibration_per_origin.csv", index=False)

    # reliability-diagram data (Jun-Jul pooled and Aug) for raw / platt last2 / isotonic last2 / oracle
    rel = []
    for scope, tg in scopes.items():
        for m in ["Raw (uncalibrated)", "Platt, last 2", "Isotonic, last 2", "Platt, Apr-May (oracle)", "Platt, Apr-May"]:
            r = pooled(tg, m)
            if r is None:
                continue
            y, p, _ = r
            edges = np.quantile(p, np.linspace(0, 1, 11))
            idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
            for b in range(10):
                if (idx == b).any():
                    rel.append({"Scope": scope, "Method": m, "Bin": b, "MeanPred": p[idx == b].mean(),
                                "Observed": y[idx == b].mean(), "N": int((idx == b).sum())})
    pd.DataFrame(rel).to_csv(OUT / "revision_reliability_bins.csv", index=False)

    # ===== 6. headline discrimination + baselines on locked origins ======
    base_rows = []
    for m, d in preds.items():
        for scope, os_ in {"Confirmatory Jun-Jul": CONFIRM, "Aug": HOLDOUT, "All three": CONFIRM + HOLDOUT}.items():
            a = [roc_auc_score(g["y"], g["p"]) for o, g in d.groupby("Origin") if o in os_]
            base_rows.append({"Model": m, "Scope": scope, "MeanAUC": float(np.mean(a))})
    pd.DataFrame(base_rows).to_csv(OUT / "revision_locked_auc_all_models.csv", index=False)

    # ===== 7. seed stability =============================================
    seed_rows = []
    for kind, cols, name in (("GB", BASE, "GB-7"), ("HGB", B13, "HGB-Lookback"), ("LR", BASE, "LR-Base")):
        for s in range(5):
            aucs = []
            for o in [pd.Period(x, "M") for x in CONFIRM + HOLDOUT]:
                d = fit_predict(folds, o, kind, cols, seed=s)
                aucs.append(roc_auc_score(d["y"], d["p"]))
            seed_rows.append({"Model": name, "Seed": s, "MeanAUC_locked": float(np.mean(aucs))})
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(OUT / "revision_seed_stability.csv", index=False)

    # ===== 8. cost illustration (contacts per 1,000 at threshold 0.5) =====
    cost = []
    for t in [pd.Period("2010-12", "M"), pd.Period("2011-07", "M")]:
        te, out = cal_preds[t]
        for m in ["Raw (uncalibrated)", "Platt, last 2", "Intercept shift to last matured rate"]:
            p, y = out[m], te["y"].values
            f = p >= 0.5
            cost.append({"Origin": str(t), "Method": m, "Flagged per 1000": 1000 * f.mean(),
                         "Truly inactive per 1000": 1000 * y.mean(),
                         "Precision among flagged": float(y[f].mean()) if f.any() else np.nan,
                         "Recall": float(f[y == 1].mean())})
    pd.DataFrame(cost).to_csv(OUT / "revision_cost_illustration.csv", index=False)

    # lag function on the seven origins (Feb-Aug 2011) where complete-case training exists
    cc_all = pd.concat(cc_preds, ignore_index=True)
    sub = [o for o in ORIGINS if str(o) in set(cc_all["Origin"])]
    sub_series = dict(series)
    sub_series["HGB-Lookback complete-case"] = {o: cc_all.loc[cc_all["Origin"] == str(o), "p"].mean() for o in sub}
    rows = []
    for n in ["GB-7", "GB-7 sliding(3)", "HGB-Lookback", "HGB-Lookback complete-case", "Training base rate"]:
        for k in range(0, 5):
            rows.append({"Series": n, "k": k, "Origins": len(sub),
                         "G": float(np.mean([abs(sub_series[n][o] - rate[o - k]) for o in sub]))})
    pd.DataFrame(rows).to_csv(OUT / "revision_lag_function_FebAug.csv", index=False)
    print("\nLAG on Feb-Aug origins\n", pd.DataFrame(rows).pivot(index="Series", columns="k", values="G").round(4))

    summary["rates"] = {str(k): float(v) for k, v in rate.items()}
    json.dump(summary, open(OUT / "revision_summary.json", "w"), indent=2, default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\nORIGIN TABLE\n", origin_tbl.round(3).to_string(index=False))
    print("\nLAG\n", lag.round(4).to_string(index=False))
    print("\nPERM p(G3)", perm_p, "\nMODEL vs TRAIN-BASE", diffs)
    print("\nMISSING\n", miss.round(3).to_string())
    print("\nSENS\n", pd.DataFrame(sens).round(4).to_string(index=False))
    print("\nCAL\n", cal_tbl.round(4).to_string(index=False))
    print("\nAUC all models\n", pd.DataFrame(base_rows).round(4).to_string(index=False))
    print("\nSEEDS\n", seeds.groupby("Model")["MeanAUC_locked"].agg(["mean", "std", "min", "max"]).round(4))
    print("\nCOST\n", pd.DataFrame(cost).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
