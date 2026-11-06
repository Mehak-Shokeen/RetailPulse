"""
Customer-inactivity feature engineering for RetailPulse.

Single source of truth for BOTH training and inference.  Every feature is
computed from transactions that fall inside the six-calendar-month observation
window ending at the snapshot (plus, for the explicit HISTORY group, earlier
transactions that are also strictly before the snapshot).  Nothing after the
snapshot is ever read: `build_snapshot_features` hard-filters its input to
`InvoiceDate <= snapshot end` before doing any calculation.

Target definition (unchanged from the existing project):
    Inactive = 1  <=>  no purchase in the 3 calendar months after the snapshot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OBSERVATION_MONTHS = 6
PREDICTION_MONTHS = 3

# Service / non-merchandise stock codes.  Used ONLY by the product-diversity
# features; the original UniqueProducts feature and the labels are untouched.
NON_PRODUCT_CODES = {"POST", "DOT", "M", "D", "S", "B", "ADJUST", "AMAZONFEE",
                     "BANK CHARGES", "CRUK", "C2"}

FEATURE_GROUPS = {
    # ---- Experiment A: strictly inside the 6-month observation window ----
    "BASE": ["Recency", "Frequency", "Monetary", "AverageOrderValue",
             "TotalQuantity", "UniqueProducts", "ActiveMonths"],
    "RECENT": ["InvoicesLast30Days", "InvoicesLast60Days", "InvoicesLast90Days",
               "SpendLast30Days", "SpendLast60Days", "SpendLast90Days"],
    "MONTHLY": ["InvoicesMonth0", "InvoicesMonth1", "InvoicesMonth2",
                "InvoicesMonth3", "InvoicesMonth4", "InvoicesMonth5"],
    "TREND": ["FrequencyTrend", "SpendTrend", "ActiveMonthsRecent",
              "AOVChange", "BasketSizeChange"],
    "TIMING": ["MeanPurchaseGapDays", "StdPurchaseGapDays",
               "LastPurchaseGapDays", "GapCV", "ActiveSpanDays"],
    "CADENCE": ["SmoothedGapDays", "RecencyOverCadence"],
    "TENURE": ["DaysSinceFirstInWindow"],
    "BASKET": ["AvgLinesPerInvoice", "AverageQuantityPerInvoice",
               "ProductDiversity", "RepeatSKUShare",
               "TopInvoiceShare", "InvoiceValueCV"],
    "COUNTRY": ["IsUK"],
    # ---- Experiment B: extends the look-back beyond the 6-month window ----
    # (still strictly before the snapshot; documented as a separate experiment)
    "PRIOR": ["PriorWindowInvoices"],
    "PRIOR_EXTRA": ["PriorWindowSpend", "PriorWindowActiveMonths",
                    "PriorWindowRecencyDays", "TwelveMonthTrend"],
    "LIFETIME": ["ObservedTenureDays"],
}
IN_WINDOW_GROUPS = ["BASE", "RECENT", "MONTHLY", "TREND", "TIMING", "CADENCE",
                    "TENURE", "BASKET", "COUNTRY"]
EXTENDED_LOOKBACK_GROUPS = ["PRIOR", "PRIOR_EXTRA", "LIFETIME"]

# Columns needed to compute derived features but not model inputs by themselves.
INGREDIENT_COLUMNS = ["Frequency", "ActiveSpanDays", "Recency"]


def snapshot_windows(snapshot_end):
    """Exact calendar-month windows for a snapshot (identical to the project's)."""
    period = pd.Timestamp(snapshot_end).to_period("M")
    return {
        "observation_start": (period - (OBSERVATION_MONTHS - 1)).start_time,
        "observation_end": period.end_time,
        "prediction_start": (period + 1).start_time,
        "prediction_end": (period + PREDICTION_MONTHS).end_time,
    }


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Identified-customer purchase lines, sorted by time, with integer IDs."""
    needed = ["Customer ID", "Invoice", "InvoiceDate", "StockCode",
              "Quantity", "TransactionValue", "Country"]
    out = df.loc[df["Customer ID"].notna(), needed].copy()
    out["Customer ID"] = out["Customer ID"].astype("int64")
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"])
    out["Invoice"] = out["Invoice"].astype(str)
    out["StockCode"] = out["StockCode"].astype(str)
    out["Country"] = out["Country"].astype(str)
    out = out.dropna(subset=["InvoiceDate", "Quantity", "TransactionValue"])
    return out.sort_values("InvoiceDate", kind="mergesort").reset_index(drop=True)


def _log_ratio(numerator, denominator):
    ratio = numerator / denominator
    return np.log(ratio.where(ratio > 0))


def build_snapshot_features(tx: pd.DataFrame, snapshot_end, data_start=None) -> pd.DataFrame:
    """One row per customer who purchased in the observation window.

    `tx` must come from `prepare_transactions`.  Rows after the snapshot's
    observation end are discarded on entry, so no future information can reach
    any feature, even if the caller passes the complete dataset.
    """
    win = snapshot_windows(snapshot_end)
    obs_start, obs_end = win["observation_start"], win["observation_end"]
    snap_year, snap_month = obs_end.year, obs_end.month

    past = tx.loc[tx["InvoiceDate"] <= obs_end]            # hard cut-off
    d = past.loc[past["InvoiceDate"] >= obs_start].copy()

    # ---- line-level (identical to the project's original seven features) ----
    d["Lag"] = (snap_year - d["InvoiceDate"].dt.year) * 12 + (snap_month - d["InvoiceDate"].dt.month)
    d["Month"] = d["InvoiceDate"].dt.year * 12 + d["InvoiceDate"].dt.month
    cust = d.groupby("Customer ID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("TransactionValue", "sum"),
        TotalQuantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        ActiveMonths=("Month", "nunique"),
        Country=("Country", "last"),
    )
    cust["Recency"] = (obs_end.normalize() - cust["LastPurchase"].dt.normalize()).dt.days
    recent_lines = d.loc[d["Lag"] <= 2]
    cust["ActiveMonthsRecent"] = (recent_lines.groupby("Customer ID")["Month"].nunique()
                                  .reindex(cust.index, fill_value=0))

    # ---- invoice-level table ----
    inv = d.groupby(["Customer ID", "Invoice"]).agg(
        InvoiceDate=("InvoiceDate", "max"),
        InvoiceValue=("TransactionValue", "sum"),
        InvoiceQty=("Quantity", "sum"),
        Lines=("StockCode", "size"),
    ).reset_index().sort_values(["Customer ID", "InvoiceDate"], kind="mergesort")
    inv["Lag"] = (snap_year - inv["InvoiceDate"].dt.year) * 12 + (snap_month - inv["InvoiceDate"].dt.month)
    g = inv.groupby("Customer ID")

    # Month-resolved activity: invoices per calendar month (0 = snapshot month).
    monthly = inv.groupby(["Customer ID", "Lag"]).size().unstack(fill_value=0)
    for lag in range(OBSERVATION_MONTHS):
        if lag in monthly.columns:
            cust[f"InvoicesMonth{lag}"] = monthly[lag].reindex(cust.index, fill_value=0)
        else:
            cust[f"InvoicesMonth{lag}"] = 0

    cust["AverageOrderValue"] = g["InvoiceValue"].mean()
    cust["AverageQuantityPerInvoice"] = g["InvoiceQty"].mean()
    cust["AvgLinesPerInvoice"] = g["Lines"].mean()
    cust["TopInvoiceShare"] = g["InvoiceValue"].max() / g["InvoiceValue"].sum()
    cust["InvoiceValueCV"] = g["InvoiceValue"].std() / g["InvoiceValue"].mean()

    # Recent windows.  Customers with no invoice in a lookback get 0 (a real zero).
    for days in (30, 60, 90):
        cutoff = obs_end - pd.Timedelta(days=days)
        rec = inv.loc[inv["InvoiceDate"] > cutoff]
        cust[f"InvoicesLast{days}Days"] = rec.groupby("Customer ID").size().reindex(cust.index, fill_value=0)
        cust[f"SpendLast{days}Days"] = (rec.groupby("Customer ID")["InvoiceValue"].sum()
                                        .reindex(cust.index, fill_value=0.0))

    # Calendar halves of the observation window: months 0-2 (recent) vs 3-5 (earlier).
    def half(mask):
        h = inv.loc[mask].groupby("Customer ID")
        return (h.size().reindex(cust.index, fill_value=0),
                h["InvoiceValue"].sum().reindex(cust.index, fill_value=0.0),
                h["InvoiceValue"].mean().reindex(cust.index),
                h["Lines"].mean().reindex(cust.index))
    n_rec, sp_rec, aov_rec, bk_rec = half(inv["Lag"] <= 2)
    n_old, sp_old, aov_old, bk_old = half(inv["Lag"] >= 3)
    cust["FrequencyTrend"] = np.log1p(n_rec) - np.log1p(n_old)
    cust["SpendTrend"] = np.log1p(sp_rec) - np.log1p(sp_old)
    cust["AOVChange"] = _log_ratio(aov_rec, aov_old)             # NaN if a half is empty
    cust["BasketSizeChange"] = _log_ratio(bk_rec, bk_old)

    # Inter-purchase timing.  Gaps exist only for customers with >= 2 invoices.
    inv["GapDays"] = g["InvoiceDate"].diff().dt.total_seconds() / 86400.0
    gg = inv.groupby("Customer ID")["GapDays"]
    cust["MeanPurchaseGapDays"] = gg.mean()
    cust["StdPurchaseGapDays"] = gg.std()                       # needs >= 3 invoices
    cust["LastPurchaseGapDays"] = gg.last()                     # last non-null gap
    cust["GapCV"] = cust["StdPurchaseGapDays"] / cust["MeanPurchaseGapDays"].where(cust["MeanPurchaseGapDays"] > 0)
    first_inv = g["InvoiceDate"].min()
    cust["ActiveSpanDays"] = (g["InvoiceDate"].max() - first_inv).dt.total_seconds() / 86400.0
    cust["DaysSinceFirstInWindow"] = (obs_end.normalize() - first_inv.dt.normalize()).dt.days

    # Product diversity, excluding service codes.
    prod = d.loc[~d["StockCode"].str.upper().isin(NON_PRODUCT_CODES), ["Customer ID", "Invoice", "StockCode"]]
    uniq = prod.groupby("Customer ID")["StockCode"].nunique()
    nlines = prod.groupby("Customer ID").size()
    cust["ProductDiversity"] = (uniq / nlines).reindex(cust.index)
    sku_inv = prod.drop_duplicates().groupby(["Customer ID", "StockCode"]).size()
    cust["RepeatSKUShare"] = (sku_inv >= 2).groupby(level="Customer ID").mean().reindex(cust.index)

    cust["IsUK"] = (cust["Country"] == "United Kingdom").astype("int8")

    # Experiment-B features: information from BEFORE the observation window
    # (never after the snapshot).  Missing = prior window not observable in data.
    first_seen = past.groupby("Customer ID")["InvoiceDate"].min()
    cust["ObservedTenureDays"] = (obs_end.normalize() - first_seen.reindex(cust.index).dt.normalize()).dt.days
    prior_start = obs_start - pd.DateOffset(months=OBSERVATION_MONTHS)
    if data_start is None:
        data_start = tx["InvoiceDate"].min().normalize().replace(day=1)
    prior_cols = ["PriorWindowInvoices", "PriorWindowSpend", "PriorWindowActiveMonths",
                  "PriorWindowRecencyDays"]
    if prior_start >= pd.Timestamp(data_start):                  # full prior window observable
        p = past.loc[(past["InvoiceDate"] >= prior_start) & (past["InvoiceDate"] < obs_start)]
        pinv = p.groupby(["Customer ID", "Invoice"]).agg(
            d=("InvoiceDate", "max"), v=("TransactionValue", "sum")).reset_index()
        pg = pinv.groupby("Customer ID")
        cust["PriorWindowInvoices"] = pg.size().reindex(cust.index, fill_value=0).astype(float)
        cust["PriorWindowSpend"] = pg["v"].sum().reindex(cust.index, fill_value=0.0)
        pm = (p["InvoiceDate"].dt.year * 12 + p["InvoiceDate"].dt.month).groupby(p["Customer ID"]).nunique()
        cust["PriorWindowActiveMonths"] = pm.reindex(cust.index, fill_value=0).astype(float)
        cust["PriorWindowRecencyDays"] = (obs_start - pg["d"].max()).dt.days.reindex(cust.index)
        cust["TwelveMonthTrend"] = np.log1p(cust["Frequency"]) - np.log1p(cust["PriorWindowInvoices"])
    else:
        for c in prior_cols + ["TwelveMonthTrend"]:
            cust[c] = np.nan                                      # explicit, not zero

    cust = cust.drop(columns=["LastPurchase", "Country"]).replace([np.inf, -np.inf], np.nan)
    cust = cust.reset_index()
    cust["Snapshot"] = obs_end
    return cust


def build_labels(tx: pd.DataFrame, snapshot_end, customers) -> pd.Series:
    """Inactive = 1 if the customer has no purchase in the 3-month prediction window."""
    win = snapshot_windows(snapshot_end)
    future = tx.loc[(tx["InvoiceDate"] >= win["prediction_start"]) &
                    (tx["InvoiceDate"] <= win["prediction_end"]), "Customer ID"].unique()
    return (~pd.Index(customers).isin(future)).astype("int8")


# ----------------------- training-fitted derived features -----------------------
def fit_derived_params(train_features: pd.DataFrame, prior_weight: float = 1.0) -> dict:
    """Population cadence prior, learned from TRAINING rows only.

    g0 = pooled mean inter-purchase gap (total span / total number of gaps)
    among training customers with at least two invoices.
    """
    multi = train_features.loc[train_features["Frequency"] >= 2]
    g0 = float(multi["ActiveSpanDays"].sum() / (multi["Frequency"] - 1).sum())
    return {"population_gap_days": g0, "prior_weight": float(prior_weight)}


def add_derived_features(features: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Shrunk cadence features.  Customers with one invoice fall back to the
    population gap (explicit, not median-imputed)."""
    out = features.copy()
    g0, k = params["population_gap_days"], params["prior_weight"]
    n_gaps = out["Frequency"] - 1
    out["SmoothedGapDays"] = (out["ActiveSpanDays"] + k * g0) / (n_gaps + k)
    out["RecencyOverCadence"] = out["Recency"] / out["SmoothedGapDays"]
    return out


def columns_for(groups) -> list:
    cols = []
    for g in groups:
        cols += [c for c in FEATURE_GROUPS[g] if c not in cols]
    return cols
