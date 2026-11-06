"""
Inference helpers for the RetailPulse customer-inactivity model.

Scoring uses exactly the same code path as training:
    prepare_transactions -> build_snapshot_features -> add_derived_features
    -> select `feature_columns` in the stored order -> pipeline.predict_proba
No transformation is re-implemented here.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.features.inactivity_features import (
    add_derived_features,
    build_snapshot_features,
    prepare_transactions,
)

REQUIRED_BUNDLE_KEYS = {"model", "feature_columns", "derived_params",
                        "data_start", "risk_band_cutoffs"}


def load_bundle(path) -> dict:
    bundle = joblib.load(path)
    missing = REQUIRED_BUNDLE_KEYS - set(bundle)
    if missing:
        raise ValueError(f"Not a valid inactivity model bundle; missing keys: {sorted(missing)}")
    return bundle


def validate_feature_frame(features: pd.DataFrame, feature_columns) -> None:
    absent = [c for c in feature_columns if c not in features.columns]
    if absent:
        raise KeyError(f"Feature columns missing from the computed frame: {absent}")


def assign_risk_band(probabilities, cutoffs) -> pd.Categorical:
    """Low < medium cutoff <= Medium < high cutoff <= High."""
    return pd.cut(np.asarray(probabilities),
                  bins=[-np.inf, cutoffs["medium"], cutoffs["high"], np.inf],
                  labels=["Low", "Medium", "High"], right=False)




def calibrate_probabilities(probabilities, calibrator_bundle: dict, method: str = "platt") -> np.ndarray:
    """Apply a separately fitted probability calibrator to model probabilities.

    The calibrator bundle must match the model that generated the probabilities.
    Calibration is deliberately opt-in: temporal drift can make a previously
    fitted mapping unreliable on a later deployment period.
    """
    if method not in {"platt", "isotonic"}:
        raise ValueError("method must be 'platt' or 'isotonic'")
    if method not in calibrator_bundle:
        raise KeyError(f"Calibrator bundle does not contain {method!r}")
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if method == "platt":
        x = np.log(p / (1 - p)).reshape(-1, 1)
        return calibrator_bundle[method].predict_proba(x)[:, 1]
    return np.clip(calibrator_bundle[method].predict(p), 0, 1)

def score_snapshot(transactions: pd.DataFrame, snapshot_end, bundle: dict,
                   already_prepared: bool = False, calibrator_bundle: dict | None = None,
                   calibration_method: str = "platt") -> pd.DataFrame:
    """Predict P(no purchase in the 3 months after `snapshot_end`) for every
    customer who purchased in the 6 calendar months ending at `snapshot_end`.

    Only transactions on or before `snapshot_end` are used.
    """
    tx = transactions if already_prepared else prepare_transactions(transactions)
    feats = build_snapshot_features(tx, snapshot_end, data_start=pd.Timestamp(bundle["data_start"]))
    feats = add_derived_features(feats, bundle["derived_params"])
    validate_feature_frame(feats, bundle["feature_columns"])
    X = feats[list(bundle["feature_columns"])]          # exact stored order
    prob = bundle["model"].predict_proba(X)[:, 1]
    calibrated = (calibrate_probabilities(prob, calibrator_bundle, calibration_method)
                  if calibrator_bundle is not None else None)
    out = pd.DataFrame({
        "Customer ID": feats["Customer ID"].values,
        "Snapshot": pd.Timestamp(feats["Snapshot"].iloc[0]).normalize(),
        "InactivityRiskScore": prob,
    })
    if calibrated is not None:
        out["InactivityRiskScoreCalibrated"] = calibrated
    out["RiskPercentile"] = out["InactivityRiskScore"].rank(pct=True, method="average")
    out["RiskBand"] = assign_risk_band(prob, bundle["risk_band_cutoffs"])
    return out.sort_values("InactivityRiskScore", ascending=False).reset_index(drop=True)
