# RetailPulse — Research Improvements Implementation Guide

## Run the complete eight-improvement audit

From the repository root, place the official two-sheet UCI Online Retail II workbook somewhere accessible, then run:

```bash
python scripts/run_all_research_improvements.py --dataset "path/to/online_retail_II.xlsx"
```

To also regenerate `data/processed/cleaned_transactions.csv` for the Streamlit product-description mapping and transaction lookup, add `--export-cleaned`.

The script validates the expected sheet names (`Year 2009-2010` and `Year 2010-2011`), reconstructs the canonical cleaned transaction population, and writes research outputs under `artifacts/research_improvements/`.

It does **not** overwrite the original model bundles, original recommendation evaluation, original association rules, or the saved segmentation assignments.

## What is implemented

1. **Baselines:** recency-only inactivity ranking and a development-fitted one-feature logistic baseline; training-period top-seller recommender baseline.
2. **Calibration:** Platt and isotonic mappings fitted on April-May 2011 development out-of-fold predictions; evaluated on June-July 2011. August 2011 is explicitly marked previously inspected.
3. **Recommendation denominators:** all-basket precision, covered-basket precision, coverage, and no-recommendation counts.
4. **Association-rule screening:** code proximity plus product-description similarity; full audit and conservative experimental filtered rule export.
5. **Segmentation:** K=2 through K=10 quality metrics; temporal stability across two non-overlapping 12-month windows.
6. **Missingness:** training-table missingness rates and subgroup metrics for prior-history availability, prior-window purchase recency, and observed tenure.
7. **Test governance:** explicit development, confirmatory, and previously inspected holdout roles.
8. **Dataset reconciliation:** source rows, exact duplicate removal, cancellation/return separation, positive-price purchases, identified-customer population, and module populations.

## Important output files

| Output | Purpose |
|---|---|
| `inactivity_recency_baseline_comparison.csv` | Recency-only baseline vs saved model evaluation |
| `inactivity_calibration_before_after.csv` | Platt/isotonic before-after metrics |
| `inactivity_calibrators_development_only.joblib` | Opt-in calibration mappings with fit metadata |
| `inactivity_missingness_subgroup_metrics.csv` | Subgroup AUC, PR-AUC, Brier |
| `recommendation_baseline_and_denominator_comparison.csv` | Top sellers vs rule recommender; denominator correction |
| `association_rule_product_description_audit.csv` | Product descriptions and related-series screening |
| `association_rules_conservative_filtered.csv` | Experimental rule set with potential same-series rules excluded |
| `segmentation_cluster_count_comparison_full.csv` | K=2..10 clustering diagnostics |
| `segmentation_temporal_stability.csv` | Temporal ARI across independent windows |
| `inactivity_temporal_data_governance.csv` | Data-role audit |
| `dataset_count_reconciliation.csv` | Stage-by-stage population counts |
| `EIGHT_IMPROVEMENTS_EXECUTION_REPORT.md` | Results and limitations summary |

## Calibration use

The calibrator is intentionally **opt-in**. `src/models/inactivity_inference.py` now supports:

```python
from src.models.inactivity_inference import load_bundle, score_snapshot

bundle = load_bundle("artifacts/inactivity/inactivity_model_v2_extended_lookback.joblib")
calibrators = joblib.load(
    "artifacts/research_improvements/inactivity_calibrators_development_only.joblib"
)
scores = score_snapshot(
    transactions,
    snapshot_end,
    bundle,
    calibrator_bundle=calibrators,
    calibration_method="platt",
)
```

The calibration mapping was evaluated on June-July 2011. Do not assume it transfers to a much later period without a new temporal validation. The dashboard's existing saved production scores remain unchanged by default.

## Research interpretation

- A higher AUC than the recency baseline supports incremental ranking value, but it does not prove business uplift.
- The rule-based recommender outperforms the top-seller baseline on the saved basket-completion protocol, but this is not a live experiment.
- The same-series filter is a heuristic requiring manual product-category review before a definitive claim.
- K=2 performs better on silhouette in this dataset. K=3 remains an interpretability choice, while temporal ARI indicates that assignments are not stable over time.
- Missing values are explicit where the source history does not cover the full prior window; they are not replaced with fabricated zeros.
- August 2011 was previously inspected and must never be described as a pristine final test set.
- The raw 5,942-customer / 53,628-invoice counts are pre-cleaning counts. The canonical cleaned identified-customer population is 5,878 customers and 40,077 invoices.

