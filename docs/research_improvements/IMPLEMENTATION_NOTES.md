# Research improvements — implementation and reproducibility notes

## Important source-data compatibility finding

The project artifacts are from a larger Online Retail II-style source: the saved association metadata reports **1,007,913 valid purchase lines, 40,077 invoice baskets and 4,916 products**, while the saved segmentation output contains **5,878 customers**.

The separately supplied `online+retail(3).zip` contains a different workbook, **Online Retail.xlsx** (one sheet): 541,909 rows, 25,900 invoices, 4,372 identified customers, dated 2010-12-01 through 2011-12-09. It is not the two-sheet Online Retail II workbook expected by the project's original preprocessing notebook. Mixing it with the existing artifacts would invalidate comparisons and could silently create inconsistent research populations. It is therefore intentionally NOT substituted for the project's original source.

Before claiming all eight empirical tests are complete, restore the exact Online Retail II source workbook used to create the current artifacts, then run preprocessing and the new research evaluation using that same source/version.

## What is implemented now

Run:

```bash
python scripts/run_research_improvements.py
```

The script writes machine-readable audit outputs under `artifacts/research_improvements/`:

- `recommendation_denominator_comparison.csv`: all-basket and covered-basket Precision@K.
- `association_rule_diagnostics.csv` and summary: screens possible related-code families. Numeric code proximity is only a heuristic; product descriptions must be reviewed before removing rules.
- `inactivity_locked_diagnostics.csv`: descriptive AUC/Brier/probability-prevalence gap from the saved locked predictions. It deliberately does not fit calibration on locked data.
- `segmentation_cluster_count_comparison.csv`: recomputes K=2..8 on the saved RFM customer table.
- `segmentation_seed_stability.csv`: random-initialisation stability on the same data (not temporal stability).
- `segmentation_feature_profiles.csv`: profiles the saved segment assignments.
- `dataset_count_reconciliation.csv`: separates module populations instead of treating counts as interchangeable.

### Inputs required for the remaining full empirical tests

The script accepts these optional inputs:

```bash
python scripts/run_research_improvements.py \
  --transactions data/processed/cleaned_transactions.csv \
  --inactivity-feature-evaluation data/processed/inactivity_temporal_evaluation_with_features.csv \
  --development-predictions data/processed/inactivity_development_predictions.csv \\
  --calibration-test-predictions data/processed/inactivity_untouched_test_predictions.csv
```

`--transactions` must be the canonical cleaned transaction file from the same source/version as the existing artifacts, with `Invoice`, `StockCode`, `InvoiceDate`, `Quantity`, `Price`, `Customer ID`, and `TransactionValue`. It enables a chronological top-sellers recommender baseline and a temporal segmentation-stability test. The generated baseline comparison is only comparable with the existing recommender if the same transaction source, temporal split, and eligible basket construction are used.

`--inactivity-feature-evaluation` must contain temporal evaluation rows with `Origin`, `Customer ID`, `y`, and `Recency`; include `p` and the prior-history/tenure feature columns to obtain missingness-subgroup metrics. This enables the recency-only baseline and subgroup reporting.

`--development-predictions` must contain development-only `y` and `p` values. Isotonic and Platt calibrators are fitted only on this development file and saved as `inactivity_calibrators.joblib`. Optionally pass a distinct later untouched file with `--calibration-test-predictions`; the script applies the development-fitted calibrators and exports `inactivity_calibration_untouched_test.csv`. Do not pass locked/confirmatory predictions as calibration training data.

## Eight-improvement status and safeguards

1. **Baselines:** recommender popularity baseline is implemented when canonical transactions are supplied. Recency-only inactivity baseline is implemented when temporal feature/evaluation rows are supplied. The current saved locked prediction file lacks Recency, so no valid recency AUC can be inferred from it.
2. **Calibration:** Platt and isotonic fitting utilities are implemented for development predictions only. Existing locked predictions show mean scores above observed prevalence, but locked data is not used to fit a correction. Independent post-calibration evaluation remains required.
3. **Recommendation denominators:** computed from saved evaluation metrics. K=5: 15.30% all-basket precision and 19.43% among covered baskets; coverage 78.74%, with 1,542/7,253 baskets receiving no recommendations.
4. **Association rules:** a first-pass numeric-code proximity screen flags 325/1,344 rules (24.18%) as possible related-code families. This is NOT a semantic product-series determination. Verify descriptions and report results with/without confirmed related-family rules before excluding them.
5. **Segmentation:** K=2..8 quality metrics and same-data seed stability are recomputed. The saved data's K=2 silhouette is higher than K=3. Temporal stability is implemented when matching transaction data is supplied. This evidence does not justify automatically replacing K=3; business utility and temporal reproducibility should be reported separately.
6. **Missingness subgroups:** utility computes AUC and Brier by prior-history missingness and tenure when supplied temporal evaluation rows include features and predictions. Existing metadata confirms 35.76% missingness for PriorWindowInvoices and 60.63% for PriorWindowRecencyDays; subgroup performance cannot be derived from aggregate metrics alone.
7. **Holdout governance:** locked-prediction diagnostics retain the explicit warning that August 2011 was previously inspected. Development calibration and model selection must remain separate from confirmatory/locked results. Do not describe August 2011 as a pristine final test.
8. **Count reconciliation:** output separates transaction lines, invoice baskets, customer populations and evaluation cases. Exact stage-by-stage row counts require the matching original raw workbook and preprocessing outputs; module counts are not interchangeable.

## Existing artifact diagnostics (not new independent experiments)

- Saved locked predictions: the 12-month lookback model AUC is approximately 0.780 and mean predicted probability approximately 0.539, versus observed inactivity prevalence approximately 0.434.
- Saved recommendation evaluation: K=5 coverage is approximately 0.787.
- Saved segmentation: 5,878 customers, with cluster 0 mean recency approximately 473 days.
- Saved association rules: 1,344 rules.

These are descriptive audits of existing outputs, not a replacement for rerunning the complete research pipeline from the original source data.
