# RetailPulse — Research Improvement Audit (current package)

## Key results computed from the saved project artifacts

| Improvement | Current audit result |
|---|---|
| Recommendation denominators | K=5 precision = 0.152985 across all 7,253 eligible baskets; 5,711 baskets had recommendations (78.74% coverage); conditional precision on covered baskets = 0.194292. |
| Association-rule redundancy screen | 325 of 1,344 rules (24.18%) are flagged by a numeric StockCode proximity heuristic. These are candidates for manual product-description review, not confirmed trivial rules. |
| Inactivity probability gap | Saved 12-month lookback predictions have mean probability 0.5389 and observed inactivity prevalence 0.4337 on the pooled locked/confirmatory predictions. This is a calibration warning, not a reason to fit on the holdout. |
| Segmentation K comparison | Recomputed K=2..8 on saved customer RFM fields; K=2 has the highest silhouette under this explicit log1p + StandardScaler audit (0.4386 vs 0.3470 for K=3). |
| Segment profile | Saved Segment 0 has mean Recency 472.95 days, Frequency 1.79 and Monetary 555.30. |
| Data populations | Saved artifacts describe different populations: 1,007,913 association purchase lines / 40,077 baskets / 4,916 products; 5,878 segmentation customers; 7,253 recommendation evaluation baskets. |

## Important limitation

The additional uploaded workbook is `Online Retail.xlsx`, one sheet, 541,909 rows, 25,900 invoices and 4,372 customers. The existing project notebooks and artifacts were produced from a different, larger Online Retail II source. Do not mix the uploaded workbook with existing artifacts and present the combined figures as one experiment. A matching original source workbook is required to finish baseline re-training, missingness subgroup evaluation, temporal cluster stability, and exact cleaning-stage reconciliation on the same cohort.

The implementation utilities are in `scripts/run_research_improvements.py`; detailed usage and required schemas are in `docs/research_improvements/IMPLEMENTATION_NOTES.md`. Missing inputs are not silently imputed and no unsupported empirical result is fabricated.
