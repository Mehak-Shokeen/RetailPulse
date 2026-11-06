# RetailPulse — Recommendation Engine Methodology

## Purpose
Module 4 converts the association rules exported by Module 3 into a basket-based product recommendation function.

## Inputs
- `artifacts/association/association_rules.csv`
- `data/processed/cleaned_transactions.csv` (StockCode-to-Description mapping only)

## Recommendation logic
1. Accept one or more products, identified by StockCode or exact Description.
2. Resolve inputs to StockCodes.
3. Select rules for which every antecedent product is present in the input basket.
4. Collect consequent products not already in the basket.
5. Deduplicate candidates and rank by confidence, then lift, then support, all descending.
6. Return product code, product description, metrics, and the matched antecedent.

## Metrics
- **Support:** fraction of invoice baskets containing the complete itemset.
- **Confidence:** fraction of baskets containing the antecedent that also contain the consequent.
- **Lift:** confidence divided by the consequent's support. Values above 1 indicate positive association.

## Outputs
- `artifacts/recommendation/recommendation_engine_sample.csv`
- `artifacts/recommendation/recommendation_engine_metadata.json`

## Scope and limitations
This is a product-basket association recommender, not a personalised user recommender. Association does not imply causation. The sample basket is selected from an existing rule to demonstrate the function; it is not an unbiased evaluation. A formal offline evaluation on held-out invoices is required before reporting recommendation accuracy or business impact.


## Offline baseline and denominator audit (research improvement)

The temporal basket-completion evaluation now reports both denominator conventions:

- all eligible test baskets, where a basket receiving no recommendation contributes zero precision;
- covered baskets only, conditional on at least one suggestion being returned.

On the 7,253 eligible test baskets, the saved association-rule recommender covered 5,711 (78.74%), with Precision@5 = 0.1530 over all baskets and 0.1943 over covered baskets. A training-period top-selling-products baseline returned recommendations for all baskets and achieved Precision@5 = 0.0473 on the same basket-completion protocol. This supports incremental ranking value over popularity for this offline protocol, not live conversion uplift.

The exact comparison is exported to `artifacts/research_improvements/recommendation_baseline_and_denominator_comparison.csv`.

## Potentially trivial same-series rules

The 1,344 saved rules are enriched with product descriptions and screened using code proximity plus description-token evidence. 271 rules (20.16%) are flagged as potential same-series rules; the flag is a screening heuristic, not definitive product taxonomy. The unfiltered audit is preserved in `association_rule_product_description_audit.csv`; an experimental filtered set is available in `association_rules_conservative_filtered.csv`. The Streamlit interface allows the user to compare the original and filtered rule sets.
