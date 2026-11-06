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
