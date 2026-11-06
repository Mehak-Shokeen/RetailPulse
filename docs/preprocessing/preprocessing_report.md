# RetailPulse — Data Preprocessing Report

## 1. Dataset
- Source: UCI Online Retail II
- Original records: 1,067,371
- Original columns: 8
- Combined dataset period: December 2009 – December 2011

## 2. Duplicate Removal
- Exact duplicate copies identified and removed: 34,335
- Records after deduplication: 1,033,036
- Comparison performed using all eight original columns, excluding SourceSheet.
- First occurrence retained.
- Limitation: Identical legitimate invoice lines cannot always be distinguished from accidental duplicates.

## 3. Transaction Separation

| Category | Records |
|---|---:|
| Ordinary purchase events | 1,010,539 |
| Cancellation records | 19,104 |
| Return records | 3,393 |
| Zero-quantity records | 0 |
| Total | 1,033,036 |

## 4. Missing Values
- Missing Customer IDs in ordinary purchases: 231,044
- Missing descriptions in ordinary purchases: 1,642
- All 1,642 missing descriptions occurred in zero-price records.
- Customer identifiers were not artificially imputed.
- Anonymous transactions were retained for product-level analysis.

## 5. Price Validation

| Category | Records |
|---|---:|
| Positive-price records | 1,007,913 |
| Zero-price records | 2,621 |
| Negative-price records | 5 |

Zero-price and negative-price records were retained separately.

## 6. Transaction Value
Transaction value was calculated as:

Transaction Value = Quantity × Price

- Positive-price transactions processed: 1,007,913
- Total recorded transaction value: GBP 20,476,260.45
- Minimum transaction value: GBP 0.001

## 7. Outlier Investigation
- Quantity Q1: 1
- Quantity Q3: 12
- IQR: 11
- Upper screening threshold: 45
- Records exceeding threshold: 37,865

Extreme quantities were flagged, not deleted. Bulk purchases may be legitimate retail transactions.

## 8. Limitations
- The dataset represents one online retailer, not the entire retail market.
- Transactions are heavily concentrated in the United Kingdom.
- Missing customer identifiers restrict customer-level analysis.
- Duplicate removal may remove indistinguishable legitimate repeated lines.
- The final dataset contains historical transactions and does not establish causal relationships.

## 9. Output
Processed datasets are stored in `data/processed/`.

The cleaned positive-price transaction dataset is the primary input for subsequent feature engineering, subject to model-specific filtering and validation.