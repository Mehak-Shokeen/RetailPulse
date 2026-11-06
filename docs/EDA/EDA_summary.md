
# RetailPulse — Exploratory Data Analysis Summary

## 1. Overview

This document consolidates the exploratory data analysis performed
on the Online Retail II dataset for the RetailPulse project.

The objective of the analysis was to understand the dataset's
structure, data quality, customer purchasing behaviour,
transaction characteristics, product distributions, and
temporal patterns before preprocessing and model development.

The findings provide the foundation for the subsequent
development of customer segmentation, purchase inactivity
prediction, and product association models.

## 2. Dataset Summary

| Attribute | Description |
|---|---|
| Dataset | Online Retail II |
| Source | UCI Machine Learning Repository |
| Business domain | Online retail |
| Primary market | United Kingdom |
| Workbook sheets | 2 |
| Combined raw records | 1,067,371 |
| Number of features | 8 |
| Number of countries | 40 |
| Observation start | 1 December 2009 |
| Observation end | 9 December 2011 |
| Currency | GBP |

The combined record count represents the concatenation of
both workbook sheets before duplicate removal.

## 3. Major Data Quality Findings

### 3.1 Missing Values

The first workbook sheet contained:

- 107,927 missing Customer ID values (20.54%).
- 2,928 missing Description values (0.56%).

Missing Customer IDs are particularly relevant because
customer-level modelling requires reliable customer identifiers.

These percentages must be recalculated on the combined
dataset during preprocessing.

### 3.2 Duplicate Records

The investigation identified:

- 6,865 duplicate rows in the first sheet.
- 5,268 duplicate rows in the second sheet.
- 34,335 exact duplicate rows in the concatenated dataset.
- 22,523 matching records across the two sheets.

The results establish the need for a verified exact-record
deduplication procedure.

### 3.3 Cancellations and Returns

The first sheet contained 10,206 cancellation invoice rows.

Additionally, 2,121 negative-quantity records did not have
a cancellation invoice prefix.

Therefore, invoice prefixes alone are insufficient for
identifying all returns or transaction reversals.

### 3.4 Price and Quantity Anomalies

The first sheet contained:

- 12,326 negative-quantity records.
- 3,687 zero-price records.
- 3 negative-price records.

The three negative-price records were identified as
"Adjust bad debt" entries.

These findings indicate the presence of non-standard
financial and operational records.

## 4. Major Behavioural Findings

### 4.1 Customer Purchase Frequency

Based on the exploratory customer-level analysis:

- Identified customers: 5,942.
- Mean invoices per customer: 7.55.
- Median invoices per customer: 4.
- Maximum invoices per customer: 510.

The distribution was strongly right-skewed, indicating
substantial differences in purchasing frequency.

### 4.2 Customer Recorded Spending

Preliminary customer spending statistics:

- Mean: GBP 2,801.80.
- Median: GBP 843.97.
- Minimum: GBP -25,111.09.
- Maximum: GBP 598,215.22.

The substantial difference between mean and median indicates
a highly right-skewed spending distribution.

Negative spending values indicate that returns and adjustments
must be handled before constructing final monetary features.

### 4.3 Transaction Basket Size

The exploratory basket analysis produced:

- 53,628 unique invoices.
- Mean distinct products per invoice: 19.05.
- Median: 9.
- Maximum: 1,110.

The basket-size distribution was strongly right-skewed.

Large baskets require investigation before association mining.

### 4.4 Product Popularity

The exploratory analysis identified products with substantial
recorded unit volumes.

The highest-ranked product was
"WORLD WAR 2 GLIDERS ASSTD DESIGNS", with 110,138 recorded units.

The top products provide an initial understanding of
product popularity, although final counts will depend on
preprocessing and transaction validation.

## 5. Temporal Behaviour

Monthly transaction and revenue trends showed considerable
variation throughout the observation period.

Noticeable increases occurred towards November in both
observed years.

December 2011 contains observations only through 9 December
and must not be treated as a complete month.

Customer activity broadly followed the monthly transaction trend.

Purchasing activity also varied across weekdays and hours,
with the exploratory graphs showing:

- Highest weekday activity on Thursday.
- Highest hourly activity around noon.
- Lower activity during evening hours.
- Very limited Saturday activity.

These patterns are descriptive observations and should not
be interpreted as universal retail behaviour.

## 6. Implications for Machine Learning

### 6.1 Customer Segmentation

The observed skewness in customer frequency and monetary
values suggests investigating:

- RFM feature engineering.
- Log transformations.
- StandardScaler and RobustScaler.
- K-Means and alternative clustering algorithms.
- Cluster stability and internal evaluation metrics.

### 6.2 Purchase Inactivity Prediction

The dataset provides transaction timestamps and customer
identifiers suitable for temporal feature construction.

The development process should include:

- Historical observation windows.
- Future inactivity label construction.
- Temporal train-validation-test separation.
- Leakage prevention.
- Class imbalance analysis.
- Evaluation using ROC-AUC, PR-AUC, precision, recall,
  F1-score, and Brier score.

### 6.3 Product Association Mining

Invoice identifiers and stock codes enable transaction
basket construction.

The model development process should compare:

- Apriori.
- FP-Growth.
- Support.
- Confidence.
- Lift.
- Execution time and rule quality.

Transaction validation and basket construction are essential
before calculating association rules.

## 7. Dataset Limitations

1. Customer identifiers are missing in a substantial
   proportion of the first sheet's records.
2. Duplicate records occur within and across sheets.
3. Returns and cancellations are present.
4. Financial adjustments and zero-price records require
   separate treatment.
5. The dataset is heavily dominated by UK transactions.
6. The dataset represents one retailer rather than the
   entire UK retail industry.
7. December 2011 has incomplete temporal coverage.
8. The two-year observation period limits conclusions about
   long-term seasonal behaviour.

## 8. Outstanding Preprocessing Decisions

The following decisions must be finalized before model development:

1. Exact duplicate removal.
2. Missing customer identifier treatment.
3. Cancellation and return handling.
4. Zero-price transaction treatment.
5. Financial adjustment exclusion.
6. Outlier handling.
7. Revenue calculation.
8. Model-specific dataset construction.
9. Temporal prediction-window definition.
10. Product basket validation.

These decisions should be documented and justified separately
for each modelling task where necessary.

## 9. Research Handover Notes

The EDA findings establish the initial understanding of
the UK retail dataset.

The results support further investigation of customer
segmentation, future purchase inactivity, and product
association mining.

The findings should be used as exploratory evidence rather
than final experimental results.

All numerical statistics must be recalculated after
preprocessing before being used as final model inputs
or reported as cleaned-dataset findings.

## 10. Conclusion

The exploratory analysis demonstrates that Online Retail II
contains substantial transaction-level information and
provides a suitable starting point for developing the
RetailPulse retail intelligence framework.

However, data-quality issues, customer behaviour skewness,
transaction irregularities, and temporal limitations
require careful treatment.

The next project phase will establish a reproducible,
model-aware preprocessing pipeline, followed by feature
engineering and machine learning experimentation.


## 5. Raw-source counts versus cleaned modelling populations

The 5,942 identified-customer and 53,628 invoice counts above describe the raw workbook before cleaning. They are not the final customer-level modelling population. After exact duplicate removal, cancellation/return handling, and positive-price filtering, the canonical positive-price population contains 1,007,913 purchase lines, 40,077 invoices, and 5,878 identified customers. Customer-level model rows exclude anonymous customer IDs; product-level association mining retains eligible anonymous purchase lines. See `docs/preprocessing/preprocessing_report.md` and `artifacts/research_improvements/dataset_count_reconciliation.csv` for the reconciled stages.
