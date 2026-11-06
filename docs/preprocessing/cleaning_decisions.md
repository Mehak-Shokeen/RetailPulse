
# RetailPulse — Data Preprocessing Policy

## 1. Objective

Establish a reproducible preprocessing strategy for the
Online Retail II dataset before feature engineering and
machine learning model development.

The preprocessing workflow will preserve the original data
and generate separate datasets for customer segmentation,
purchase inactivity prediction, and product association mining.

## 2. General Principles

1. Preserve the original workbook without modification.
2. Record the number of observations removed at each step.
3. Avoid irreversible transformations on raw data.
4. Validate data types and required columns.
5. Maintain reproducibility through reusable Python functions.
6. Apply model-specific filtering where necessary.
7. Recalculate data-quality statistics after cleaning.

## 3. Proposed Cleaning Decisions

### 3.1 Duplicate Records

Identify exact duplicates using the original eight transaction
columns, excluding source-sheet metadata.

Investigate duplicate occurrences within and across sheets.

Remove verified duplicate records while retaining an audit
of the number of removed observations.

Do not deduplicate using Invoice or StockCode alone.

### 3.2 Missing Customer Identifiers

Records without Customer ID cannot reliably contribute to
individual customer-level segmentation or inactivity prediction.

They may still be useful for aggregate transaction analysis
or product analysis if the remaining fields are valid.

### 3.3 Cancellations and Returns

Identify cancellation invoices using the invoice prefix.

Also identify negative-quantity transactions independently,
since not all negative quantities have cancellation prefixes.

Exclude cancellation and return records from ordinary
purchase-event construction.

Preserve them separately for audit and possible return analysis.

### 3.4 Zero and Negative Prices

Investigate zero-price records before deciding whether
they represent free products, damaged goods, or adjustments.

Exclude negative-price financial adjustments from ordinary
purchase-spending calculations.

### 3.5 Extreme Values

Investigate unusually large quantities, prices, and basket sizes.

Do not automatically delete statistical outliers.

Consider transformations and robust scaling during
model-specific feature engineering.

### 3.6 Monetary Features

Calculate purchase revenue from validated positive-quantity
and positive-price transactions.

Retain original monetary values and currency metadata.

Do not treat currency conversion as purchasing-power
normalization.

## 4. Model-Specific Data Requirements

### Customer Segmentation

Requires identifiable customers and validated purchase
transactions.

Customer features will be constructed from purchase history,
including recency, frequency, and monetary value.

### Purchase Inactivity Prediction

Requires identifiable customers, chronological purchase
history, and sufficient future observation periods.

Training labels must be constructed without future-data leakage.

### Product Association Mining

Requires valid invoice identifiers and identifiable products.

Transaction baskets will be constructed from validated
purchase records.

Customer identifiers are not strictly required for
invoice-level product association mining.

## 5. Reproducibility and Audit

Each preprocessing operation must record:

- Input row count.
- Output row count.
- Number of excluded records.
- Reason for exclusion.
- Relevant validation checks.

The original workbook must remain unchanged.

## 6. Status

This document defines the proposed preprocessing policy.

The decisions will be implemented and validated during
the preprocessing phase. They must not be interpreted
as completed cleaning operations.


