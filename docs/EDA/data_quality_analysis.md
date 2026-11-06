
# RetailPulse — Data Quality Analysis

## 1. Objective

This analysis examines the quality and reliability of the Online Retail II
dataset before preprocessing and model development.

The investigation focuses on missing values, duplicate records,
cancellations, returns, unusual prices, quantities, and temporal coverage.

All findings in this document are based on the exploratory analysis
performed on the original workbook.

## 2. Missing Value Analysis

The first workbook sheet contains 525,461 records.

| Feature | Missing Values | Missing Percentage |
|---|---:|---:|
| Customer ID | 107,927 | 20.54% |
| Description | 2,928 | 0.56% |
| StockCode | 0 | 0% |
| Invoice | 0 | 0% |
| Quantity | 0 | 0% |
| InvoiceDate | 0 | 0% |
| Price | 0 | 0% |
| Country | 0 | 0% |

### Observations

Customer ID has the highest proportion of missing values.

This is particularly important because customer-level modelling
requires reliable customer identifiers.

Records without customer identifiers cannot directly contribute
to individual customer segmentation or inactivity prediction.

Description has a relatively small proportion of missing values.
StockCode remains available, which may still support product-level
analysis in some cases.

**Note:** These missing-value statistics refer to the first workbook
sheet only. Combined-dataset missingness will be recalculated during
preprocessing.

## 3. Duplicate Record Analysis

The workbook contains duplicate records within both sheets and
matching records across the two sheets.

| Investigation | Duplicate Records |
|---|---:|
| First sheet | 6,865 |
| Second sheet | 5,268 |
| Combined dataset | 34,335 |
| Exact cross-sheet matching records | 22,523 |

### Observations

The two sheets overlap temporally between 1 December 2010 and
9 December 2010.

The cross-sheet comparison identified 22,523 matching records.

The combined dataset contains 34,335 exact duplicate rows.

Duplicate records may inflate transaction frequency, customer
spending, product popularity, and association-rule support.

A verified exact-record deduplication procedure will therefore
be considered during preprocessing.

Deduplication will be performed using the original transaction
columns rather than invoice number alone, preserving distinct
product lines within an invoice.

## 4. Cancellation and Negative Quantity Analysis

Cancellation invoices are identified by invoice numbers beginning
with the character 'C'.

The investigation produced the following cross-tabulation:

| Invoice Category | Positive Quantity | Negative Quantity |
|---|---:|---:|
| Normal invoice | 513,134 | 2,121 |
| Cancellation invoice | 1 | 10,205 |
| Total | 513,135 | 12,326 |

### Observations

A total of 10,206 records have cancellation invoice identifiers.

Among these, 10,205 have negative quantities.

An additional 2,121 records contain negative quantities despite
not having cancellation invoice identifiers.

This demonstrates that negative quantities cannot be identified
exclusively through invoice prefixes.

Negative quantities may represent returns, reversals, or other
transaction corrections.

These records require separate treatment before customer spending
and product purchasing features are constructed.

## 5. Price and Quantity Anomalies

The first sheet contains:

| Condition | Number of Records |
|---|---:|
| Negative quantities | 12,326 |
| Zero quantities | 0 |
| Negative prices | 3 |
| Zero prices | 3,687 |

### 5.1 Negative Prices

All three negative-price records have:

- StockCode: B
- Description: Adjust bad debt
- Quantity: 1

These records represent financial adjustments rather than
ordinary product purchases.

### 5.2 Zero Prices

There are 3,687 zero-price records.

Common descriptions include:

- ?
- damages
- damaged
- missing
- OVL DOORSTOP

These records require investigation because they may represent
free items, damaged goods, operational adjustments, or other
non-standard transactions.

### 5.3 Extreme Values

The observed quantity range is:

- Minimum: -9,600
- Maximum: 19,152

The observed price range is:

- Minimum: -53,594.36 GBP
- Maximum: 25,111.09 GBP

These extreme observations may influence feature scaling,
customer spending calculations, and clustering performance.

## 6. Temporal Coverage

| Workbook Sheet | Start Date | End Date |
|---|---|---|
| Year 2009-2010 | 1 Dec 2009 | 9 Dec 2010 |
| Year 2010-2011 | 1 Dec 2010 | 9 Dec 2011 |

The workbook contains an overlapping observation period
between 1 December 2010 and 9 December 2010.

The final observation month, December 2011, is incomplete.

This limitation must be considered when constructing temporal
prediction windows and comparing monthly activity.

## 7. Initial Data Quality Implications

The findings have implications for all three RetailPulse models.

### Customer Segmentation

Missing customer identifiers, duplicate records, returns,
and extreme spending values may distort RFM features.

### Purchase Inactivity Prediction

Incorrect transaction histories and incomplete observation
windows may lead to misleading inactivity labels.

### Product Association Mining

Cancellations, returns, zero-price adjustments, and duplicate
product records may distort transaction baskets and
association-rule metrics.

## 8. Proposed Preprocessing Considerations

The following actions are proposed for the subsequent
preprocessing phase:

1. Investigate and remove verified exact duplicate records.
2. Exclude unidentified customers from customer-level modelling.
3. Separate cancellations and negative-quantity records
   from ordinary purchases.
4. Investigate zero-price records before deciding their treatment.
5. Handle financial adjustments separately.
6. Examine extreme values before applying transformations.
7. Preserve the original workbook for reproducibility.
8. Recalculate data-quality statistics on the combined dataset.

These are proposed decisions, not completed preprocessing operations.

## 9. Conclusion

The dataset provides substantial transaction-level information
for retail intelligence research.

However, missing customer identifiers, duplicate records,
cancellations, returns, financial adjustments, and extreme values
must be addressed before reliable customer-level and
transaction-level modelling can be performed.

A model-specific preprocessing strategy will be established
in the next project phase.
