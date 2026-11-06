
# RetailPulse — Dataset Overview

## 1. Dataset Information

**Dataset Name:** Online Retail II

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://doi.org/10.24432/C5CG6D

**Domain:** Retail / E-commerce

**Primary Market:** United Kingdom

**Data Format:** Excel workbook

**License:** CC BY 4.0

## 2. Dataset Description

Online Retail II contains transaction-level records from a UK-based
online retail business. The dataset records product purchases,
transaction quantities, invoice dates, unit prices, customer
identifiers, and customer countries.

The workbook contains two sheets covering consecutive observation
periods.

## 3. Dataset Structure

| Sheet | Number of Records | Observation Period |
|---|---:|---|
| Year 2009-2010 | 525,461 | 01 Dec 2009 – 09 Dec 2010 |
| Year 2010-2011 | 541,910 | 01 Dec 2010 – 09 Dec 2011 |
| Combined | 1,067,371 | 01 Dec 2009 – 09 Dec 2011 |

The combined record count represents the raw concatenation of both
sheets, before duplicate removal.

## 4. Feature Description

| Feature | Description | Data Type |
|---|---|---|
| Invoice | Transaction identifier | Object |
| StockCode | Product identifier | Object |
| Description | Product description | Object |
| Quantity | Number of units purchased or reversed | Integer |
| InvoiceDate | Transaction date and time | Datetime |
| Price | Unit price in GBP | Float |
| Customer ID | Customer identifier | Float |
| Country | Customer country | String |

## 5. Market Coverage

The workbook contains records from 40 countries.

The first sheet is heavily dominated by the United Kingdom,
with 485,852 of its 525,461 records originating from UK customers.

Therefore, the dataset should primarily be treated as a UK retail
dataset rather than a balanced international retail dataset.

International country-level analysis will require separate
consideration of sample sizes and market characteristics.

## 6. Relevance to RetailPulse

The dataset is suitable for investigating three retail intelligence
tasks:

### 6.1 Customer Segmentation
Customer IDs, invoice identifiers, transaction dates, quantities,
and prices can be used to derive customer-level behavioural features.

### 6.2 Purchase Inactivity Prediction
The transaction dates and customer identifiers enable the
construction of historical observation windows and future
purchase-inactivity labels.

### 6.3 Product Association Mining
Invoice identifiers and product codes enable the construction
of transaction-level product baskets and association rules.

## 7. Initial Dataset Limitations

- Customer identifiers are missing in a substantial number of records.
- Cancellation and return transactions are present.
- Negative and zero prices occur.
- Exact duplicate records exist within and across workbook sheets.
- The final month, December 2011, contains observations only
  through 9 December.
- The dataset represents one retailer and cannot be considered
  representative of the entire UK retail industry.

## 8. Data Handling Status

The original workbook is retained in the raw data directory.

Both sheets have been loaded and concatenated for exploratory
analysis. No final preprocessing policy has been applied yet.

The combined raw dataset contains 1,067,371 records.

Duplicate removal, missing-value treatment, return handling,
and feature engineering will be performed in subsequent phases.

