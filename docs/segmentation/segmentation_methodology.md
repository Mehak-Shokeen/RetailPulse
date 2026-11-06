# RetailPulse — Customer Segmentation Methodology

## 1. Objective

The objective of the customer segmentation module is to identify groups of customers exhibiting similar purchasing behaviours using unsupervised machine learning.

The segmentation module is designed to help retailers understand customer engagement, purchasing frequency and monetary contribution, enabling differentiated customer relationship and marketing strategies.

## 2. Dataset

The experiments use the UCI Online Retail II dataset.

- Original combined records: 1,067,371
- Records after exact duplicate removal: 1,033,036
- Ordinary purchase records: 1,010,539
- Positive-price transactions used for RFM construction: 1,007,913
- Identified customers in the final transaction dataset: 5,878
- Transaction period: December 2009 – December 2011

Customer-level modelling was restricted to transactions with available Customer IDs.

## 3. Feature Engineering

The RFM framework was used to represent customer purchasing behaviour.

### 3.1 Recency

Recency measures the number of days since the customer's most recent purchase.

Reference date:

One day after the latest transaction date in the modelling dataset.

\[
Recency = ReferenceDate - LastPurchaseDate
\]

Lower recency values indicate more recent purchasing activity.

### 3.2 Frequency

Frequency represents the number of distinct invoices associated with a customer.

\[
Frequency = Count(Distinct\ Invoices)
\]

Distinct invoices were used instead of transaction-row counts to represent purchase occasions rather than individual product lines.

### 3.3 Monetary

Monetary represents the total recorded transaction value for each customer.

\[
Monetary = \sum (Quantity \times Price)
\]

Only positive-price ordinary purchase transactions were used.

## 4. Data Transformation

Exploratory distribution analysis showed substantial right skewness in Frequency and Monetary.

The original skewness values were:

| Feature | Skewness |
|---|---:|
| Recency | 0.8872 |
| Frequency | 12.6399 |
| Monetary | 25.0702 |

A logarithmic transformation using log1p was applied to Frequency and Monetary.

\[
X' = \log(1+X)
\]

The resulting skewness values were:

| Feature | Transformed skewness |
|---|---:|
| Recency | 0.8872 |
| Frequency | 1.0015 |
| Monetary | 0.2691 |

Recency was retained without logarithmic transformation.

## 5. Feature Scaling

StandardScaler was applied to the three RFM features.

\[
Z = \frac{X-\mu}{\sigma}
\]

This standardisation ensures that the features have approximately zero mean and unit variance.

Customer ID was excluded from the feature matrix because it is an identifier rather than a behavioural variable.

## 6. Clustering Algorithms

Three unsupervised clustering approaches were investigated.

### 6.1 K-Means

K-Means partitions observations into a predefined number of clusters by minimising within-cluster squared distances.

The number of clusters was investigated for K values from 2 to 10.

### 6.2 Agglomerative Clustering

Agglomerative Clustering uses a bottom-up hierarchical approach, initially treating each observation as a separate cluster and progressively merging clusters.

Ward linkage was used to minimise increases in within-cluster variance.

### 6.3 DBSCAN

DBSCAN is a density-based clustering algorithm that identifies dense regions and marks observations outside those regions as noise.

The k-distance graph was used to guide the selection of candidate epsilon values.

The tested epsilon values were 0.25, 0.30, 0.35, 0.40 and 0.45, with min_samples fixed at 5.

## 7. Model Evaluation

The following evaluation metrics were considered:

- Silhouette Score: measures cluster cohesion and separation; higher values are preferred.
- Davies–Bouldin Index: measures similarity between clusters; lower values are preferred.
- Calinski–Harabasz Index: compares between-cluster dispersion with within-cluster dispersion; higher values are generally preferred.
- Adjusted Rand Index: measures agreement between clustering assignments obtained under different initialisations.

Cluster sizes and behavioural interpretability were also considered.

## 8. Model Selection

K-Means with three clusters was selected as the proposed segmentation baseline.

Although K-Means with two clusters obtained a higher Silhouette Score, the three-cluster solution provided more differentiated customer profiles and useful business interpretations.

The selection therefore considered both quantitative clustering performance and the practical usefulness of the resulting segments.

## 9. Reproducibility

The selected model was trained using:

- Algorithm: K-Means
- Number of clusters: 3
- Random state: 42
- Initialisations: 10
- Features: Recency, Frequency and Monetary
- Transformation: log1p on Frequency and Monetary
- Scaling: StandardScaler

The trained model and fitted scaler were saved as reusable artifacts.

## 10. Scope

The segmentation module is one component of RetailPulse. Its outputs will support subsequent customer inactivity analysis and business recommendation generation.