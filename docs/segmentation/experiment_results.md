# RetailPulse — Customer Segmentation Experimental Results

## 1. K-Means: Number of Clusters Selection

K-Means was evaluated for K values ranging from 2 to 10.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 8880.66 | 0.4187 |
| 3 | 5716.04 | 0.4010 |
| 4 | 4473.01 | 0.3616 |
| 5 | 3627.77 | 0.3664 |
| 6 | 3072.47 | 0.3469 |
| 7 | 2726.34 | 0.3353 |
| 8 | 2464.19 | 0.3155 |
| 9 | 2282.79 | 0.3093 |
| 10 | 2133.37 | 0.2954 |

K=2 produced the highest silhouette score. K=3 was retained as an alternative because it offered a more differentiated customer segmentation.

## 2. Comparative Algorithm Evaluation

| Algorithm | Configuration | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---|---:|---:|---:|
| K-Means | K=2 | 0.4187 | 0.8904 | 5791.78 |
| K-Means | K=3 | 0.4010 | 0.8726 | 6124.90 |
| Agglomerative | K=2 | 0.3921 | 0.7742 | 4021.98 |
| Agglomerative | K=3 | 0.3501 | 0.9504 | 5015.26 |
| DBSCAN | eps=0.40 | 0.3186 | 1.0663 | 3093.68 |

DBSCAN evaluation metrics were calculated on non-noise observations only. Its noise percentage was 1.36%.

## 3. DBSCAN Parameter Investigation

| Epsilon | Clusters | Noise points | Noise % | Silhouette |
|---:|---:|---:|---:|---:|
| 0.25 | 9 | 260 | 4.42% | -0.0306 |
| 0.30 | 6 | 163 | 2.77% | 0.0678 |
| 0.35 | 4 | 111 | 1.89% | 0.1412 |
| 0.40 | 2 | 80 | 1.36% | 0.3186 |
| 0.45 | 2 | 57 | 0.97% | 0.3172 |

The selected DBSCAN candidate for comparison was eps=0.40 and min_samples=5.

The 80 noise observations had an average monetary value of approximately £51,423.23. These customers were treated as unusual behavioural observations rather than invalid records.

## 4. K-Means Stability Testing

Adjusted Rand Index was calculated by comparing clustering assignments across five random seeds against the reference clustering produced using random_state=42.

| K | Mean ARI | Standard Deviation | Minimum ARI | Maximum ARI |
|---:|---:|---:|---:|---:|
| 2 | 0.9999 | 0.0003 | 0.9993 | 1.0000 |
| 3 | 0.9945 | 0.0064 | 0.9854 | 1.0000 |

Both configurations demonstrated high assignment stability.

## 5. Selected Segmentation Model

The selected baseline model is K-Means with K=3.

The selection was based on the balance between clustering quality, stability and commercial interpretability.

K=2 achieved a higher silhouette score, while K=3 produced more differentiated customer groups and the highest Calinski–Harabasz score among the tested configurations.

## 6. Final Customer Segment Profiles

| Segment | Customers | Share | Average Recency | Average Frequency | Average Monetary |
|---|---:|---:|---:|---:|---:|
| 0 | 1,840 | 31.3% | 472.95 | 1.79 | £555.30 |
| 1 | 1,675 | 28.5% | 57.29 | 15.95 | £8,561.95 |
| 2 | 2,363 | 40.2% | 91.94 | 2.94 | £851.37 |

### Segment Interpretation

**Segment 0 — Long-inactive, low-frequency customers**

These customers have the highest average recency and relatively low purchasing frequency and monetary contribution.

**Segment 1 — High-value, frequent customers**

These customers exhibit the lowest average recency, highest purchase frequency and highest monetary contribution.

**Segment 2 — Moderately active, occasional customers**

These customers have more recent purchasing activity than Segment 0 but lower frequency and monetary contribution than Segment 1.

These names are provisional behavioural interpretations, not predefined labels used to train the model.

## 7. Saved Artifacts

The following artifacts were generated:

- `kmeans_model.pkl`
- `rfm_scaler.pkl`
- `customer_segments.csv`
- `segment_profiles.csv`

They are stored under `artifacts/segmentation/`.

A reload-and-predict test reproduced the original assignments for five sample customers.

## 8. Limitations

1. The segmentation uses only three RFM features and may not capture all dimensions of customer behaviour.
2. The dataset represents a single online retailer and is heavily concentrated in the United Kingdom.
3. Customers without identifiers could not be included in customer-level segmentation.
4. The selected three-cluster solution prioritises business interpretability over the highest silhouette score.
5. Cluster stability was tested across random initialisations on the same dataset; it does not establish stability over time or across different samples.
6. The clustering results are descriptive and do not establish causal relationships.
7. The reference date is derived from the final transaction in the dataset. It should not be interpreted as a real-time customer status date.

## 9. Next Steps

- Validate segment behaviour on independent or later-period data.
- Investigate additional behavioural features.
- Integrate customer assignments into the RetailPulse application.
- Use segmentation outputs alongside the future purchase inactivity prediction module.
- Develop segment-specific business recommendations.

## 10. Research-improvement addendum: expanded K selection and temporal stability

K-Means was recomputed for K=2 through K=10 on the saved 5,878-customer RFM population. K=2 produced silhouette 0.4386, while K=3 produced 0.3470. K=2 therefore has stronger separation under silhouette; K=3 is retained as an interpretability alternative rather than the statistically best solution.

Temporal stability was evaluated by independently fitting/scaling RFM clusters on two non-overlapping 12-month windows (Dec 2009-Nov 2010 and Dec 2010-Nov 2011), then comparing assignments for the 2,713 customers present in both windows. Adjusted Rand Index was 0.2233 for K=2 and 0.1401 for K=3. These low values indicate substantial temporal movement. The current K=3 assignments should therefore be interpreted as a descriptive snapshot, not as persistent customer identities.

The expanded metrics are in `artifacts/research_improvements/segmentation_cluster_count_comparison_full.csv` and `segmentation_temporal_stability.csv`.
