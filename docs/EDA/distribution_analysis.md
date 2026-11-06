
# RetailPulse — Distribution and Behavioural Analysis

## 1. Objective

This document examines customer purchasing behaviour, transaction
patterns, product popularity, revenue distribution, and temporal
purchasing activity in the Online Retail II dataset.

The purpose is to identify behavioural patterns and understand
their implications for customer segmentation, purchase inactivity
prediction, and product association mining.

**Important:** The statistics in this document are exploratory
and are based on data that has not undergone final preprocessing.
They may change after duplicate removal and transaction filtering.

## 2. Monthly Transaction Trends

The monthly unique invoice analysis indicates considerable
variation in transaction activity throughout the observation period.

Key observations:

- Transaction activity fluctuates across months.
- Noticeable peaks occur towards November in both observed years.
- December 2010 shows a reduction following the November peak.
- December 2011 has incomplete coverage, ending on 9 December.

The November peaks may indicate seasonal purchasing behaviour.
However, the limited observation period does not establish a
general recurring seasonal pattern.

**Research implication:** Temporal variation should be considered
when constructing customer inactivity labels and validation windows.

## 3. Monthly Recorded Revenue

Monthly recorded revenue broadly follows the monthly transaction
pattern, with noticeable increases towards November.

However, this measure is preliminary and includes records involving
cancellations, returns, zero prices, and financial adjustments.

It should not be interpreted as finalized net sales revenue.

**Research implication:** Revenue-based features must be recalculated
after applying a documented transaction-cleaning policy.

## 4. Customer Purchase Frequency

Customer purchase frequency was calculated using the number of
unique invoice identifiers associated with each identified customer.

| Statistic | Value |
|---|---:|
| Identified customers | 5,942 |
| Mean invoices per customer | 7.55 |
| Standard deviation | 15.97 |
| Minimum | 1 |
| 25th percentile | 2 |
| Median | 4 |
| 75th percentile | 8 |
| Maximum | 510 |

### Observations

The distribution is strongly right-skewed.

Most customers have relatively few unique invoices, while a small
number of customers have substantially higher purchase frequency.

The mean is higher than the median, indicating that highly
frequent customers influence the overall average.

**Research implications:**

- Frequency features may require logarithmic transformation.
- Outliers may influence distance-based clustering algorithms.
- Customer segmentation should compare different clustering
  approaches and scaling strategies.

## 5. Customer Recorded Spending Distribution

Customer spending was calculated by summing the preliminary
transaction-line revenue for each identified customer.

| Statistic | Value (GBP) |
|---|---:|
| Mean | 2,801.80 |
| Standard deviation | 13,973.92 |
| Minimum | -25,111.09 |
| 25th percentile | 325.10 |
| Median | 843.97 |
| 75th percentile | 2,182.01 |
| Maximum | 598,215.22 |

### Observations

The spending distribution is strongly right-skewed.

The mean is substantially higher than the median, indicating
the influence of customers with very high recorded spending.

Negative customer spending values are also present because
the exploratory calculation includes returns and adjustments.

**Research implications:**

- Monetary features must be recalculated using validated
  purchase transactions.
- Logarithmic transformations and robust scaling should
  be investigated.
- Market-relative monetary features may be useful for
  cross-market comparisons.

## 6. Monthly Active Customers

Monthly active customers were calculated as the number of
distinct identified customers appearing in each month.

The resulting trend broadly follows the monthly invoice trend.

Observations include:

- Variation in the number of active customers across months.
- Increased activity towards November in both years.
- Lower activity during some early-year months.
- Incomplete December 2011 observations.

**Research implication:** Monthly customer activity can support
temporal analysis and provide context for inactivity prediction.

## 7. Transaction Basket Size

Basket size was calculated as the number of distinct stock codes
associated with each invoice.

| Statistic | Value |
|---|---:|
| Number of invoices | 53,628 |
| Mean distinct products per invoice | 19.05 |
| Standard deviation | 37.84 |
| Minimum | 1 |
| 25th percentile | 1 |
| Median | 9 |
| 75th percentile | 22 |
| Maximum | 1,110 |

### Observations

The basket-size distribution is strongly right-skewed.

Most invoices contain relatively few distinct products, while
some invoices contain substantially larger product assortments.

The maximum basket size of 1,110 is an extreme observation
that requires further investigation.

**Research implications:**

- Transaction baskets provide the foundation for association
  rule mining.
- Extremely large baskets may affect algorithm performance
  and association-rule support.
- Basket construction must be performed after transaction
  validation and deduplication.

## 8. Product Popularity

Products were ranked using total recorded units sold in an
exploratory purchase subset.

The subset excluded cancellation invoices, non-positive
quantities, and non-positive prices.

Top products included:

| Product | Units Sold |
|---|---:|
| WORLD WAR 2 GLIDERS ASSTD DESIGNS | 110,138 |
| WHITE HANGING HEART T-LIGHT HOLDER | 96,683 |
| ASSORTED COLOUR BIRD ORNAMENT | 81,809 |
| PAPER CRAFT, LITTLE BIRDIE | 80,995 |
| JUMBO BAG RED RETROSPOT | 79,279 |

These results indicate that a relatively small set of products
accounts for substantial recorded unit volume.

The analysis measures total units, not the number of distinct
invoices containing each product.

**Research implication:** Product popularity and co-purchase
relationships can support product intelligence and cross-selling
analysis.

## 9. Weekday Purchasing Activity

Transaction-line activity was examined across the seven
days of the week.

Observations:

- Thursday records the highest activity in the observed graph.
- Monday and Tuesday also show substantial activity.
- Friday and Sunday have lower activity.
- Saturday has very little recorded activity.

The low Saturday activity may reflect the retailer's operating
schedule or the characteristics of the dataset.

Further investigation is needed before interpreting this as
a general customer preference.

## 10. Hourly Purchasing Activity

Transaction activity was examined using the hour component
of the invoice timestamp.

Observations:

- Activity increases during the morning.
- Transaction volume peaks around midday.
- Activity remains substantial throughout the afternoon.
- Transaction volume declines during the evening.

The highest observed activity occurs around noon.

**Research implication:** Hourly purchasing patterns can be
used in descriptive business analytics and may help explain
differences in transaction behaviour.

## 11. Consolidated Findings and Model Implications

| Observed Pattern | Model or Product Implication |
|---|---|
| Highly skewed customer frequency | Test log transformation and robust scaling |
| Highly skewed monetary values | Recalculate monetary features and test transformations |
| Variable monthly activity | Use temporal validation |
| Large basket-size variation | Investigate basket outliers before association mining |
| Concentrated product popularity | Examine product frequency and association rules |
| Weekday and hourly variation | Include temporal business analytics |
| Seasonal-looking November peaks | Investigate seasonality cautiously |

## 12. Limitations

The findings are based on the exploratory dataset before
final preprocessing.

Duplicate records, returns, cancellations, and financial
adjustments may influence some statistics.

The dataset represents one UK-based retailer, and the observed
patterns should not be generalized to all retail businesses.

The observation period covers approximately two years,
with incomplete coverage in December 2011.

## 13. Conclusion

The exploratory analysis reveals substantial variation in
customer purchasing frequency, monetary behaviour, transaction
basket sizes, product popularity, and purchasing activity over time.

These observations establish the need for careful preprocessing,
model-specific feature engineering, and appropriate evaluation
strategies before developing the RetailPulse machine learning
components.
