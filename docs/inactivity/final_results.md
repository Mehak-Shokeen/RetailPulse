# Customer Inactivity Prediction — Final Experimental Report

## 1. Research Objective

The objective was to investigate customer inactivity prediction
and future purchasing behaviour using historical retail transactions.

Three modelling approaches were investigated:
- Discriminative binary classification.
- Probabilistic repeat-purchase modelling.
- Time-to-event survival modelling.

## 2. Gradient Boosting Classification

The classification target was defined as:
Inactive = 1 when a customer made no purchases during the
three-month prediction window.

The modelling dataset contained 48,394 customer-snapshot records
across 16 temporal snapshots.

The original feature set contained:
- Recency
- Frequency
- Monetary
- AverageOrderValue
- TotalQuantity
- UniqueProducts
- ActiveMonths

### Rolling-origin evaluation

Six historical prediction origins were evaluated from
December 2010 through May 2011.

| Metric | Mean |
|---|---:|
| ROC-AUC | 0.7540 |
| PR-AUC | 0.7582 |
| Frequency baseline ROC-AUC | 0.7303 |
| Mean ROC-AUC improvement | 0.0237 |

Gradient Boosting outperformed the frequency-only baseline
in all six evaluated periods.

These periods share customers and are not independent
experimental replications.

### Customer heterogeneity

Validation ROC-AUC varied by historical purchase frequency:

| Customer group | ROC-AUC |
|---|---:|
| One invoice | 0.58 |
| Two to three invoices | 0.61 |
| Four or more invoices | 0.76 |

This indicates substantial differences in predictability
between customer groups.

### Feature ablation

RFM achieved approximately the same ROC-AUC and PR-AUC
as the full seven-feature configuration.

The mean ROC-AUC difference (RFM minus full model)
was approximately 0.000873.

The mean PR-AUC difference (RFM minus full model)
was approximately -0.000279.

The additional features did not demonstrate a meaningful
performance improvement in these experiments.

RFM is retained as a parsimonious candidate, not as a
statistically proven superior model.

## 3. BG/NBD Probabilistic Purchase Modelling

BG/NBD was fitted using calibration-period customer
transaction histories.

The calibration population contained 4,967 customers.
The future holdout period covered 183 days.

All 4,967 eligible customers were retained in evaluation,
including customers with zero holdout purchases.

### Purchase-count forecasting

| Metric | Result |
|---|---:|
| MAE | 1.18 |
| RMSE | 2.28 |
| Pearson correlation | 0.86 |
| Aggregate prediction error | -3.84% |

Actual mean holdout purchases: 1.7687.
Predicted mean holdout purchases: 1.7008.

### Zero-purchase ranking

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8051 |
| PR-AUC | 0.7602 |
| Zero-purchase prevalence | 0.4812 |

BG/NBD ranked customers with zero holdout purchases
reasonably well.

These results concern a six-month holdout and must not
be directly compared with the three-month classification
metrics as if they represented the same prediction task.

BG/NBD purchase forecasts are not calibrated probabilities
of permanent customer churn.

## 4. Survival Analysis

A Cox Proportional Hazards model was explored using
time to first subsequent purchase.

The dataset contained 4,967 customers:
- 2,577 observed purchase events.
- 2,390 right-censored observations.
- 183-day observation horizon.

The fitted model produced a convergence warning.
Its reported concordance index was 0.372.

The model's mean predicted six-month survival probability
was 0.8841, compared with an observed no-purchase rate
of 0.4812.

The model was not accepted as a successful predictive model.
Its convergence issue and poor apparent ranking performance
require further investigation.

The reported concordance index was calculated on the
fitting population and is not an independent test estimate.

## 5. Overall Conclusions

The experiments demonstrate that customer purchase history
contains useful information about future purchasing behaviour.

Gradient Boosting provided relatively stable ranking
performance across six historical rolling-origin periods.

BG/NBD provided a distinct probabilistic approach to
forecasting future purchases and ranking zero-purchase risk.

The Cox experiment did not produce an acceptable result
and is retained as a documented negative experiment.

No claim is made that one model is universally superior,
because the models were evaluated on different prediction
targets and temporal horizons.

## 6. Limitations

- The dataset represents one online retailer.
- Customer purchasing behaviour is non-contractual.
- Customer histories are incomplete for some customers.
- Temporal periods share customers and are not independent.
- The final classification test period has limited temporal coverage.
- The BG/NBD holdout covers one six-month period.
- BG/NBD assumptions may not fully represent retail purchasing behaviour.
- The Cox model did not converge successfully.
- Predictive association does not establish causal effects.

## 7. Research Decision

The classification and BG/NBD experiments are retained
as the principal predictive approaches.

The Cox experiment is retained as a negative result.

Further work should focus on rigorous comparison,
reproducibility, documentation and integration into RetailPulse.