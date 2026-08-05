# Locked Research Design

## Research question

Do observed MiFID-style risk profiles provide incremental out-of-time predictive value beyond revealed trading behavior for account-level responses to repeated security-specific extreme negative-return events?

## Primary outcome

Any trade in the affected security during trading days t+1 through t+5 after the event.

## Secondary outcome

Sell rather than Buy, conditional on a trade.

## Primary event definition

Raw return <= -5% and market-adjusted abnormal return at or below the security's rolling prior 1st percentile, using days t-260 through t-11 and a minimum of 200 observations. Events within 20 trading days for the same security are deduplicated.

## Primary sample

- XATH stocks with at least 500 prices and at least 30% nonzero-return observations.
- Mass/Premium customers with observed Conservative, Income, Balanced, or Aggressive profiles.
- Questionnaire date on or before the event.
- Positive observed holdings before the event.
- Customer-security pairs with any reconstructed negative transaction ID excluded.

## Model sequence

- M0: event and position
- M1: M0 + customer profile
- M2: M0 + revealed behavior
- M3: M0 + profile + revealed behavior
- M4: M3 + prior event responses

## Validation

- Train: 2019-2020
- Validation and probability calibration: 2021
- Test: 2022
- Primary metric: PR-AUC
- Guardrails: ROC-AUC, Brier score, log loss, calibration, top-fraction recall/precision
- Uncertainty: event-block bootstrap
- Inference: two-way clustered standard errors by customer and date-security event

## Claims not permitted

- Causal effect of price shocks
- Causal effect of the questionnaire or profile
- Rationality or panic labels based on Buy/Sell
- Automatic client intervention recommendations
- “First study” priority claim without a complete systematic review

## Locked External Validation Addendum (2026-08-01)

1. The official benchmark source is documented but not represented as acquired unless the binary workbook is present locally.
2. Event robustness uses four cross-sectional market proxies and a three-of-four consensus sample.
3. Four high-confidence ATHEX-documented corporate-action overlaps are removed in a targeted screen.
4. A conservative screen removes the targeted events, non-broad declines of at least 20%, returns near -20%/-30% limit bands, and long previous-price gaps.
5. The final claim distinguishes two findings:
   - broad revealed transaction history is consistently more useful than stated profile labels;
   - the extra value of prior extreme-event responses is positive in several specifications but sensitive to event screening.
6. No causal language or universal first-study claim is permitted.
