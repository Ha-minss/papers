# Validation Report

## Overall assessment: Share with caveats

The data pipeline, headline counts, chronological model comparison, clustered inference, bootstrap uncertainty, notebook execution, figures, technical report, and manuscript rendering were validated. The package is suitable for portfolio use and for a serious pre-submission draft. It is not yet appropriate to claim causal price shocks or immediate journal submission without the external-data checks listed below.

## Verified components

### Raw-to-derived reconstruction

A fresh run from the uploaded FAR-Trans CSV files reproduced:

- 35,424 customer-security-event exposures
- 5,492 customers
- 552 represented events
- 2,482 actions
- 1,420 buys
- 1,062 sells

Identifiers, event dates, response labels, holdings, raw and abnormal returns, risk profiles, prior transaction variables, lagged event-response variables, and alternative response-window indicators matched the canonical derived dataset.

### Predictive rerun

The deterministic rerun on the 2022 holdout produced:

| Model | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|
| Event + position | 0.245 | 0.826 | 0.0375 |
| + customer profile | 0.243 | 0.828 | 0.0375 |
| + revealed behavior | 0.253 | 0.831 | 0.0370 |
| Profile + behavior | 0.256 | 0.826 | 0.0369 |
| + prior event response | 0.286 | 0.840 | 0.0363 |

### Event-block uncertainty

- Profile bundle minus baseline PR-AUC: -0.0023; 95% interval [-0.0192, 0.0129].
- Prior event response minus profile+behavior PR-AUC: +0.0308; 95% interval [0.0139, 0.0482].

The primary-sample conclusion is that prior observed response history adds predictive value; external event screens show that the magnitude of this additional gain is sensitive to event construction. The broader and more stable conclusion is that general revealed transaction history outperforms the profile-only bundle.

### Inferential checks

Two-way-clustered action-model odds ratios relative to Conservative:

- Income: 1.09, p=0.511
- Balanced: 1.15, p=0.248
- Aggressive: 1.12, p=0.422

Revealed behavior is stronger:

- Prior transaction count, 1 SD: OR 1.47, p<0.001
- Prior extreme-event action rate, 1 SD: OR 1.35, p<0.001
- Prior sell rate in the conditional sell model, 1 SD: OR 1.24, p<0.001

## Issues and caveats

1. **High - corporate actions:** ex-dividend dates, rights issues, splits, and suspension resumptions are not exhaustively linked. Use “extreme negative-return event,” not “exogenous shock.”
2. **Medium - market benchmark:** the primary market proxy is an internal median XATH return, not the official ATHEX index.
3. **Medium - external validity:** the sample comes from one European institution and is concentrated in the Athens market.
4. **Medium - missing context:** no outside holdings, demographics, full household wealth, intraday timing, news, or questionnaire item responses are available.
5. **Medium - novelty review:** the literature review is broad but not a formal systematic search with complete Scopus/Web of Science coverage. Do not use a “first study” claim.
6. **Low - model instability:** small differences between adjacent CatBoost feature bundles are not stable enough for strong interpretation. The final paper emphasizes event-block intervals and the larger M4-M3 contrast.

## Pre-submission status

Completed analytical validation:

- The official ATHEX benchmark and Bank of Greece historical source were identified and recorded; because the binary workbook was unavailable in this isolated runtime, four internal market proxies and a three-of-four consensus screen were used instead of falsely claiming an official-index rerun.
- Four high-confidence official corporate-action overlaps and a broader conservative mechanical-event screen were analyzed.
- A documented structured literature-search log and 12-study evidence matrix were created.

Remaining publication administration:

- Optionally rerun with the official ATHEX daily workbook when it can be supplied from a networked environment.
- Adapt format, declarations, authorship, and cover letter to the selected journal.
- Obtain an independent human methodological review before submission.

## External-Source and Event-Screen Validation (2026-08-01 update)

### Overall assessment

**Share with caveats.** The primary conclusion survives conservative event screens and alternative internal market proxies: stated profiles do not reliably improve out-of-time ranking, while revealed transaction history does. The incremental value of the customer's prior extreme-event response is positive in the primary and conservative mechanical screens but smaller or uncertain under some narrower event definitions.

### Official market benchmark

- The official benchmark is the ATHEX Composite Share Price Index (GD; ISIN GRI99117A004).
- The Bank of Greece publishes an Athens Exchange-sourced historical workbook for January 2001-December 2023 and a current series from January 2022.
- The official metadata and source availability were verified, but the binary workbook could not be retrieved in the isolated runtime. The report and manuscript therefore do not claim an official-index rerun.
- Four dataset-internal alternatives were evaluated: median, equal-weight mean, 10% trimmed mean, and 2% winsorized mean. A three-of-four consensus sample retained 457 represented events and 27,075 account-event exposures.

### Corporate-action screen

Four high-confidence ATHEX-documented events overlapping the sample were removed:

1. ATTICA BANK trading resumption/adjusted trading conditions, 2021-04-29.
2. ATTICA BANK reverse split, 2021-09-30.
3. ATTICA BANK ex-rights/share-capital-increase adjustment, 2021-11-22.
4. PIRAEUS capital-restructuring sequence effective 2021-04-19, matched to the 2021-04-20 return event.

A conservative mechanical-event screen additionally excludes non-broad raw declines of at least 20%, returns near the -20%/-30% limit bands, and previous-price gaps longer than seven days. This leaves 31,701 exposures, 5,383 customers, and 517 represented events.

### Model sensitivity

LightGBM was used as an independent robustness model family. 2022 PR-AUC:

| Sample | Event/position | + profile | + revealed behavior | + prior-event response |
|---|---:|---:|---:|---:|
| Original | 0.222 | 0.224 | 0.256 | 0.274 |
| Official actions removed | 0.222 | 0.224 | 0.274 | 0.274 |
| Three-of-four market-proxy consensus | 0.210 | 0.207 | 0.263 | 0.267 |
| Conservative mechanical screen | 0.230 | 0.222 | 0.258 | 0.275 |

The robust claim is therefore:

- Profiles alone have weak and unstable incremental predictive value.
- General revealed behavior improves ranking in every sensitivity sample.
- Prior event-response history can add further value, but the magnitude is sensitive to how mechanical events and market proxies are screened.

### Novelty search

A structured public-source search and a 12-study evidence matrix were completed. No publicly indexed study was found that combines observed MiFID-style profiles, actual account-level transactions, repeated security-specific extreme negative-return responses, and chronological profile-versus-revealed-behavior prediction. Scopus, Web of Science, and ProQuest were not directly searched behind institutional login, so the manuscript avoids “first study” language.
