# Data

This repository does not commit raw, derived, or generated analysis data.

Expected local raw-data layout:

```text
data/raw/FAR-Trans/
  asset_information.csv
  customer_information.csv
  transactions.csv
  close_prices.csv
  markets.csv
```

You can also set `FAR_TRANS_DATA_DIR` or pass `--data-dir` to the numbered scripts.

Generated files such as derived exposure datasets, model metrics, predictions, figures, and validation tables should be written locally and kept out of Git.
