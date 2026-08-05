# Data directory

- `derived/` contains the verified derived event and exposure datasets used by the included analysis checks.
- `external/` contains the official-source manifest and high-confidence corporate-action screen.
- `raw/` is intentionally ignored by Git and is the expected local location for an extracted FAR-Trans copy.

Expected raw layout:

```text
data/raw/FAR-Trans/
├── asset_information.csv
├── customer_information.csv
├── transactions.csv
├── close_prices.csv
└── markets.csv
```

The raw FAR-Trans data are not redistributed in this repository. Consult the original dataset source and CC BY 4.0 terms.
