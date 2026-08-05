"""Fast integrity checks for the GitHub research package.

This script uses the included derived data and does not require the raw
FAR-Trans archive or model retraining.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "derived" / "final_exposure_dataset.csv.gz"
SUMMARY = ROOT / "results_final" / "analysis_summary.json"
PAPER = ROOT / "paper" / "Stated_or_Revealed_Risk_IEEE.pdf"

EXPECTED = {
    "exposures": 35_424,
    "customers": 5_492,
    "events": 552,
    "actions": 2_482,
    "buys": 1_420,
    "sells": 1_062,
}


def main() -> None:
    missing = [path for path in (DATASET, SUMMARY, PAPER) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(f"Required package files are missing:\n{formatted}")

    data = pd.read_csv(
        DATASET,
        usecols=["customerID", "ISIN", "event_date", "acted", "bought", "sold"],
        dtype={"ISIN": str},
    )
    event_ids = data["event_date"].astype(str) + "|" + data["ISIN"]
    observed = {
        "exposures": len(data),
        "customers": int(data["customerID"].nunique()),
        "events": int(event_ids.nunique()),
        "actions": int(data["acted"].sum()),
        "buys": int(data["bought"].sum()),
        "sells": int(data["sold"].sum()),
    }

    recorded = json.loads(SUMMARY.read_text(encoding="utf-8"))
    failures: list[str] = []
    for key, expected in EXPECTED.items():
        if observed[key] != expected:
            failures.append(f"derived {key}: expected {expected}, found {observed[key]}")
        if key in recorded and int(recorded[key]) != expected:
            failures.append(f"recorded {key}: expected {expected}, found {recorded[key]}")

    if failures:
        raise AssertionError("Package integrity check failed:\n- " + "\n- ".join(failures))

    print("Package integrity check passed.")
    for key, value in observed.items():
        print(f"{key}: {value:,}")
    print(f"paper: {PAPER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
