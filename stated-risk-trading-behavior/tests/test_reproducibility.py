from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived" / "final_exposure_dataset.csv.gz"
SUMMARY = ROOT / "results_final" / "analysis_summary.json"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA, low_memory=False)


def test_expected_sample_counts() -> None:
    data = load_data()
    event_ids = data["event_date"].astype(str) + "|" + data["ISIN"].astype(str)
    assert len(data) == 35_424
    assert data["customerID"].nunique() == 5_492
    assert event_ids.nunique() == 552
    assert int(data["acted"].sum()) == 2_482
    assert int(data["bought"].sum()) == 1_420
    assert int(data["sold"].sum()) == 1_062


def test_recorded_summary_matches_derived_data() -> None:
    recorded = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert int(recorded["exposures"]) == 35_424
    assert int(recorded["customers"]) == 5_492
    assert int(recorded["events"]) == 552
    assert int(recorded["actions"]) == 2_482


def test_questionnaire_precedes_event() -> None:
    data = load_data()
    questionnaire_columns = [
        name for name in data.columns if "questionnaire" in name.lower() and "date" in name.lower()
    ]
    if questionnaire_columns:
        event_date = pd.to_datetime(data["event_date"], errors="coerce")
        for name in questionnaire_columns:
            questionnaire_date = pd.to_datetime(data[name], errors="coerce")
            valid = questionnaire_date.notna()
            assert (questionnaire_date[valid] <= event_date[valid]).all()


def test_chronological_split_years_are_present() -> None:
    years = pd.to_datetime(load_data()["event_date"]).dt.year
    assert set(years.unique()).issuperset({2019, 2020, 2021, 2022})
    assert years.min() >= 2019
    assert years.max() <= 2022
