import pandas as pd

from scripts.core.diagnostic_analysis import calibration_bins
from scripts.core.hierarchical_logistic import effect_code_sectors
from scripts.core.research_pipeline import (
    candidate_60_columns,
    make_rolling_splits,
    parse_filename,
)


def test_filename_parser_extracts_target_sector_and_years():
    parsed = parse_filename("bankrupt_retail_16_year_13_14_15.csv")
    assert parsed == {
        "target": 1,
        "sector": "retail",
        "eval_year": 2016,
        "financial_years": (2013, 2014, 2015),
    }


def test_candidate_feature_set_contains_20_ratios_across_three_lags():
    columns = candidate_60_columns()
    assert len(columns) == 60
    assert columns[0] == "ratio_01_t_minus_3"
    assert columns[-1] == "ratio_20_t_minus_1"


def test_rolling_splits_never_train_on_future_years():
    frame = pd.DataFrame({"eval_year": [2013, 2014, 2015, 2016]})
    assert make_rolling_splits(frame) == [
        ((2013,), 2014),
        ((2013, 2014), 2015),
        ((2013, 2014, 2015), 2016),
    ]


def test_effect_coding_has_zero_sum_across_sector_levels():
    encoded, columns = effect_code_sectors(
        ["agriculture", "construction", "manufacture", "retail"]
    )
    assert columns == ["sector_agriculture", "sector_construction", "sector_manufacture"]
    assert encoded.shape == (4, 3)
    assert (encoded.sum(axis=0) == 0).all()
    assert (encoded[-1] == -1).all()


def test_calibration_bins_return_observed_and_predicted_risk():
    table = calibration_bins(
        y_true=[0, 0, 1, 1],
        scores=[0.05, 0.15, 0.75, 0.95],
        n_bins=2,
    )
    assert list(table.columns) == [
        "bin",
        "count",
        "mean_predicted",
        "observed_rate",
        "score_min",
        "score_max",
    ]
    assert table["count"].sum() == 4
