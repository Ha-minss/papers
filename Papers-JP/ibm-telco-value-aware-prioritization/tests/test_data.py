import pandas as pd
import pytest

from telco_churn.data import MODEL_FEATURES, prepare_telco_frame, validate_telco_frame


def sample_frame() -> pd.DataFrame:
    row = {
        "CustomerID": "A-001",
        "Tenure Months": 0,
        "Monthly Charges": 50.0,
        "Total Charges": None,
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Churn Label": "No",
        "Churn Value": 0,
        "Churn Score": 12,
        "CLTV": 3000,
        "Churn Reason": None,
    }
    return pd.DataFrame([row])


def test_prepare_restores_zero_total_charges_at_zero_tenure():
    frame = prepare_telco_frame(sample_frame())
    assert frame.loc[0, "Total Charges"] == 0.0


def test_model_features_exclude_leakage_and_cltv():
    forbidden = {"Churn Label", "Churn Value", "Churn Score", "Churn Reason", "CLTV"}
    assert forbidden.isdisjoint(MODEL_FEATURES)


def test_validate_rejects_duplicate_customer_ids():
    frame = pd.concat([sample_frame(), sample_frame()], ignore_index=True)
    with pytest.raises(ValueError, match="CustomerID must be unique"):
        validate_telco_frame(frame)


def test_load_xlsx_accepts_single_sheet_with_arbitrary_name(tmp_path):
    pytest.importorskip("openpyxl")
    from telco_churn.data import load_telco_data

    path = tmp_path / "renamed-workbook.xlsx"
    sample_frame().to_excel(path, sheet_name="Data", index=False)
    loaded = load_telco_data(path)
    assert loaded.loc[0, "CustomerID"] == "A-001"


def test_validate_rejects_non_numeric_cltv():
    frame = sample_frame()
    frame["CLTV"] = frame["CLTV"].astype(object)
    frame.loc[0, "CLTV"] = "unknown"
    with pytest.raises(ValueError, match="CLTV must be numeric"):
        validate_telco_frame(frame)
