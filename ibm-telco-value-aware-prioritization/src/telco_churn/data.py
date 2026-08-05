from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = set(MODEL_FEATURES) | {
    "CustomerID",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
}
LEAKAGE_COLUMNS = {"Churn Label", "Churn Value", "Churn Score", "Churn Reason"}
POLICY_ONLY_COLUMNS = {"CLTV"}


@dataclass(frozen=True)
class TelcoDataset:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: np.ndarray
    cltv: np.ndarray


def load_telco_data(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Place the IBM Telco CSV/XLSX file in data/raw; see data/README.md."
        )
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".xlsx", ".xlsm"}:
        workbook = pd.ExcelFile(data_path, engine="openpyxl")
        if "Telco_Churn" in workbook.sheet_names:
            return pd.read_excel(workbook, sheet_name="Telco_Churn")
        matching_sheets: list[str] = []
        for sheet_name in workbook.sheet_names:
            header = pd.read_excel(workbook, sheet_name=sheet_name, nrows=0)
            if REQUIRED_COLUMNS.issubset(header.columns):
                matching_sheets.append(sheet_name)
        if len(matching_sheets) == 1:
            return pd.read_excel(workbook, sheet_name=matching_sheets[0])
        if len(workbook.sheet_names) == 1:
            return pd.read_excel(workbook, sheet_name=workbook.sheet_names[0])
        raise ValueError(
            "Could not identify the Telco data sheet. Available sheets: "
            + ", ".join(workbook.sheet_names)
        )
    raise ValueError(f"Unsupported input type: {suffix}. Use .csv or .xlsx.")


def validate_telco_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if frame["CustomerID"].isna().any():
        raise ValueError("CustomerID must not contain missing values.")
    if frame["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be unique.")
    if not frame["Churn Value"].isin([0, 1]).all():
        raise ValueError("Churn Value must contain only 0 and 1.")
    label_as_int = frame["Churn Label"].map({"No": 0, "Yes": 1})
    if label_as_int.isna().any() or not label_as_int.equals(frame["Churn Value"].astype(int)):
        raise ValueError("Churn Label and Churn Value are inconsistent.")
    cltv = pd.to_numeric(frame["CLTV"], errors="coerce")
    if cltv.isna().any():
        raise ValueError("CLTV must be numeric and non-missing.")
    if (cltv < 0).any():
        raise ValueError("CLTV must be non-negative.")


def prepare_telco_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    validate_telco_frame(prepared)
    prepared["Total Charges"] = pd.to_numeric(prepared["Total Charges"], errors="coerce")
    zero_tenure = prepared["Tenure Months"].eq(0) & prepared["Total Charges"].isna()
    prepared.loc[zero_tenure, "Total Charges"] = 0.0
    median_total = prepared["Total Charges"].median()
    prepared["Total Charges"] = prepared["Total Charges"].fillna(median_total)
    return prepared


def build_dataset(frame: pd.DataFrame, max_rows: int | None = None, seed: int = 42) -> TelcoDataset:
    prepared = prepare_telco_frame(frame)
    if max_rows is not None and len(prepared) > max_rows:
        pieces = []
        for _, group in prepared.groupby("Churn Value", sort=False):
            n_group = max(1, round(max_rows * len(group) / len(prepared)))
            pieces.append(group.sample(min(n_group, len(group)), random_state=seed))
        prepared = pd.concat(pieces).sample(frac=1.0, random_state=seed).head(max_rows).reset_index(drop=True)
    features = prepared[MODEL_FEATURES].copy()
    target = prepared["Churn Value"].astype(int).to_numpy()
    cltv = prepared["CLTV"].astype(float).to_numpy()
    return TelcoDataset(prepared, features, target, cltv)


def data_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    total_charges = pd.to_numeric(frame["Total Charges"], errors="coerce")
    rows = [
        ("Rows", len(frame), "PASS"),
        ("Columns", frame.shape[1], "PASS"),
        ("Duplicate CustomerID", int(frame["CustomerID"].duplicated().sum()), "PASS"),
        ("Exact duplicate rows", int(frame.duplicated().sum()), "PASS"),
        ("Missing Total Charges", int(total_charges.isna().sum()), "DOCUMENTED"),
        ("Missing Churn Reason", int(frame["Churn Reason"].isna().sum()), "EXPECTED_POST_OUTCOME"),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "status"])
