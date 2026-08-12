from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED_XLSX_SHA256 = "dd5ecd947dcfc26ebd2d82716879d600a10d99e04166c18174c33976cc60f5d2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(data: dict, sheet: str) -> list[dict]:
    header = data[sheet][0]
    return [
        dict(zip(header, row))
        for row in data[sheet][1:]
        if any(value is not None for value in row)
    ]


def audit_set4_json(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    demand = [
        row
        for row in _rows(data, "Demand")
        if row["probleminstanceid"] == "SET4"
    ]
    keys = [(row["materialid"], float(row["deliverydate"])) for row in demand]
    products = {key[0] for key in keys}
    dates = sorted({key[1] for key in keys})

    problem = next(
        row
        for row in _rows(data, "ProblemInstance")
        if row["probleminstanceid"] == "SET4"
    )
    capacity = [
        row
        for row in _rows(data, "Capacity")
        if row["probleminstanceid"] == "SET4"
    ]
    start = float(problem["planningstartdate"])
    end = max(float(row["validitydateto"]) for row in capacity)
    official = [date for date in dates if start <= date <= end]

    return {
        "date_count": len(dates),
        "product_count": len(products),
        "row_count": len(demand),
        "duplicate_key_count": len(keys) - len(set(keys)),
        "missing_quantity_count": sum(row["quantity"] is None for row in demand),
        "negative_quantity_count": sum(float(row["quantity"] or 0) < 0 for row in demand),
        "weekly_gap_violation_count": sum(
            abs((right - left) - 7.0) > 1e-12
            for left, right in zip(dates, dates[1:])
        ),
        "official_date_count": len(official),
    }

