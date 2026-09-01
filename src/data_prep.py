"""Dataset schemas and cleaning for auto-insurance fraud modelling."""
from __future__ import annotations

import pandas as pd

DEMO_TARGET = "is_fraud"
DEMO_NUMERIC = ["claim_amount", "claim_age_days", "policyholder_age", "prior_claims", "vehicle_age_years", "annual_premium", "incident_hour"]
DEMO_CATEGORICAL = ["claim_type", "vehicle_type", "policy_state"]

INSURANCE_TARGET = "fraud reported"
INSURANCE_NUMERIC = [
    "age_of_driver", "safety_rating", "annual_income", "high_education", "address_change",
    "past_num_of_claims", "liab_prct", "police_report", "age_of_vehicle", "vehicle_price",
    "total_claim", "injury_claim", "policy deductible", "annual premium", "days open",
    "form defects", "claim_month", "claim_day_of_month",
]
INSURANCE_CATEGORICAL = [
    "gender", "marital_status", "property_status", "zip_code", "claim_day_of_week",
    "accident_site", "witness_present", "channel", "vehicle_category", "vehicle_color",
]


def prepare_data(data: pd.DataFrame, training: bool) -> tuple[pd.DataFrame, pd.Series | None, dict]:
    """Return cleaned features, optional labels, and schema metadata.

    Supports the supplied ``insurance_fraud_data.csv`` as well as the project's
    synthetic demo schema. Claim identifiers are deliberately excluded.
    """
    data = data.copy().replace("*", pd.NA)
    if INSURANCE_TARGET in data.columns or "claim_number" in data.columns:
        if training and INSURANCE_TARGET not in data.columns:
            raise ValueError(f"Dataset is missing training target: {INSURANCE_TARGET!r}")
        if "claim_date" not in data.columns:
            raise ValueError("Dataset is missing required column: 'claim_date'")
        dates = pd.to_datetime(data["claim_date"], format="mixed", errors="coerce")
        data["claim_month"] = dates.dt.month
        data["claim_day_of_month"] = dates.dt.day
        features = INSURANCE_NUMERIC + INSURANCE_CATEGORICAL
        missing = set(features) - set(data.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        for column in INSURANCE_NUMERIC:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        y = None
        if training:
            labels = data[INSURANCE_TARGET].map({"Y": 1, "N": 0})
            valid = labels.notna()
            data, labels = data.loc[valid], labels.loc[valid].astype(int)
            if labels.nunique() < 2:
                raise ValueError("The fraud target must contain both Y and N labels.")
            y = labels
        return data[features], y, {"features": features, "numeric": INSURANCE_NUMERIC, "categorical": INSURANCE_CATEGORICAL, "claim_value_column": "total_claim", "dataset": "insurance_fraud_data"}

    features = DEMO_NUMERIC + DEMO_CATEGORICAL
    missing = set(features + ([DEMO_TARGET] if training else [])) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    y = data[DEMO_TARGET].astype(int) if training else None
    if training and (not set(y.unique()).issubset({0, 1}) or y.nunique() < 2):
        raise ValueError("is_fraud must contain both 0 and 1 labels.")
    return data[features], y, {"features": features, "numeric": DEMO_NUMERIC, "categorical": DEMO_CATEGORICAL, "claim_value_column": "claim_amount", "dataset": "demo"}
