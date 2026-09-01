"""Create synthetic claims for testing the FraudShield interface only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_test_claims(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, rows), unit="D")
    accident_site = rng.choice(["Highway", "Local", "Parking Lot"], rows, p=[.30, .50, .20])
    channel = rng.choice(["Broker", "Phone", "Online"], rows, p=[.30, .38, .32])
    prior_claims = rng.poisson(.7, rows)
    form_defects = rng.integers(0, 14, rows)
    total_claim = np.round(rng.lognormal(9.8, .55, rows), 2)
    police_report = rng.integers(0, 2, rows)
    risk = (-3.1 + .000025 * total_claim + .18 * prior_claims + .12 * form_defects
            + .50 * (accident_site == "Parking Lot") + .32 * (channel == "Online")
            + .30 * police_report)
    probability = 1 / (1 + np.exp(-risk))
    return pd.DataFrame({
        "claim_number": rng.integers(100000000, 999999999, rows),
        "age_of_driver": rng.integers(18, 81, rows),
        "gender": rng.choice(["M", "F"], rows),
        "marital_status": rng.integers(0, 2, rows),
        "safety_rating": rng.integers(2, 101, rows),
        "annual_income": np.round(rng.lognormal(11.0, .45, rows), 2),
        "high_education": rng.integers(0, 2, rows),
        "address_change": rng.integers(0, 2, rows),
        "property_status": rng.choice(["Own", "Rent"], rows, p=[.66, .34]),
        "zip_code": rng.choice([2000, 3000, 4000, 5000, 6000, 7000], rows),
        "claim_date": pd.Series(dates).dt.strftime("%m/%d/%Y"),
        "claim_day_of_week": pd.Series(dates).dt.day_name(),
        "accident_site": accident_site,
        "past_num_of_claims": prior_claims,
        "witness_present": rng.integers(0, 2, rows),
        "liab_prct": rng.integers(0, 101, rows),
        "channel": channel,
        "police_report": police_report,
        "age_of_vehicle": rng.integers(0, 15, rows),
        "vehicle_category": rng.choice(["Compact", "Medium", "Large"], rows, p=[.28, .48, .24]),
        "vehicle_price": np.round(rng.lognormal(10.1, .45, rows), 2),
        "vehicle_color": rng.choice(["black", "blue", "gray", "red", "silver", "white"], rows),
        "total_claim": total_claim,
        "injury_claim": np.round(total_claim * rng.uniform(.12, .42, rows), 2),
        "policy deductible": rng.choice([500, 1000, 2000], rows, p=[.30, .50, .20]),
        "annual premium": np.round(rng.normal(1250, 260, rows).clip(450, 2500), 2),
        "days open": np.round(rng.uniform(1, 30, rows), 2),
        "form defects": form_defects,
        "fraud reported": np.where(rng.random(rows) < probability, "Y", "N"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic insurance claims for UI testing.")
    parser.add_argument("--output", default="data/test/insurance_fraud_test_data.csv")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    make_test_claims(args.rows, args.seed).to_csv(destination, index=False)
    print(f"Created synthetic test data: {destination}")


if __name__ == "__main__":
    main()
