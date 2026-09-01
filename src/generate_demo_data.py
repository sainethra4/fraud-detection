"""Generate synthetic claims solely to exercise the training pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_claims(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    claim_type = rng.choice(["collision", "theft", "windscreen", "liability"], rows, p=[.58, .12, .18, .12])
    vehicle_type = rng.choice(["sedan", "suv", "ute", "hatchback"], rows, p=[.32, .34, .14, .20])
    state = rng.choice(["NSW", "VIC", "QLD", "WA", "SA"], rows, p=[.33, .27, .20, .12, .08])
    claim_amount = np.round(rng.lognormal(8.0, .75, rows), 2)
    claim_age_days = rng.integers(0, 3650, rows)
    prior_claims = rng.poisson(.55, rows)
    incident_hour = rng.integers(0, 24, rows)
    risk = (-4.2 + .00014 * claim_amount + .95 * (claim_type == "theft")
            + .85 * (claim_age_days < 45) + .28 * prior_claims
            + .50 * np.isin(incident_hour, [0, 1, 2, 3, 4]))
    fraud_probability = 1 / (1 + np.exp(-risk))
    return pd.DataFrame({
        "claim_id": [f"CLM-{i:06d}" for i in range(1, rows + 1)],
        "claim_amount": claim_amount,
        "claim_age_days": claim_age_days,
        "policyholder_age": rng.integers(18, 85, rows),
        "prior_claims": prior_claims,
        "vehicle_age_years": rng.integers(0, 22, rows),
        "annual_premium": np.round(rng.normal(1350, 330, rows).clip(450, 3500), 2),
        "claim_type": claim_type,
        "vehicle_type": vehicle_type,
        "policy_state": state,
        "incident_hour": incident_hour,
        "is_fraud": rng.binomial(1, fraud_probability),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/demo_claims.csv")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_claims(args.rows, args.seed).to_csv(output, index=False)
    print(f"Created synthetic demo data: {output}")


if __name__ == "__main__":
    main()

