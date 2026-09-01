"""Score auto-insurance claims using a trained XGBoost pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from data_prep import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Unlabelled or labelled claims CSV")
    parser.add_argument("--model", default="models/fraud_xgboost.joblib")
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    data = pd.read_csv(args.data)
    features, _, _ = prepare_data(data, training=False)
    missing = set(bundle["features"]) - set(features.columns)
    if missing:
        raise ValueError(f"Dataset is missing model features: {sorted(missing)}")
    probability = bundle["pipeline"].predict_proba(features[bundle["features"]])[:, 1]
    output = data.copy()
    output["fraud_probability"] = probability.round(4)
    output["investigate"] = probability >= bundle["threshold"]
    output = output.sort_values("fraud_probability", ascending=False)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(f"Scored {len(output)} claims. Saved: {destination}")


if __name__ == "__main__":
    main()
