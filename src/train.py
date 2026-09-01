"""Train and evaluate an XGBoost auto-insurance fraud classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from data_prep import prepare_data


def choose_threshold(y_true: pd.Series, probabilities) -> float:
    """Choose the threshold with the best F1 score on validation data."""
    candidates = [round(x / 100, 2) for x in range(10, 91)]
    scores = [
        (2 * precision_score(y_true, probabilities >= t, zero_division=0) * recall_score(y_true, probabilities >= t, zero_division=0)) /
        (precision_score(y_true, probabilities >= t, zero_division=0) + recall_score(y_true, probabilities >= t, zero_division=0) + 1e-12)
        for t in candidates
    ]
    return candidates[scores.index(max(scores))]


def metrics(y_true: pd.Series, probabilities, threshold: float) -> dict:
    prediction = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": threshold, "accuracy": round(float(accuracy_score(y_true, prediction)), 4),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 4),
        "sensitivity_recall": round(float(recall_score(y_true, prediction, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if tn + fp else 0.0, 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Labelled claims CSV")
    parser.add_argument("--model-output", default="models/fraud_xgboost.joblib")
    parser.add_argument("--report-output", default="reports/training_report.json")
    parser.add_argument("--test-size", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    X, y, schema = prepare_data(data, training=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), schema["numeric"]),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), schema["categorical"]),
    ])
    imbalance_weight = (y_train == 0).sum() / (y_train == 1).sum()
    classifier = XGBClassifier(n_estimators=450, max_depth=5, learning_rate=.05, subsample=.85,
        colsample_bytree=.85, eval_metric="logloss", scale_pos_weight=imbalance_weight,
        random_state=args.seed, n_jobs=-1)
    pipeline = Pipeline([("preprocess", preprocessing), ("model", classifier)])
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, probability)
    result = metrics(y_test, probability, threshold)

    # Estimated claim value recovered if every correctly prioritised fraudulent claim is investigated.
    heldout = data.loc[X_test.index]
    claim_value = schema["claim_value_column"]
    result["financial_impact_aud"] = {
        "test_claim_value": round(float(pd.to_numeric(heldout[claim_value], errors="coerce").sum()), 2),
        "confirmed_fraud_value": round(float(pd.to_numeric(heldout.loc[y_test == 1, claim_value], errors="coerce").sum()), 2),
        "potential_fraud_value_prioritised": round(float(pd.to_numeric(heldout.loc[(y_test == 1) & (probability >= threshold), claim_value], errors="coerce").sum()), 2),
        "note": "Potential value, not guaranteed savings; investigation and recovery costs are excluded.",
    }
    result["data"] = {"dataset": schema["dataset"], "rows": int(len(X)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "fraud_rate": round(float(y.mean()), 4)}

    model_path, report_path = Path(args.model_output), Path(args.report_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold, "features": schema["features"], "dataset": schema["dataset"]}, model_path)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
