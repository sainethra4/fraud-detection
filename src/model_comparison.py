"""Reproducible, leakage-safe model comparison for insurance fraud detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from data_prep import prepare_data

RANDOM_STATE = 42


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    # Tree algorithms do not require scaled values. Imputing/encoding happens inside CV folds.
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def _without_train_duplicates(X_train: pd.DataFrame, X_other: pd.DataFrame, schema: dict):
    """Remove exact duplicate columns discovered from training data only."""
    duplicated = X_train.columns[X_train.T.duplicated()].tolist()
    kept_numeric = [item for item in schema["numeric"] if item not in duplicated]
    kept_categorical = [item for item in schema["categorical"] if item not in duplicated]
    return X_train.drop(columns=duplicated), X_other.drop(columns=duplicated), kept_numeric, kept_categorical, duplicated


def _threshold_from_validation(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose the validation threshold that maximises F1; the test set is never consulted."""
    thresholds = np.arange(.10, .91, .01)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def _evaluation(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    recall = float(recall_score(y_true, predicted, zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": recall,
        "sensitivity_recall": recall,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1_score": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "threshold": float(threshold),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
    }


def _curve_data(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    return {
        "roc": pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr}),
        "pr": pd.DataFrame({"recall": recall, "precision": precision}),
    }


def _feature_importance(model: Pipeline, limit: int = 15) -> pd.DataFrame:
    transformer = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    values = classifier.feature_importances_
    names = transformer.get_feature_names_out()
    labels = [name.replace("numeric__", "").replace("categorical__", "").replace("onehot__", "") for name in names]
    return (pd.DataFrame({"Feature": labels, "Importance": values})
              .sort_values("Importance", ascending=False).head(limit).reset_index(drop=True))


@dataclass
class ComparisonResult:
    xgb_bundle: dict
    metrics: pd.DataFrame
    details: dict
    curves: dict
    feature_importance: pd.DataFrame
    removed_columns: list[str]


def run_comparison(data: pd.DataFrame, search_iterations: int = 2, tuning_row_limit: int = 4000) -> ComparisonResult:
    """Tune models using only training data, then evaluate once on untouched test data."""
    X, y, schema = prepare_data(data, training=True)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test, numeric, categorical, removed = _without_train_duplicates(X_train_raw, X_test_raw, schema)
    preprocessor = _preprocessor(numeric, categorical)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    negative_to_positive = float((y_train == 0).sum() / (y_train == 1).sum())
    if len(X_train) > tuning_row_limit:
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train, train_size=tuning_row_limit, random_state=RANDOM_STATE, stratify=y_train
        )
    else:
        X_tune, y_tune = X_train, y_train

    xgb_pipeline = Pipeline([("preprocess", preprocessor), ("model", XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1,
    ))])
    xgb_parameters = {
        "model__n_estimators": [80, 120, 160], "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [.02, .03, .05, .08], "model__min_child_weight": [1, 3, 5],
        "model__subsample": [.70, .85, 1.0], "model__colsample_bytree": [.70, .85, 1.0],
        "model__gamma": [0, .1, .3], "model__reg_alpha": [0, .01, .1],
        "model__reg_lambda": [1, 2, 5], "model__scale_pos_weight": [1.0, round(negative_to_positive, 3)],
    }
    xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_parameters, n_iter=search_iterations,
        scoring="roc_auc", cv=cv, refit=True, random_state=RANDOM_STATE, n_jobs=1)
    xgb_search.fit(X_tune, y_tune)
    xgb_best = {key.replace("model__", ""): value for key, value in xgb_search.best_params_.items()}

    # Validation data chooses the stopping iteration and operating threshold; the test set stays untouched.
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train, y_train, test_size=.20, random_state=RANDOM_STATE, stratify=y_train
    )
    early_preprocessor = clone(preprocessor).fit(X_fit)
    early_xgb = XGBClassifier(**xgb_best, objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=30, random_state=RANDOM_STATE, n_jobs=1)
    early_xgb.fit(early_preprocessor.transform(X_fit), y_fit,
                  eval_set=[(early_preprocessor.transform(X_validation), y_validation)], verbose=False)
    xgb_threshold = _threshold_from_validation(y_validation, early_xgb.predict_proba(early_preprocessor.transform(X_validation))[:, 1])
    xgb_best["n_estimators"] = max(1, int(getattr(early_xgb, "best_iteration", xgb_best["n_estimators"] - 1)) + 1)
    final_xgb = Pipeline([("preprocess", clone(preprocessor)), ("model", XGBClassifier(
        **xgb_best, objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1,
    ))])
    final_xgb.fit(X_train, y_train)

    rf_pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=1,
    ))])
    rf_parameters = {
        "model__n_estimators": [60, 90, 120], "model__max_depth": [8, 12, 16, None],
        "model__min_samples_split": [2, 5, 10], "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", .6, 1.0], "model__class_weight": [None, "balanced"],
    }
    rf_search = RandomizedSearchCV(rf_pipeline, rf_parameters, n_iter=search_iterations,
        scoring="roc_auc", cv=cv, refit=True, random_state=RANDOM_STATE, n_jobs=1)
    rf_search.fit(X_tune, y_tune)
    rf_validation_model = clone(rf_search.best_estimator_).fit(X_fit, y_fit)
    rf_threshold = _threshold_from_validation(y_validation, rf_validation_model.predict_proba(X_validation)[:, 1])
    final_rf = clone(rf_search.best_estimator_).fit(X_train, y_train)

    xgb_probability = final_xgb.predict_proba(X_test)[:, 1]
    rf_probability = final_rf.predict_proba(X_test)[:, 1]
    xgb_result = _evaluation(y_test, xgb_probability, xgb_threshold)
    rf_result = _evaluation(y_test, rf_probability, rf_threshold)
    heldout = data.loc[X_test_raw.index]
    claim_value = pd.to_numeric(heldout[schema["claim_value_column"]], errors="coerce")
    xgb_flagged = xgb_probability >= xgb_threshold
    xgb_result["financial_impact_aud"] = {
        "test_claim_value": round(float(claim_value.sum()), 2),
        "confirmed_fraud_value": round(float(claim_value.loc[y_test == 1].sum()), 2),
        "potential_fraud_value_prioritised": round(float(claim_value.loc[(y_test == 1) & xgb_flagged].sum()), 2),
        "note": "Potential value, not guaranteed savings; investigation and recovery costs are excluded.",
    }
    xgb_result["data"] = {"dataset": schema["dataset"], "rows": int(len(X)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "fraud_rate": round(float(y.mean()), 4)}
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc", "pr_auc"]
    comparison = pd.DataFrame([
        {"Algorithm": "XGBoost", **{name: xgb_result[name] for name in metric_names}},
        {"Algorithm": "Random Forest", **{name: rf_result[name] for name in metric_names}},
    ])
    differences = {name: (xgb_result[name] - rf_result[name]) * 100 for name in metric_names}
    relative_differences = {
        name: ((xgb_result[name] - rf_result[name]) / rf_result[name] * 100) if rf_result[name] else 0.0
        for name in metric_names
    }
    return ComparisonResult(
        xgb_bundle={"pipeline": final_xgb, "threshold": xgb_threshold, "features": list(X_train.columns), "dataset": schema["dataset"]},
        metrics=comparison,
        details={"XGBoost": xgb_result, "Random Forest": rf_result, "metric_differences_percentage_points": differences,
                 "relative_metric_differences_percent": relative_differences,
                 "xgb_cv_roc_auc": float(xgb_search.best_score_), "rf_cv_roc_auc": float(rf_search.best_score_),
                 "tuning_rows": int(len(X_tune)),
                 "fraud_rate": float(y.mean()), "class_ratio_negative_to_positive": negative_to_positive,
                 "xgb_parameters": xgb_best, "rf_parameters": rf_search.best_params_},
        curves={"XGBoost": _curve_data(y_test, xgb_probability), "Random Forest": _curve_data(y_test, rf_probability)},
        feature_importance=_feature_importance(final_xgb), removed_columns=removed,
    )
