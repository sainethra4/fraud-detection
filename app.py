"""Streamlit interface for the auto-insurance fraud detection project."""
from __future__ import annotations

import json
import sys
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_prep import prepare_data  # noqa: E402
from model_comparison import run_comparison  # noqa: E402
from train import choose_threshold, metrics  # noqa: E402
from pdf_report import build_performance_pdf  # noqa: E402

MODEL_PATH = Path("models/insurance_fraud_xgboost.joblib")
COMPARISON_CACHE_DIR = Path("models/comparison_cache")
COMPARISON_CACHE_VERSION = "v2"


def train_model(data: pd.DataFrame) -> tuple[dict, dict]:
    """Fit the pipeline and return its saved-model bundle plus evaluation results."""
    features, labels, schema = prepare_data(data, training=True)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=.2, random_state=42, stratify=labels
    )
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), schema["numeric"]),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), schema["categorical"]),
    ])
    imbalance_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=450, max_depth=5, learning_rate=.05, subsample=.85,
        colsample_bytree=.85, eval_metric="logloss", scale_pos_weight=imbalance_weight,
        random_state=42, n_jobs=-1,
    )
    pipeline = Pipeline([("preprocess", preprocessing), ("model", model)])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, probabilities)
    report = metrics(y_test, probabilities, threshold)
    heldout = data.loc[X_test.index]
    claim_value = schema["claim_value_column"]
    report["financial_impact_aud"] = {
        "test_claim_value": round(float(pd.to_numeric(heldout[claim_value], errors="coerce").sum()), 2),
        "confirmed_fraud_value": round(float(pd.to_numeric(heldout.loc[y_test == 1, claim_value], errors="coerce").sum()), 2),
        "potential_fraud_value_prioritised": round(float(pd.to_numeric(heldout.loc[(y_test == 1) & (probabilities >= threshold), claim_value], errors="coerce").sum()), 2),
        "note": "Potential value, not guaranteed savings; investigation and recovery costs are excluded.",
    }
    report["data"] = {"rows": int(len(features)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "fraud_rate": round(float(labels.mean()), 4)}
    bundle = {"pipeline": pipeline, "threshold": threshold, "features": schema["features"], "dataset": schema["dataset"]}
    return bundle, report


def score_claims(data: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    features, _, _ = prepare_data(data, training=False)
    probabilities = bundle["pipeline"].predict_proba(features[bundle["features"]])[:, 1]
    scored = data.copy()
    scored["fraud_probability"] = probabilities.round(4)
    scored["investigate"] = probabilities >= bundle["threshold"]
    return scored.sort_values("fraud_probability", ascending=False)


def cached_tuned_comparison(csv_content: bytes):
    """Load a matching finished comparison or create it once and cache it on disk.

    A disk cache deliberately avoids holding model objects in Streamlit's resource
    lifecycle, which can leave background resources attached during reload/shutdown.
    """
    fingerprint = sha256(COMPARISON_CACHE_VERSION.encode("utf-8") + csv_content).hexdigest()
    cache_path = COMPARISON_CACHE_DIR / f"comparison_{fingerprint}.joblib"
    if cache_path.exists():
        try:
            return joblib.load(cache_path), True
        except Exception:
            # A partial/corrupt cache is ignored and replaced atomically below.
            pass
    comparison = run_comparison(pd.read_csv(BytesIO(csv_content)), search_iterations=2)
    COMPARISON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    joblib.dump(comparison, temporary_path)
    temporary_path.replace(cache_path)
    return comparison, False


st.set_page_config(page_title="FraudShield", page_icon="FS", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background: #F7F9FC; }
    [data-testid="stSidebar"] { background: #102A43; }
    [data-testid="stSidebar"] * { color: #F8FAFC !important; }
    .hero { background: linear-gradient(120deg, #102A43, #1E5A87); border-radius: 18px; padding: 2rem 2.2rem; color: white; margin-bottom: 1.25rem; }
    .hero h1 { color: white; margin: 0; font-size: 2.35rem; }
    .hero p { margin: .4rem 0 0; color: #D9EAF7; font-size: 1.05rem; }
    .stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 600; }
    [data-testid="stMetric"] { background: white; border: 1px solid #DCE5EF; border-radius: 12px; padding: .8rem; }
    [data-baseweb="tab-list"] { gap: 1.25rem; }
    [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## FraudShield")
    st.caption("Insurance investigation workspace")
    st.divider()
    st.markdown("**Workflow**")
    st.markdown("1. Upload claims data\n2. Review model results\n3. Identify high-risk claims\n4. Download reports")
    st.divider()
    st.caption("Scores support human review. They do not prove fraud or automate adverse decisions.")

st.markdown("<div class='hero'><h1>Insurance Fraud Detection</h1><p>Analyze claim patterns, identify high-risk claims, and support faster and more informed claim investigations.</p></div>", unsafe_allow_html=True)
st.warning("Decision-support only: results help prioritise claims for human review and do not determine claim outcomes automatically.")

uploaded = st.file_uploader("Upload claims data (CSV)", type="csv")
if uploaded is None:
    st.markdown("Upload `insurance_fraud_data.csv` to train the model and view fraud-risk predictions.")
    st.stop()

try:
    uploaded_content = uploaded.getvalue()
    claims = pd.read_csv(BytesIO(uploaded_content))
except Exception as error:
    st.error(f"The CSV could not be read: {error}")
    st.stop()

st.success(f"Loaded {len(claims):,} claims and {len(claims.columns)} columns.")
train_tab, compare_tab, score_tab, impact_tab, data_tab = st.tabs(["Model performance", "Compare models", "High-risk claims", "Impact & compliance", "Data preview"])

with train_tab:
    st.subheader("Fraud-risk model assessment")
    st.write("Create a model assessment to understand how well the system identifies claims that may need closer review.")
    if st.button("Create model assessment", type="primary"):
        try:
            with st.spinner("Training XGBoost and evaluating its holdout performance..."):
                bundle, report = train_model(claims)
                MODEL_PATH.parent.mkdir(exist_ok=True)
                joblib.dump(bundle, MODEL_PATH)
                st.session_state["model"] = bundle
                st.session_state["report"] = report
            st.success("Model assessment complete.")
        except Exception as error:
            st.error(f"Training failed: {error}")
    if "report" in st.session_state:
        report = st.session_state["report"]
        first, second, third, fourth = st.columns(4)
        first.metric("ROC-AUC", report["roc_auc"])
        second.metric("Fraud recall", f"{report['sensitivity_recall']:.2%}")
        third.metric("Precision", f"{report['precision']:.2%}")
        fourth.metric("Accuracy", f"{report['accuracy']:.2%}")
        st.subheader("Confusion matrix")
        st.json(report["confusion_matrix"])
        st.subheader("Estimated financial impact (AUD)")
        impact = report["financial_impact_aud"]
        st.metric("Potential fraud value prioritised", f"${impact['potential_fraud_value_prioritised']:,.2f}")
        st.caption(impact["note"])
        report_actions, report_space = st.columns([1, 3])
        with report_actions:
            st.download_button("Download PDF report", build_performance_pdf(report), "fraudshield_performance_report.pdf", "application/pdf", type="primary")
            st.download_button("Download raw JSON", json.dumps(report, indent=2), "training_report.json", "application/json")
        with st.expander("Model Details"):
            st.write(f"Review threshold: {report['threshold']:.2f}")
            st.json(report["confusion_matrix"])

with compare_tab:
    st.subheader("XGBoost vs Random Forest")
    st.write("Compare two fraud-risk models to identify the approach that provides the strongest overall investigation support.")
    if st.button("Run model comparison", type="primary"):
        try:
            with st.spinner("Preparing the model comparison..."):
                comparison, reused = cached_tuned_comparison(uploaded_content)
                st.session_state["comparison"] = comparison
                st.session_state["model"] = st.session_state["comparison"].xgb_bundle
                st.session_state["report"] = st.session_state["comparison"].details["XGBoost"]
                MODEL_PATH.parent.mkdir(exist_ok=True)
                joblib.dump(st.session_state["model"], MODEL_PATH)
            st.success("Previously completed comparison loaded." if reused else "Comparison complete. Future runs with this same file will reuse these results.")
        except Exception as error:
            st.error(f"Comparison failed: {error}")
    if "comparison" in st.session_state:
        result = st.session_state["comparison"]
        comparison = result.metrics.copy()
        details = result.details
        xgb_accuracy = details["XGBoost"]["accuracy"]
        rf_accuracy = details["Random Forest"]["accuracy"]
        best = "XGBoost" if details["XGBoost"]["roc_auc"] > details["Random Forest"]["roc_auc"] else "Random Forest"
        left, middle, right = st.columns(3)
        left.metric("Recommended model", best, "Highest test ROC-AUC")
        middle.metric("XGBoost test accuracy", f"{xgb_accuracy:.2%}")
        right.metric("Random Forest test accuracy", f"{rf_accuracy:.2%}")
        if best == "XGBoost":
            st.success("XGBoost is visually highlighted because it has the stronger ROC-AUC on this untouched test set. Review all metrics before selecting a production model.")
        else:
            st.info("Random Forest has the stronger ROC-AUC on this test set. The dashboard reports this result without changing any metric.")
        st.subheader("Test-set performance comparison")
        chart_data = comparison.set_index("Algorithm")[["accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"]]
        chart_data.columns = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC"]
        st.subheader("Performance comparison")
        st.bar_chart(chart_data, use_container_width=True)
        display = comparison.rename(columns={"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall / sensitivity", "specificity": "Specificity", "f1_score": "F1-score", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
        differences = pd.DataFrame([{
            "Metric": key.replace("_", " ").upper(),
            "Difference (percentage points)": value,
            "Relative difference (%)": details["relative_metric_differences_percent"][key],
        } for key, value in details["metric_differences_percentage_points"].items()])
        st.dataframe(display.style.format({column: "{:.2%}" for column in display.columns if column != "Algorithm"}), use_container_width=True, hide_index=True)
        st.subheader("Metric difference")
        st.dataframe(differences.style.format({"Difference (percentage points)": "{:+.2f}", "Relative difference (%)": "{:+.2f}"}), use_container_width=True, hide_index=True)
        matrix_xgb, matrix_rf = st.columns(2)
        with matrix_xgb:
            st.markdown("**XGBoost confusion matrix**")
            st.json(details["XGBoost"]["confusion_matrix"])
        with matrix_rf:
            st.markdown("**Random Forest confusion matrix**")
            st.json(details["Random Forest"]["confusion_matrix"])
        st.subheader("ROC curves")
        roc_chart_data = pd.concat([curve["roc"].assign(Algorithm=name) for name, curve in result.curves.items()])
        st.altair_chart(alt.Chart(roc_chart_data).mark_line().encode(x=alt.X("false_positive_rate", title="False positive rate"), y=alt.Y("true_positive_rate", title="True positive rate"), color="Algorithm", tooltip=["Algorithm", "false_positive_rate", "true_positive_rate"]), use_container_width=True)
        st.subheader("Precision-recall curves")
        pr_chart_data = pd.concat([curve["pr"].assign(Algorithm=name) for name, curve in result.curves.items()])
        st.altair_chart(alt.Chart(pr_chart_data).mark_line().encode(x=alt.X("recall", title="Recall"), y=alt.Y("precision", title="Precision"), color="Algorithm", tooltip=["Algorithm", "recall", "precision"]), use_container_width=True)
        st.subheader("XGBoost feature importance")
        importance_chart = alt.Chart(result.feature_importance).mark_bar().encode(x=alt.X("Importance", title="Importance"), y=alt.Y("Feature", sort="-x", title=None), tooltip=["Feature", alt.Tooltip("Importance", format=".3f")])
        st.altair_chart(importance_chart, use_container_width=True)
        st.dataframe(result.feature_importance.style.format({"Importance": "{:.3f}"}), use_container_width=True, hide_index=True)
        st.caption("The recommended model is selected using its test performance. Results should support investigator judgement rather than replace it.")
        report_payload = {"details": details, "metrics": comparison.to_dict(orient="records"), "feature_importance": result.feature_importance.to_dict(orient="records"), "removed_duplicate_columns": result.removed_columns}
        st.download_button("Download comparison report (JSON)", json.dumps(report_payload, indent=2, default=str), "tuned_model_comparison.json", "application/json")
        st.download_button("Download comparison CSV", comparison.to_csv(index=False).encode("utf-8"), "tuned_model_comparison.csv", "text/csv")
        with st.expander("Model Details"):
            st.write("Both models were evaluated on the same reserved test claims. Tuning and threshold selection were performed before this final evaluation.")
            st.write(f"Fraud rate: {details['fraud_rate']:.2%}; class ratio: {details['class_ratio_negative_to_positive']:.2f}:1.")
            st.write(f"Cross-validation ROC-AUC - XGBoost: {details['xgb_cv_roc_auc']:.4f}; Random Forest: {details['rf_cv_roc_auc']:.4f}. Tuning used {details['tuning_rows']:,} training claims only.")
            st.json({"XGBoost settings": details["xgb_parameters"], "Random Forest settings": details["rf_parameters"], "Removed exact duplicate columns": result.removed_columns})

with score_tab:
    st.subheader("High-risk claims for review")
    bundle = st.session_state.get("model")
    if bundle is None and MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        st.caption("Using the most recently saved model.")
    if bundle is None:
        st.warning("Create a model assessment or run a comparison first.")
    elif st.button("Score uploaded claims"):
        try:
            scored = score_claims(claims, bundle)
            st.session_state["scored"] = scored
        except Exception as error:
            st.error(f"Scoring failed: {error}")
    if "scored" in st.session_state:
        scored = st.session_state["scored"]
        investigate_count = int(scored["investigate"].sum())
        st.metric("Claims recommended for investigation", f"{investigate_count:,}")
        st.dataframe(scored.head(100), use_container_width=True, hide_index=True)
        st.download_button("Download scored claims", scored.to_csv(index=False).encode("utf-8"), "insurance_predictions.csv", "text/csv")

with impact_tab:
    st.subheader("Financial impact and cost-saving measures")
    if "report" in st.session_state:
        impact = st.session_state["report"]["financial_impact_aud"]
        left, middle, right = st.columns(3)
        left.metric("Test claim value", f"${impact['test_claim_value']:,.0f}")
        middle.metric("Confirmed fraud value", f"${impact['confirmed_fraud_value']:,.0f}")
        right.metric("Potential fraud value prioritised", f"${impact['potential_fraud_value_prioritised']:,.0f}")
        st.caption(impact["note"])
    else:
        st.info("Train the model to calculate financial impact on the hold-out test sample.")
    financial_doc = Path("docs/financial_recommendations.md").read_text(encoding="utf-8")
    st.markdown(financial_doc)
    st.download_button("Download financial recommendations", financial_doc, "financial_recommendations.md", "text/markdown")
    st.divider()
    st.subheader("Australian compliance checklist")
    compliance_doc = Path("docs/australian_compliance.md").read_text(encoding="utf-8")
    st.markdown(compliance_doc)
    st.download_button("Download compliance documentation", compliance_doc, "australian_compliance.md", "text/markdown")

with data_tab:
    st.dataframe(claims.head(100), use_container_width=True, hide_index=True)
    if "fraud reported" in claims.columns:
        st.bar_chart(claims["fraud reported"].value_counts(dropna=False))
