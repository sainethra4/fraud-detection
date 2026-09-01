# Auto Insurance Fraud Claim Detection

An XGBoost-based Python project for prioritising potentially fraudulent auto-insurance claims. It follows the supplied proposal's objectives: preprocessing claims data, detecting fraud patterns, reporting sensitivity and specificity, and estimating financial impact.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the web dashboard

```powershell
streamlit run app.py
```

Your browser will open the FraudShield dashboard. Upload `insurance_fraud_data.csv`, select **Train model**, then select **Score uploaded claims** and download the scored results.

After training, use **Download PDF report** for a plain-English, two-page summary of performance metrics, classification results, financial impact, and recommended next actions.

## Model comparison

The **Compare models** dashboard tab evaluates XGBoost and Random Forest on the same untouched 80/20 test split. It tunes both models with 3-fold cross-validation on a stratified subset of training data only, uses validation data to select operating thresholds, and uses early stopping for XGBoost. It presents performance and difference tables, confusion matrices, ROC and precision-recall curves, and XGBoost feature importance. A completed comparison is cached on disk using the uploaded dataset fingerprint, so repeating the same comparison reloads results instead of training again. The displayed winner is calculated from actual test results; the dashboard does not alter metrics to force a preferred outcome.

## Synthetic test dataset

Use [insurance_fraud_test_data.csv](data/test/insurance_fraud_test_data.csv) to test the dashboard without changing the supplied dataset. It is synthetic, has the same expected columns, and includes a `fraud reported` label so it can be used for both training and scoring. It must not be used for real insurance decisions.

## Proposal deliverables

- Predictive analytics model and downloadable scored claims.
- Performance report with accuracy, sensitivity/recall, specificity, precision, ROC-AUC, PR-AUC, threshold, and confusion matrix.
- Financial assessment and operational recommendations: [financial recommendations](docs/financial_recommendations.md).
- Australian regulatory and governance checklist: [compliance documentation](docs/australian_compliance.md). This is a prototype checklist, not legal advice.

## Run the demo

The demo data is synthetic and is only for validating the workflow; replace it with anonymised claims data before drawing business conclusions.

```powershell
python src\generate_demo_data.py --output data\raw\demo_claims.csv
python src\train.py --data data\raw\demo_claims.csv
python src\predict.py --data data\raw\demo_claims.csv --model models\fraud_xgboost.joblib --output reports\demo_predictions.csv
```

Training writes the fitted model to `models/` and a machine-readable performance and financial-impact report to `reports/training_report.json`.

## Train with the supplied insurance dataset

The pipeline recognises `insurance_fraud_data.csv` directly. It converts `fraud reported` from `Y`/`N` to the training target, removes the eight unlabelled records, treats `*` as a missing value, and derives month and day from `claim_date`. The claim number is not used as a model feature.

```powershell
python src\train.py --data "C:\Users\sathv\Downloads\insurance_fraud_data.csv"
python src\predict.py --data "C:\Users\sathv\Downloads\insurance_fraud_data.csv" --model models\fraud_xgboost.joblib --output reports\insurance_predictions.csv
```

## Real-data contract

Supply a CSV with one row per claim. Required columns are:

| Column | Meaning |
| --- | --- |
| `is_fraud` | Label: `1` for confirmed fraud, `0` otherwise (required for training) |
| `claim_amount` | Claim amount in AUD |
| `claim_age_days` | Days from policy start to claim |
| `policyholder_age` | Policyholder age |
| `prior_claims` | Number of prior claims |
| `vehicle_age_years` | Vehicle age |
| `annual_premium` | Annual premium in AUD |
| `claim_type` | e.g. collision, theft, windscreen |
| `vehicle_type` | e.g. sedan, SUV, ute |
| `policy_state` | Australian state or territory |
| `incident_hour` | Hour of incident, 0-23 |

Extra columns are permitted; they are ignored. For prediction, omit `is_fraud` if labels are unavailable.

## Important use notes

This tool ranks claims for human investigation; it must not automatically deny, price, or settle claims. Review false-positive impact, data drift, and fairness across relevant groups before operational use. Store only necessary personal information, limit access, and follow the applicable Privacy Act 1988 and Insurance Contracts Act 1984 obligations.
