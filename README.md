
# Credit Risk Probability Model Using Alternative Data

A production-ready machine learning system that predicts the probability of loan default using alternative financial and behavioral data. Designed to support responsible lending decisions, this system aims to reduce credit risk exposure and improve financial inclusion.

## Business Problem

Financial institutions must assess the likelihood of borrower default before approving loans. Traditional credit scoring methods often disadvantage individuals with limited formal credit history, especially in emerging markets.

This project builds a robust and explainable credit risk model that estimates the probability of default using alternative data, enabling lenders to reduce portfolio risk while safely expanding access to credit.

## Solution Overview

This project implements a modular, production-ready machine learning pipeline that includes:

- **Structured Data Preprocessing**
- **Feature Engineering**
- **Model Training and Evaluation**
- **Probability-based Risk Scoring**
- **SHAP-based Explainability**
- **Automated Testing and CI/CD Validation**
- **Interactive Dashboard for Stakeholder Exploration**

The system outputs a probability of default (PD) and classifies applicants into risk tiers to support credit decision-making.

## Key Results

- **ROC-AUC**: ≥ 0.80 (model discrimination strength)
- **Automated CI Pipeline** with 100% test pass rate
- **Reproducible Environment** with dependency control
- **Explainable Predictions** using SHAP
- **Interactive Dashboard** for business users

## Project Structure

```plaintext
credit-risk-model/
│
├── app/                  # Streamlit dashboard
├── src/                  # Core ML pipeline modules
├── models/               # Saved trained model
├── tests/                # Unit & integration tests
├── .github/workflows/    # CI/CD configuration
├── requirements.txt
├── main.py
└── README.md
```

## Quick Start

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/credit-risk-model.git
   cd credit-risk-model
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   ```bash
   python main.py
   ```

4. **Run Tests**:
   ```bash
   pytest
   ```

5. **Launch the Dashboard**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Technical Architecture

1. **Data Pipeline**:
   - Missing value imputation
   - Standard scaling for numerical features
   - One-hot encoding for categorical variables
   - Reusable sklearn `ColumnTransformer`

2. **Model**:
   - **Random Forest Classifier** (baseline upgradeable to XGBoost)
   - Stratified train-test split
   - Probability-based output for risk scoring

3. **Evaluation Metrics**:
   - **ROC-AUC** (primary discrimination metric)
   - Precision, Recall, F1-score
   - Confusion Matrix

4. **Explainability**:
   - **SHAP (SHapley Additive Explanations)** is used to:
     - Identify globally important risk drivers
     - Explain individual loan decisions
     - Detect potentially biased or unstable patterns

This approach supports transparency requirements in regulated financial environments.

## Risk Scoring Framework

The model outputs a probability of default (PD). Applicants are categorized as follows:

- **Low Risk**: PD < 0.30
- **Medium Risk**: 0.30 ≤ PD < 0.60
- **High Risk**: PD ≥ 0.60

These thresholds can be adjusted based on institutional risk appetite.

## Testing & Reliability

To ensure production readiness:

- Unit tests validate preprocessing and training logic
- Integration tests validate the end-to-end pipeline
- GitHub Actions automatically runs tests on push
- No hard-coded parameters; configuration is managed via `dataclass`
- Fully reproducible environment via `requirements.txt`

This ensures reliability and reduces operational risk.

## Reproducibility

- All dependencies are listed in `requirements.txt`.
- Model training is deterministic via fixed random seeds.
- Saved model artifacts are stored in the `models/` directory.

Any user can clone the repository to reproduce results.

## Dashboard (Business Interface)

The Streamlit dashboard allows users to:

- Simulate a loan applicant profile
- View predicted probability of default
- See risk tier classification
- Understand the model's predictions (via SHAP)

This bridges the gap between data science and underwriting teams.

## Business Impact

This solution helps financial institutions:

- Reduce default rates through improved risk discrimination
- Expand credit access using alternative data
- Improve transparency in underwriting decisions
- Support regulatory compliance through explainability
- Standardize and automate risk assessment processes

---

