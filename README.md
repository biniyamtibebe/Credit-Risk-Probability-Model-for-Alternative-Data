# Credit Risk Probability Model for Alternative Data
---
### Project Objective

- Develop a robust credit risk probability model using alternative eCommerce transaction data to enable safe buy-now-pay-later services for Bati Bank's partner platform.  

_____

# Credit Risk Probability Model for Alternative Data

---
## Project Overview
This project builds an end-to-end **credit risk scoring system** for Bati Bank to support a **Buy-Now-Pay-Later (BNPL)** product using alternative eCommerce transaction data.  
The objective is to transform customer behavioral data into a **probability of default**, a **credit score**, and **loan recommendations** in a compliant and reproducible way.

---

## 🎯 Task 1 — Credit Scoring Business Understanding

### 1. Basel II & Model Interpretability
The **Basel II Capital Accord** requires financial institutions to:
- Quantify credit risk accurately
- Maintain **transparent, auditable, and well-documented models**
- Hold sufficient capital based on measured risk

Because of this, our credit scoring model must be:
- **Interpretable** (clear drivers of risk)
- **Reproducible** (same inputs → same outputs)
- **Well-documented** (for audit and regulatory review)

This strongly influences model choice, feature design, and reporting.

---

### 2. Why a Proxy Default Variable Is Required
The dataset does **not include an explicit default label**.  
To proceed, we define a **proxy default variable** using customer behavior patterns such as:
- Transaction **Recency**
- Transaction **Frequency**
- Transaction **Monetary value** (RFM)

Customers with risky behavioral patterns are labeled as **high risk**, while stable customers are labeled **low risk**.

⚠️ **Business Risk**:  
Since this is a proxy and not a true default label:
- Some customers may be misclassified
- Conservative thresholds and continuous monitoring are required
- Model outputs should support decisions, not replace human judgment

---

### 3. Simple vs Complex Models — Key Trade-offs

| Aspect | Interpretable Models (Logistic Regression, WoE) | Complex Models (XGBoost, Random Forest) |
|------|-----------------------------------------------|----------------------------------------|
| Interpretability | High | Medium–Low |
| Regulatory Acceptance | Strong | Moderate |
| Predictive Power | Moderate | High |
| Explainability | Clear coefficients | Requires SHAP/LIME |
| Production Risk | Low | Higher |

**Project Approach**:
- Start with interpretable models for baseline compliance
- Use advanced models for performance comparison
- Apply **SHAP** to explain complex model behavior

---

## ✅ Outcome of Task 1
- Clear business understanding aligned with Basel II
- Justified use of proxy default variable
- Defined modeling philosophy balancing risk, performance, and regulation

---
## 📂 Repository Structure
        credit-risk-model/

             ├── .github/workflows/ci.yml   # For CI/CD
             ├── data/                       # add this folder to .gitignore
             │   ├── raw/                   # Raw data goes here 
             │   └── processed/             # Processed data for training   
             ├── notebooks/
             │   └── eda.ipynb          # Exploratory, one-off analysis
             ├── src/
             │   ├── __init__.py
             │   ├── data_processing.py     # Script for feature engineering
             │   ├── train.py               # Script for model training
             │   ├── predict.py             # Script for inference
             │   └── api/
             │       ├── main.py            # FastAPI application
             │       └── pydantic_models.py # Pydantic models for API
             ├── tests/
             │   └── test_data_processing.py # Unit tests
             ├── Dockerfile
             ├── docker-compose.yml
             ├── requirements.txt
             ├── .gitignore
             └── README.md

____
# Task 2 – Exploratory Data Analysis (EDA)

## Objective
The goal of this task is to explore the dataset to uncover patterns, identify data quality issues, and form hypotheses that will guide your feature engineering for modeling.

All exploratory work is performed in the Jupyter Notebook:  
`notebooks/eda.ipynb`  

> **Note:** This notebook is for exploration only and is not intended for production code.

---

## Instructions

### 1. Overview of the Data
- Understand the structure of the dataset:
  - Number of rows and columns
  - Data types of each feature
  - Sample records to inspect values

### 2. Summary Statistics
- Compute descriptive statistics for numerical features:
  - Mean, median, standard deviation
  - Minimum, maximum, quartiles
- Identify general trends and distributions in the data

### 3. Distribution of Numerical Features
- Visualize numerical features using:
  - Histograms
  - Density plots
- Identify patterns, skewness, and potential outliers

### 4. Distribution of Categorical Features
- Analyze categorical features using:
  - Bar plots
  - Frequency tables
- Gain insights into category distribution and variability

### 5. Correlation Analysis
- Examine relationships between numerical features using:
  - Correlation matrices
  - Heatmaps
- Identify strongly correlated variables

### 6. Identifying Missing Values
- Detect missing or null values in the dataset
- Decide on appropriate strategies for handling missing data:
  - Imputation (mean, median, mode)
  - Removal or domain-specific handling

### 7. Outlier Detection
- Use box plots and scatter plots to identify outliers
- Determine if outliers require transformation, capping, or removal

---









