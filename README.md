# Credit Risk Probability Model for Alternative Data
---
### Project Objective

- Develop a robust credit risk probability model using alternative eCommerce transaction data to enable safe buy-now-pay-later services for Bati Bank's partner platform.  

---
# Influence of Basel II Accord on Model Interpretability and Documentation

## How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord, established by the Basel Committee on Banking Supervision, introduces a three-pillar framework to enhance banking stability:

- **Pillar 1:** Sets minimum capital requirements based on risk-weighted assets.
- **Pillar 2:** Involves supervisory review processes.
- **Pillar 3:** Promotes market discipline through disclosure.

Its emphasis on risk-sensitive measurement, particularly through the Internal Ratings-Based (IRB) approach, allows banks to use internal models for estimating Probability of Default (PD), but requires these models to be robust, validated, and transparent.

This influences our project by necessitating an interpretable model (e.g., allowing stakeholders to understand how features contribute to risk scores) to facilitate regulatory approval, supervisory audits, and compliance. A well-documented model ensures traceability of decisions, supports Pillar 2 reviews, and aligns with incentives for improved risk management, ultimately reducing capital charges while maintaining financial stability.

## Necessity of Creating a Proxy Variable

Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In the absence of a direct "default" label, which is common in alternative data scenarios like eCommerce behavioral patterns, creating a proxy variable (e.g., derived from Recency, Frequency, Monetary, and Standard Deviation—RFMS—metrics in transaction data) is essential to categorize customers as high-risk (bad) or low-risk (good). 

This proxy serves as a surrogate target for training supervised models, enabling the transformation of observable behaviors into predictive risk signals for creditworthiness assessment. Without it, model development would be infeasible, as there's no ground truth for default outcomes.

However, relying on a proxy introduces business risks, including:
- **Misalignment with True Default Behavior:** Leading to inaccurate risk predictions and potential misclassification of borrowers.
- **Approval of High-Risk Loans:** This can increase default rates and financial losses.
- **Rejection of Low-Risk Loans:** This may reduce revenue and market share.
- **Data Biases:** These can amplify unfair outcomes.
- **Regulatory Non-Compliance:** If the proxy lacks validation.
- **Reputational Damage:** From erroneous decisions.
- **Operational Challenges:** Like model overfitting or poor generalization in real-world scenarios.

## Trade-offs Between Simple and Complex Models

What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

In a regulated financial environment, simple models like Logistic Regression with Weight of Evidence (WoE) offer high interpretability, making it easier to explain predictions (e.g., via coefficients showing feature impacts on default probability), validate against regulatory standards (e.g., Basel II's IRB requirements), and ensure fairness/transparency for audits and consumer disclosures. 

These models are:
- **Computationally Efficient:** Require less data and reduce risks of overfitting.
- **Potential Downsides:** May underperform on complex, non-linear patterns, leading to lower predictive accuracy and potentially higher capital requirements due to conservative risk estimates.

Conversely, complex models like Gradient Boosting provide superior performance by capturing intricate relationships in large datasets, improving AUC metrics and enabling better risk differentiation for profitability. However, they are often black-box models, complicating interpretability. This heightens risks of undetected biases, regulatory scrutiny (e.g., under Pillar 2 for model risk management), and challenges in explaining decisions to stakeholders.

### Summary of Trade-offs

The trade-offs thus center on balancing:
- **Accuracy and Efficiency** vs. **Compliance, Explainability, and Ethical Considerations.**
- Simple models prioritize regulatory alignment and trust, while complex ones favor predictive power but demand additional tools (e.g., SHAP or LIME) for post-hoc interpretability to mitigate opacity.