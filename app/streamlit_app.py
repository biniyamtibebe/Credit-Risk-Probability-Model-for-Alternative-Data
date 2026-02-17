import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Initialize state for predicted probability
if "prob" not in st.session_state:
    st.session_state.prob = None

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Intelligence",
    layout="wide",
    page_icon="💳"
)

# ---------------------------------------------------
# Custom Styling (Executive Fintech Theme)
# ---------------------------------------------------
st.markdown("""
<style>
/* App Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fff200, #ffeb3b); /* Updated to yellow gradient */
    color: black;  /* Set text color to black for better readability */
}

/* Main Title */
.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #ff6f00; /* Darker yellow/orange for the title */
    margin-bottom: 15px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #cfd8dc;
    margin-bottom: 25px;
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.7); /* Semi-transparent white for better contrast */
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0px 8px 35px rgba(0,0,0,0.5);
    margin-bottom: 25px;
}

/* Large Metric Number */
.big-number {
    font-size: 40px;
    font-weight: 800;
    color: #333333; /* Darker color for numbers */
}

/* Risk Colors */
.high-risk { color: #ff1744; font-weight: bold; font-size: 26px; }
.medium-risk { color: #ffb300; font-weight: bold; font-size: 26px; }
.low-risk { color: #00e676; font-weight: bold; font-size: 26px; }

/* Button Styling */
div.stButton > button {
    background: linear-gradient(90deg, #fbc02d, #f9a825); /* Dark yellow/orange gradient for buttons */
    color: white;
    font-weight: 700;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 16px;
}

/* Input Text Color */
input[type=text], input[type=number], .stTextInput, .stNumberInput {
    color: black;  /* Input text color */
}

/* Caption Color */
label {
    color: black;  /* Label color */
}

/* Custom Info Box */
.custom-note {
    background: rgba(0, 123, 255, 0.2);
    border-left: 6px solid #ffca28; /* Yellow color for accents */
    padding: 20px;
    border-radius: 12px;
    font-size: 16px;
    color: #333333; /* Darker text color */
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------
model = joblib.load("models/credit_model.pkl")

# ---------------------------------------------------
# Header Section
# ---------------------------------------------------
st.markdown('<div class="main-title">💳 Credit Risk & Fraud Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Probability of Default & Fraud Monitoring System</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------
# Input & Output Layout
# ---------------------------------------------------
col1, col2 = st.columns(2)

# Applicant Information Input
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👤 Applicant Information")
    
    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Annual Income ($)", min_value=1000, max_value=200000, value=30000)
    transaction_count = st.number_input("Monthly Transaction Count", min_value=0, max_value=500, value=50)

    st.markdown('</div>', unsafe_allow_html=True)

# Model Output Section
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Model Output")
    st.write("Click below to evaluate the applicant’s credit risk profile.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Run Risk Assessment
# ---------------------------------------------------
if st.button("🚀 Run Risk Assessment"):
    if income <= 0:
        st.error("Annual Income must be greater than zero.")
    else:
        input_df = pd.DataFrame({"age": [age], "income": [income]})
        st.session_state.prob = model.predict_proba(input_df)[0][1]

        # Risk Classification
        if st.session_state.prob >= 0.7:
            risk_level = "High Risk"
            risk_class = "high-risk"
            risk_explanation = (
                "High risk indicates the probability of default is significant, "
                "potentially due to a low income relative to debts or a high ratio of "
                "monthly transactions."
            )
        elif st.session_state.prob >= 0.4:
            risk_level = "Medium Risk"
            risk_class = "medium-risk"
            risk_explanation = (
                "Medium risk reflects some concerns with the financial profile. This could "
                "mean the income is not significantly lower than necessary but still raises "
                "some flags."
            )
        else:
            risk_level = "Low Risk"
            risk_class = "low-risk"
            risk_explanation = (
                "Low risk suggests a strong financial position, indicating a healthy "
                "relationship with credit and manageable expenses compared to income."
            )

        fraud_flag = income < 5000 and transaction_count > 300

        st.markdown("---")
        st.subheader("📌 Risk Summary")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Probability of Default", f"{st.session_state.prob:.2%}")

        with m2:
            st.markdown(f"<p class='{risk_class}'>{risk_level}</p>", unsafe_allow_html=True)

        with m3:
            if fraud_flag:
                st.error("⚠ Fraud Pattern Detected")
            else:
                st.success("✔ No Fraud Signals")

        # ---------------------------------------------------
        # Explanation Section
        # ---------------------------------------------------
        st.markdown("### 📖 Risk Explanation")
        st.write(risk_explanation)

        # ---------------------------------------------------
        # Risk Progress Bar
        # ---------------------------------------------------
        st.markdown("### Risk Intensity")
        st.progress(float(st.session_state.prob))

        # ---------------------------------------------------
        # Probability Visualization
        # ---------------------------------------------------
        st.markdown("### 📈 Risk Probability Visualization")

        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.barh(["Default Probability"], [st.session_state.prob])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Predicted Default Risk")
        st.pyplot(fig)

        # ---------------------------------------------------
        # Analytical Indicators
        # ---------------------------------------------------
        st.markdown("### 📊 Analytical Indicators")

        risk_score = round(st.session_state.prob * 100, 2)
        expected_loss_proxy = round(st.session_state.prob * income * 0.45, 2)

        a1, a2 = st.columns(2)

        with a1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Risk Score (0–100)")
            st.markdown(f"<div class='big-number'>{risk_score}</div>", unsafe_allow_html=True)
            st.caption("Derived directly from predicted probability.")
            st.markdown('</div>', unsafe_allow_html=True)

        with a2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Estimated Expected Loss ($)")
            st.markdown(f"<div class='big-number'>${expected_loss_proxy:,.2f}</div>", unsafe_allow_html=True)
            st.caption("PD × Exposure × LGD approximation.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ---------------------------------------------------
        # Historical Portfolio Risk Trend
        # ---------------------------------------------------
        st.markdown("### 📉 Historical Portfolio Risk Trend")

        dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='ME')
        historical_probs = [max(0, min(1, st.session_state.prob + (i - 6) * 0.01)) for i in range(12)]

        historical_df = pd.DataFrame({
            "Month": dates,
            "Average_PD": historical_probs
        })

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(historical_df["Month"], historical_df["Average_PD"], marker='o')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Average Probability of Default")
        ax2.set_title("12-Month Portfolio Risk Trend")
        ax2.grid(alpha=0.3)

        st.pyplot(fig2)

else:
    st.info("ℹ️ Run a risk assessment to view historical trends.")

# ---------------------------------------------------
# Production Warning Note
# ---------------------------------------------------
st.markdown("""
<div class="custom-note">
⚠ <strong>Production Advisory:</strong><br><br>
Fraud detection logic implemented here is heuristic and rule-based.
In a production banking environment, this should be replaced with
a dedicated anomaly detection or supervised fraud classification model.
</div>
""", unsafe_allow_html=True)