import streamlit as st
import joblib
import pandas as pd

st.title("Credit Risk Probability Model")

model = joblib.load("models/credit_model.pkl")

age = st.slider("Age", 18, 70, 30)
income = st.number_input("Annual Income", 1000, 100000, 30000)

input_df = pd.DataFrame({
    "age": [age],
    "income": [income]
})

if st.button("Predict Risk"):
    prob = model.predict_proba(input_df)[0][1]
    st.write(f"Probability of Default: {prob:.2%}")

    if prob > 0.5:
        st.error("High Risk")
    else:
        st.success("Low Risk")
