import streamlit as st
import requests
import json
import os
import pandas as pd
from PIL import Image

# Streamlit Page Configuration
st.set_page_config(
    page_title="Insurance Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL (Read from environment for Docker, fallback for local dev)
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview Dashboard", "Make a Prediction", "Model Explainability (SHAP)"])

if page == "Overview Dashboard":
    st.title("🛡️ Insurance Fraud Detection System")
    st.markdown("""
        Welcome to the **Automated Insurance Fraud Detection System** dashboard.
        This system leverages an XGBoost machine learning model to predict the likelihood of an insurance claim being fraudulent.
    """)
    
    st.info("👈 Use the sidebar to navigate to other pages: \n- **Make a Prediction**: Test individual claims against the model.\n- **Model Explainability (SHAP)**: Understand how the model arrives at its decisions.")

    st.subheader("System Infrastructure")
    col1, col2, col3 = st.columns(3)
    col1.metric("FastAPI Backend", "Online 🟢")
    col2.metric("XGBoost Model", "Loaded 🤖")
    col3.metric("Data Simulation", "Ready 🎲")

elif page == "Make a Prediction":
    st.title("🔍 Fraud Prediction Input")
    st.write("Enter the claim details below to predict fraud probability.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Claim Amounts")
            total_claimed = st.number_input("Total Claimed ($)", value=50000.0)
            total_approved = st.number_input("Total Approved ($)", value=20000.0)
            
        with col2:
            st.subheader("Claimant Financials")
            credit_score = st.number_input("Credit Score", value=650, min_value=300, max_value=850)
            annual_income = st.number_input("Annual Income ($)", value=70000.0)
            dti_ratio = st.number_input("Debt to Income Ratio", value=0.4)
            
        with col3:
            st.subheader("Claimant Behavior")
            claim_freq = st.number_input("Claim Frequency (past year)", value=2, min_value=0)
            late_payments = st.number_input("Late Payments", value=1, min_value=0)
            policy_changes = st.number_input("Policy Changes", value=0, min_value=0)
            
        st.subheader("Coverage Breakdown")
        col4, col5, col6, col7, col8 = st.columns(5)
        bil = col4.number_input("BIL Claimed", value=10000.0)
        pdl = col5.number_input("PDL Claimed", value=15000.0)
        pip = col6.number_input("PIP Claimed", value=5000.0)
        colli = col7.number_input("Collision", value=10000.0)
        comp = col8.number_input("Comprehensive", value=10000.0)
        
        submitted = st.form_submit_button("Predict Fraud Probability")
        
        if submitted:
            # Derived features
            claimed_income_ratio = total_claimed / max(annual_income, 1)
            approved_claimed_ratio = total_approved / max(total_claimed, 1)
            
            payload = {
                "TotalClaimed": total_claimed,
                "TotalApproved": total_approved,
                "CreditScore": credit_score,
                "AnnualIncome": annual_income,
                "DebtToIncomeRatio": dti_ratio,
                "ClaimFrequency": claim_freq,
                "LatePayments": late_payments,
                "PolicyChanges": policy_changes,
                "CoverageBIL": bil,
                "CoveragePDL": pdl,
                "CoveragePIP": pip,
                "CoverageCollision": colli,
                "CoverageComprehensive": comp,
                "ClaimedToIncomeRatio": claimed_income_ratio,
                "ApprovedToClaimedRatio": approved_claimed_ratio
            }
            
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction Successful!")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
                    metric_col2.metric("Prediction", "Fraudulent 🚫" if result['fraud_prediction'] == 1 else "Legitimate ✅")
                    metric_col3.metric("Risk Level", result['risk_level'])
                else:
                    st.error(f"Error from API: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the prediction API. Ensure the FastAPI backend is running.")

elif page == "Model Explainability (SHAP)":
    st.title("📈 Model Explainability (XAI)")
    st.write("Understand how the XGBoost model makes decisions using SHAP (SHapley Additive exPlanations).")
    
    st.subheader("Global Feature Importance (Summary Plot)")
    summary_plot_path = "frontend/static/images/shap_summary.png"
    if os.path.exists(summary_plot_path):
        image = Image.open(summary_plot_path)
        st.image(image, caption="SHAP Summary Plot", use_column_width=True)
        st.write("This plot shows the most important features driving the model's predictions globally across the dataset. Red indicates high feature value, blue indicates low feature value.")
    else:
        st.warning("SHAP summary plot not found. Run `src/model/explain.py` to generate it.")
        
    st.subheader("Local Interpretability (Waterfall Plot for Sample Claim)")
    waterfall_plot_path = "frontend/static/images/shap_waterfall.png"
    if os.path.exists(waterfall_plot_path):
        image = Image.open(waterfall_plot_path)
        st.image(image, caption="SHAP Waterfall Plot", use_column_width=True)
        st.write("This plot shows how each feature pushed the model's output from the base value to the final predicted value for a single specific claim.")
    else:
        st.warning("SHAP waterfall plot not found. Run `src/model/explain.py` to generate it.")
