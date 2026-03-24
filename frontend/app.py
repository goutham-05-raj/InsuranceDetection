import streamlit as st
import json
import os
import pandas as pd
from PIL import Image
import xgboost as xgb

# Streamlit Page Configuration
st.set_page_config(
    page_title="Insurance Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for White Background, Pale Pink Sparkles, and Shades
page_bg_css = """
<style>
.stApp {
    background-color: #ffffff; /* White */
    background-image: 
        radial-gradient(circle at 15% 25%, rgba(255, 182, 193, 0.4) 2px, transparent 3px),
        radial-gradient(circle at 75% 15%, rgba(255, 182, 193, 0.3) 2px, transparent 3px),
        radial-gradient(circle at 45% 75%, rgba(255, 182, 193, 0.5) 3px, transparent 4px),
        radial-gradient(circle at 85% 65%, rgba(255, 182, 193, 0.4) 2px, transparent 3px),
        linear-gradient(45deg, transparent 48%, rgba(255, 182, 193, 0.2) 49%, rgba(255, 182, 193, 0.2) 51%, transparent 52%),
        linear-gradient(-45deg, transparent 48%, rgba(255, 182, 193, 0.15) 49%, rgba(255, 182, 193, 0.15) 51%, transparent 52%);
    background-size: 100px 100px, 150px 150px, 120px 120px, 90px 90px, 80px 80px, 80px 80px;
}
/* Ensure text remains readable over the custom background */
h1, h2, h3, p, label, .stMarkdown {
    color: #1a1a1a !important;
}
.stMetric value {
    color: #000000 !important;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# Load XGBoost model directly for standalone deployment
@st.cache_resource
def load_model():
    paths_to_check = [
        "artifacts/xgb_model.json",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "xgb_model.json")
    ]
    
    for model_path in paths_to_check:
        if os.path.exists(model_path):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            return model, None
    return None, f"Checked paths: {paths_to_check}"

model, error_msg = load_model()

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
    col1.metric("Frontend", "Streamlit 🟢")
    col2.metric("XGBoost Model", "Loaded 🤖" if model else "Not Found ❌")
    col3.metric("Deployment", "Standalone ⚡")
    
    if not model:
        st.error(f"Debug Info: Could not locate model. {error_msg}")

elif page == "Make a Prediction":
    st.title("🔍 Fraud Prediction Input (Live)")
    st.write("Enter the claim details below. The AI will instantly predict fraud probability in real-time as you type!")
    
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
    
    st.markdown("---")
    
    if st.button("Predict Fraud Probability", type="primary"):
        try:
            if model is None:
                st.error(f"Model file not found. Please Reboot the App on Streamlit Cloud to fetch the latest commit. Debug: {error_msg}")
            else:
                # Make prediction natively
                input_df = pd.DataFrame([payload])
                prob = model.predict_proba(input_df)[0][1]
                pred = int(prob >= 0.5)
                
                def map_risk_level(p):
                    if p >= 0.7: return "High"
                    elif p >= 0.4: return "Medium"
                    return "Low"
                
                st.session_state['prediction_results'] = {
                    'prob': prob,
                    'pred': pred,
                    'risk': map_risk_level(prob)
                }
        except Exception as e:
            st.error(f"Error making prediction: {e}")

    if 'prediction_results' in st.session_state:
        res = st.session_state['prediction_results']
        st.subheader("🚨 Prediction Results")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Fraud Probability", f"{float(res['prob']):.2%}")
        metric_col2.metric("Prediction", "Fraudulent 🚫" if res['pred'] == 1 else "Legitimate ✅")
        metric_col3.metric("Risk Level", res['risk'])

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
