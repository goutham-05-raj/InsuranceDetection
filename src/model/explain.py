import shap
import xgboost as xgb
import yaml
import os
import matplotlib.pyplot as plt
from src.data.preprocess import load_and_preprocess
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_shap_explanations(claim_idx=0):
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model_path = config["data"]["model_path"]
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Run train.py first.")
        return
        
    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    logger.info(f"Loaded XGBoost model from {model_path}")
    
    # Load some data to explain
    raw_claims_file = f"{config['data']['raw_path']}/claims.json"
    df = load_and_preprocess(raw_claims_file)
    X = df.drop(columns=["target"])
    
    logger.info("Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Ensure images directory exists
    os.makedirs("frontend/static/images", exist_ok=True)
    
    # Save summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    sum_plot_path = "frontend/static/images/shap_summary.png"
    plt.savefig(sum_plot_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP summary plot to {sum_plot_path}")
    
    # Save waterfall plot for the specific claim
    # Note: Waterfall expects an Explanation object
    plt.figure()
    explanation = shap.Explanation(values=shap_values[claim_idx], 
                                   base_values=explainer.expected_value, 
                                   data=X.iloc[claim_idx], 
                                   feature_names=X.columns)
    shap.waterfall_plot(explanation, show=False)
    waterfall_path = "frontend/static/images/shap_waterfall.png"
    plt.savefig(waterfall_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP waterfall plot to {waterfall_path}")

if __name__ == "__main__":
    generate_shap_explanations()
