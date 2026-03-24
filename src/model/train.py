import xgboost as xgb
import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.data.preprocess import load_and_preprocess, apply_smote
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    logger.info("Starting model training pipeline...")
    
    # 1. Load data
    raw_claims_file = f"{config['data']['raw_path']}/claims.json"
    if not os.path.exists(raw_claims_file):
        logger.error(f"Raw data file not found at {raw_claims_file}. Run simulator.py first.")
        return
        
    df = load_and_preprocess(raw_claims_file)
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["model"]["test_size"], 
        random_state=config["model"]["random_state"],
        stratify=y
    )
    
    # 3. Apply SMOTE only on training data
    logger.info("Balancing training data with SMOTE...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # 4. Train XGBoost
    logger.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        learning_rate=config["model"]["learning_rate"],
        random_state=config["model"]["random_state"],
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train_res, y_train_res)
    logger.info("Model training completed.")
    
    # 5. Evaluate
    logger.info("Evaluating on test set...")
    preds = model.predict(X_test)
    logger.info(f"Classification Report:\n{classification_report(y_test, preds)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")
    
    # 6. Save model
    os.makedirs(os.path.dirname(config["data"]["model_path"]), exist_ok=True)
    model_path = config["data"]["model_path"]
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
