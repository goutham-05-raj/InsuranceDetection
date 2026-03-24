from fastapi import FastAPI, HTTPException
import xgboost as xgb
import pandas as pd
import yaml
import os
from src.api.schemas import ClaimFeatures, PredictionResponse, BatchPredictionResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Insurance Fraud Detection API",
    description="REST API to serve the XGBoost fraud detection model",
    version="1.0.0"
)

# Global variables for model
model = None

@app.on_event("startup")
def load_artifacts():
    global model
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        model_path = config["data"]["model_path"]
        if os.path.exists(model_path):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            logger.info(f"Successfully loaded XGBoost model from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Please train the model first.")
            model = None
    except Exception as e:
        logger.error(f"Error loading artifacts during startup: {e}")

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "degraded", "message": "Model not loaded. API is up."}
    return {"status": "ok", "message": "API and model are ready."}

def map_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"

@app.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        # Convert schema to dataframe
        input_data = pd.DataFrame([claim.dict()])
        
        # Make predict probability
        prob = model.predict_proba(input_data)[0][1]
        pred = int(prob >= 0.5)
        
        response = PredictionResponse(
            fraud_probability=float(prob),
            fraud_prediction=pred,
            risk_level=map_risk_level(prob)
        )
        logger.info(f"Made prediction: {response}")
        return response
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(claims: list[ClaimFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        input_data = pd.DataFrame([c.dict() for c in claims])
        probs = model.predict_proba(input_data)[:, 1]
        
        predictions = []
        for p in probs:
            predictions.append(
                PredictionResponse(
                    fraud_probability=float(p),
                    fraud_prediction=int(p >= 0.5),
                    risk_level=map_risk_level(p)
                )
            )
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])
