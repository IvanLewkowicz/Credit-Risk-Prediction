from fastapi import FastAPI, HTTPException, Request
from typing import Dict, Any
import numpy as np
from credit_risk_prediction.predict import ModelPredictor

# Initialize predictor
predictor = ModelPredictor(
    model_path="../models/xgb_model.joblib",
    explainer_path="../models/xgb_shap_explainer.joblib",
    default_lgd=0.45
)

app = FastAPI()

# --------------------------------------------
# Risk Bucket Logic
# --------------------------------------------
def get_risk_level(prob: float) -> str:
    if prob < 0.33:
        return "LOW"
    elif prob < 0.67:
        return "MEDIUM"
    else:
        return "HIGH"

# --------------------------------------------
# PD Prediction Endpoint
# --------------------------------------------
@app.post("/predict")
async def predict(request: Request):
    try:
        input_dict = await request.json()

        preds, probs = predictor.predict_from_dict(input_dict)

        pred = int(preds[0])
        prob = float(probs[0])
        risk = get_risk_level(prob)

        return {
            "prediction": pred,
            "probability": prob,
            "risk_level": risk
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --------------------------------------------
# Expected Loss Endpoint
# --------------------------------------------
@app.post("/expected_loss")
async def expected_loss(request: Request):
    try:
        input_dict = await request.json()

        # This method returns a *list of dicts*, one per row
        result = predictor.predict_from_dict_with_el(input_dict)

        # Your input is 1 observation → extract index 0
        output = result[0]

        # Add risk bucket
        risk = get_risk_level(output["pd"])

        return {
            "prediction": output["prediction"],
            "probability": output["pd"],
            "risk_level": risk,
            "expected_loss": output["expected_loss"],
            "el_percentage": output["el_percentage"],
            "lgd": output["lgd"],
            "ead": output["ead"]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --------------------------------------------
# SHAP Explainability Endpoint
# --------------------------------------------
@app.post("/shap")
async def shap_explanation(request: Request):
    try:
        input_dict = await request.json()

        preds, probs = predictor.predict_from_dict(input_dict)
        pred = int(preds[0])
        prob = float(probs[0])

        # SHAP Values (ensures consistent shape)
        shap_vals = predictor.shap_values_from_dict(input_dict)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim > 1:
            shap_vals = shap_vals[0]

        return {
            "prediction": pred,
            "probability": prob,
            "features": predictor.features,
            "shap_values": shap_vals.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
