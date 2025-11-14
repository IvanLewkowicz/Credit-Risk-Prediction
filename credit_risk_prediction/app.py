from fastapi import FastAPI, HTTPException, Request
from typing import Dict, Any
import json
import numpy as np

# Assuming you have a ModelPredictor class loaded with model and explainer
predictor = ModelPredictor(
    model_path="models/xgb_model.joblib",
    explainer_path="models/xgb_shap_explainer.joblib",
    default_lgd=0.45
)

app = FastAPI()

def get_risk_level(prob: float) -> str:
    if prob < 0.33:
        return "LOW"
    elif prob < 0.67:
        return "MEDIUM"
    else:
        return "HIGH"

@app.post("/predict")
async def predict(request: Request):
    try:
        input_dict = await request.json()
        # input_dict should have keys like "NumberOfTime30-59DaysPastDueNotWorse" with hyphens intact
        preds, probs = predictor.predict_from_dict(input_dict)
        pred = int(preds[0])
        prob = float(probs[0])
        risk = get_risk_level(prob)
        return {"prediction": pred, "probability": prob, "risk_level": risk}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/expected_loss")
async def expected_loss(request: Request):
    try:
        input_dict = await request.json()
        # Extract LGD and EAD manually, with defaults if missing
        lgd = input_dict.pop('LGD', 0.45)
        ead = input_dict.pop('EAD', 10000)
        preds, probs = predictor.predict_from_dict(input_dict)
        pred = int(preds[0])
        prob = float(probs[0])
        el = prob * lgd * ead
        el_percentage = (el / ead) * 100 if ead > 0 else 0
        risk = get_risk_level(prob)
        return {
            "prediction": pred,
            "probability": prob,
            "risk_level": risk,
            "expected_loss": el,
            "el_percentage": el_percentage,
            "lgd": lgd,
            "ead": ead
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/shap")
async def shap_explanation(request: Request):
    try:
        input_dict = await request.json()
        preds, probs = predictor.predict_from_dict(input_dict)
        pred = int(preds[0])
        prob = float(probs[0])
        shap_vals = predictor.shap_values_from_dict(input_dict)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim > 1:
            shap_vals = shap_vals[0]
        features = predictor.features
        return {
            "prediction": pred,
            "probability": prob,
            "shap_values": shap_vals.tolist(),
            "features": features
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
