from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import shap
import xgboost as xgb
from typing import Dict, Any, List


# -----------------------------------------------
# LOAD YOUR TRAINED MODEL & EXPLAINER
# -----------------------------------------------
# Adjust paths to where your model artifacts are stored
MODEL_PATH = "models/credit_risk_model.pkl"
EXPLAINER_PATH = "models/shap_explainer.pkl"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)  # Your XGBoost or other model

with open(EXPLAINER_PATH, "rb") as f:
    explainer = pickle.load(f)  # Your SHAP TreeExplainer or KernelExplainer


# -----------------------------------------------
# DEFINE PYDANTIC MODELS
# -----------------------------------------------
class PredictionInput(BaseModel):
    __root__: Dict[str, float]

    class Config:
        schema_extra = {
            "example": {
                "__root__": {
                    "RevolvingUtilizationOfUnsecuredLines": 0.3,
                    "age": 40,
                    "NumberOfTime30-59DaysPastDueNotWorse": 0,
                    "DebtRatio": 0.3,
                    "MonthlyIncome": 3000,
                    "NumberOfOpenCreditLinesAndLoans": 8,
                    "NumberOfTimes90DaysLate": 0,
                    "NumberRealEstateLoansOrLines": 1,
                    "NumberOfTime60-89DaysPastDueNotWorse": 0,
                    "NumberOfDependents": 0
                }
            }
        }


class PredictionOutput(BaseModel):
    prediction: int  # 0 or 1 (default or not)
    probability: float  # Probability of default


class SHAPOutput(BaseModel):
    shap_values: List[float]
    expected_value: float
    features: List[str]


# -----------------------------------------------
# INITIALIZE FASTAPI
# -----------------------------------------------
app = FastAPI(
    title="Credit Risk API",
    description="API for credit risk prediction with SHAP explainability",
    version="1.0"
)


# -----------------------------------------------
# ENDPOINTS
# -----------------------------------------------
@app.get("/health", tags=["health"])
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "API is running"}


@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Predict probability of default for a customer.

    Returns:
        - prediction: Binary prediction (0 = approved, 1 = default)
        - probability: Probability of default (between 0 and 1)
    """
    data_dict = input_data.__root__

    # Order features according to model
    feature_names = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents"
    ]

    X = np.array([[data_dict.get(feat, 0) for feat in feature_names]])

    # Get prediction
    if hasattr(model, "predict_proba"):
        # For classifiers with predict_proba (e.g., XGBoost classification)
        pred_proba = model.predict_proba(X)[0]
        probability = float(pred_proba[1])  # Probability of class 1 (default)
    else:
        # For other models
        probability = float(model.predict(X)[0])

    prediction = 1 if probability > 0.5 else 0

    return PredictionOutput(prediction=prediction, probability=probability)


@app.post("/shap", response_model=SHAPOutput, tags=["explainability"])
def get_shap_values(input_data: PredictionInput) -> SHAPOutput:
    """
    Get SHAP values for model explainability.

    Returns:
        - shap_values: SHAP values for each feature
        - expected_value: Base value (expected model output)
        - features: List of feature names
    """
    data_dict = input_data.__root__

    feature_names = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents"
    ]

    X = np.array([[data_dict.get(feat, 0) for feat in feature_names]])

    # Calculate SHAP values
    shap_vals = explainer.shap_values(X)

    # Handle SHAP output format (for multi-class, take class 1)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim > 1:
        shap_vals = shap_vals[0]

    expected_val = explainer.expected_value
    if isinstance(expected_val, list):
        expected_val = expected_val[1]

    return SHAPOutput(
        shap_values=shap_vals.tolist(),
        expected_value=float(expected_val),
        features=feature_names
    )


# -----------------------------------------------
# ROOT ENDPOINT
# -----------------------------------------------
@app.get("/", tags=["info"])
def root():
    """Root endpoint with API information"""
    return {
        "name": "Credit Risk Assessment API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "shap": "/shap (POST)",
            "docs": "/docs"
        }
    }


# -----------------------------------------------
# RUN THE API
# -----------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )