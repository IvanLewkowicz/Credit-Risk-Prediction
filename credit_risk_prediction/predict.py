import joblib
import pandas as pd
import numpy as np
import shap
from typing import Tuple, Dict, List, Union


class ModelPredictor:
    """
    Production-ready prediction class supporting:
    - CSV input
    - DataFrame input
    - Dict input (API/Streamlit)
    - SHAP explanations
    - Expected Loss calculations
    """

    def __init__(
        self,
        model_path: str,
        explainer_path: str = None,
        default_lgd: float = 0.45
    ):
        self.model_path = model_path
        self.explainer_path = explainer_path
        self.default_lgd = default_lgd
        
        self.model = None
        self.features = None
        self.params = None
        self.metrics = None
        self.explainer = None
        
        self._load_artifact()
        if self.explainer_path:
            self._load_explainer()

    # ----------------------------------------
    # Loaders
    # ----------------------------------------

    def _load_artifact(self):
        """Load model artifact"""
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self.features = artifact["features"]
        self.params = artifact.get("params", {})
        self.metrics = artifact.get("metrics", {})
        print(f"✅ Model loaded with {len(self.features)} features")

    def _load_explainer(self):
        """Load pre-saved SHAP explainer"""
        self.explainer = joblib.load(self.explainer_path)
        print(f"✅ SHAP explainer loaded")

    # ----------------------------------------
    # Input Formatting
    # ----------------------------------------

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct feature order and selection"""
        return df[self.features]

    def _dict_to_df(self, input_dict: dict) -> pd.DataFrame:
        """Convert dict to DataFrame with correct feature order"""
        return pd.DataFrame([{
            feat: input_dict.get(feat, None)
            for feat in self.features
        }])

    # ----------------------------------------
    # Prediction (PD)
    # ----------------------------------------

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict probability of default
        
        Returns:
            predictions: Binary predictions (0/1)
            probabilities: Probability of default (0-1)
        """
        df = self._prepare_df(df)
        preds = self.model.predict(df)
        probs = self.model.predict_proba(df)[:, 1]
        return preds, probs

    def predict_from_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CSV and predict"""
        df = pd.read_csv(csv_path)
        return self.predict(df)

    def predict_from_dict(
        self,
        input_dict: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict from dictionary input"""
        df = self._dict_to_df(input_dict)
        return self.predict(df)

    # ----------------------------------------
    # Expected Loss
    # ----------------------------------------

    def calculate_expected_loss(
        self,
        pd_prob: float,
        lgd: float,
        ead: float
    ) -> Dict[str, float]:
        """
        Calculate Expected Loss
        
        EL = PD × LGD × EAD
        """
        el = pd_prob * lgd * ead
        el_percentage = (el / ead * 100) if ead > 0 else 0.0
        
        return {
            "pd": float(pd_prob),
            "lgd": float(lgd),
            "ead": float(ead),
            "expected_loss": float(el),
            "el_percentage": float(el_percentage)
        }

    def predict_with_el(
        self,
        df: pd.DataFrame,
        lgd: float = None,
        ead: float = None
    ) -> Dict:
        """
        Predict with Expected Loss calculation
        """
        if lgd is None:
            lgd = self.default_lgd
        if ead is None:
            ead = 10000

        preds, probs = self.predict(df)
        
        results = []
        for pred, prob in zip(preds, probs):
            el_calc = self.calculate_expected_loss(prob, lgd, ead)
            el_calc["prediction"] = int(pred)
            results.append(el_calc)
        
        return results

    def predict_from_dict_with_el(
        self,
        input_dict: dict
    ) -> Dict:
        """Predict from dict with EL calculation"""
        df = self._dict_to_df(input_dict)
        lgd = input_dict.pop("LGD", self.default_lgd)
        ead = input_dict.pop("EAD", 10000)
        
        result = self.predict_with_el(df, lgd, ead)
        return result

    # ----------------------------------------
    # SHAP Explainability
    # ----------------------------------------

    def shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values"""
        if self.explainer is None:
            raise ValueError(
                "SHAP explainer not loaded. "
                "Pass explainer_path in constructor."
            )
        df = self._prepare_df(df)
        return self.explainer.shap_values(df)

    def shap_values_from_dict(self, input_dict: dict) -> np.ndarray:
        """Compute SHAP values from dict"""
        df = self._dict_to_df(input_dict)
        return self.shap_values(df)

    # ----------------------------------------
    # Model Info
    # ----------------------------------------

    def get_model_info(self) -> Dict:
        """Get model metadata"""
        return {
            "features": self.features,
            "num_features": len(self.features),
            "params": self.params,
            "metrics": self.metrics
        }


# ----------------------------------------
# Example Usage
# ----------------------------------------

if __name__ == "__main__":
    # Initialize
    predictor = ModelPredictor(
        model_path="../models/xgb_model.joblib",
        explainer_path="../models/xgb_shap_explainer.joblib"
    )

    # # Example 1: Predict from CSV
    # preds, probs = predictor.predict_from_csv("data/test_data.csv")
    # print(f"Predictions: {preds[:5]}")
    # print(f"Probabilities: {probs[:5]}")

    # Example 2: Predict from dict with EL
    customer = {
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 3500,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1,
        "LGD": 0.45,
        "EAD": 10000
    }

    result = predictor.predict_from_dict_with_el(customer)
    print(f"\nPrediction with EL:")
    print(result)

    # # Example 3: SHAP values
    # df = pd.read_csv("data/test_data.csv")
    # shap_vals = predictor.shap_values(df.head(10))
    # print(f"\nSHAP shape: {shap_vals.shape}")
