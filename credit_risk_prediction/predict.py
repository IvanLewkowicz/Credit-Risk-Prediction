import joblib
import pandas as pd
import shap


class ModelPredictor:
    """
    Loads:
        - model artifact
        - SHAP explainer
    Supports:
        - CSV input
        - DataFrame input
        - Dict custom input (Flask/Streamlit)
    """

    def __init__(self, model_path: str, explainer_path: str = None):
        self.model_path = model_path
        self.explainer_path = explainer_path
        self.model = None
        self.features = None
        self.params = None
        self.metrics = None
        self.explainer = None

        self._load_artifact()
        
        if self.explainer_path:
            self._load_explainer()

    # --------------------------------------------
    # Loaders
    # --------------------------------------------
    def _load_artifact(self):
        artifact = joblib.load(self.model_path)

        self.model = artifact["model"]
        self.features = artifact["features"]
        self.params = artifact.get("params", {})
        self.metrics = artifact.get("metrics", {})

    def _load_explainer(self):
        """Load pre-saved SHAP TreeExplainer."""
        self.explainer = joblib.load(self.explainer_path)

    # --------------------------------------------
    # Input formatting
    # --------------------------------------------
    def _prepare_df(self, df: pd.DataFrame):
        return df[self.features]

    def _dict_to_df(self, input_dict: dict):
        return pd.DataFrame([{feat: input_dict.get(feat, None)
                              for feat in self.features}])

    # --------------------------------------------
    # Prediction
    # --------------------------------------------
    def predict(self, df: pd.DataFrame):
        df = self._prepare_df(df)
        preds = self.model.predict(df)
        probs = self.model.predict_proba(df)[:, 1]
        return preds, probs

    def predict_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        return self.predict(df)

    def predict_from_dict(self, input_dict: dict):
        df = self._dict_to_df(input_dict)
        return self.predict(df)

    # --------------------------------------------
    # SHAP explainability (loaded explainer)
    # --------------------------------------------
    def shap_values(self, df: pd.DataFrame):
        """Compute shap values using pre-loaded explainer."""
        if self.explainer is None:
            raise ValueError("SHAP explainer not loaded. Pass explainer_path in constructor.")

        df = self._prepare_df(df)
        return self.explainer.shap_values(df)

    def shap_values_from_dict(self, input_dict: dict):
        df = self._dict_to_df(input_dict)
        return self.shap_values(df)


# --------------------------------------------
# Example
# --------------------------------------------
if __name__ == "__main__":
    predictor = ModelPredictor(
        model_path="model_artifact.joblib",
        explainer_path="explainer.joblib"
    )

    df = pd.read_csv("new_data.csv")

    preds, probs = predictor.predict(df)
    print(preds[:5], probs[:5])

    shap_vals = predictor.shap_values(df.head(10))
    print("Shape SHAP values:", len(shap_vals))
