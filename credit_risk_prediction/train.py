import joblib
import pandas as pd
import numpy as np
import shap
from scipy.stats import uniform, randint

from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


class ModelTrainer:

    def __init__(
        self,
        data_path: str,
        drop_columns: str,
        target: str,
        model_params: dict,
        model_path: str,
        explainer_path: str,
        model_name: str
    ):
        self.data_path = data_path
        self.drop_columns = drop_columns
        self.target = target
        self.model_params = model_params
        self.model_path = model_path
        self.explainer_path = explainer_path
        self.model_name = model_name

        # internal variables
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_params = None
        self.metrics = None


    # ----------------------------------------------------------
    # LOAD + PREPARE DATA
    # ----------------------------------------------------------
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.X = self.df.drop(columns=self.drop_columns)
        self.y = self.df[self.target]
        print(f"Loaded dataset: {self.df.shape} rows")


    def prepare_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            stratify=self.y,
            random_state=42
        )
        print("Data prepared:")
        print(f" - Train shape: {self.X_train.shape}")
        print(f" - Test shape: {self.X_test.shape}")
        print(f" - Y Train shape: {self.y_train.shape}")
        print(f" - Y Test shape: {self.y_test.shape}")


    # ----------------------------------------------------------
    # UTILS
    # ----------------------------------------------------------
    def get_pos_weight(self):
        y_arr = np.asarray(self.y_train).astype(int).reshape(-1)
        neg, pos = np.bincount(y_arr)
        w = neg / pos
        print(f"Class ratio {neg}:{pos}, scale_pos_weight = {w:.2f}")
        return w


    # ----------------------------------------------------------
    # TRAIN
    # ----------------------------------------------------------
    def train(self):
        print(f"\n>>> Training {self.model_name} ...")

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=self.get_pos_weight()
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.model_params,
            scoring='average_precision',
            n_iter=25,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=3,
        )

        search.fit(self.X_train, self.y_train)

        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        print("Best parameters:", self.best_params)


    # ----------------------------------------------------------
    # EVALUATE
    # ----------------------------------------------------------
    def evaluate(self):
        y_pred = self.best_model.predict(self.X_test)
        y_prob = self.best_model.predict_proba(self.X_test)[:, 1]

        self.metrics = {
            "model": self.model_name,
            "best_params": self.best_params,
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, zero_division=0),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "roc_auc": roc_auc_score(self.y_test, y_prob),
            "pr_auc": average_precision_score(self.y_test, y_prob),
        }

        print("\nClassification Report:\n")
        print(classification_report(self.y_test, y_pred))

        print("\nMetrics:")
        for k, v in self.metrics.items():
            print(f"{k}: {v}")

        return self.metrics


    # ----------------------------------------------------------
    # SAVE ARTIFACTS
    # ----------------------------------------------------------
    def save(self):
        # SHAP Explainer
        explainer = shap.TreeExplainer(self.best_model)

        # Save full training artifact
        artifact = {
            "model": self.best_model,
            "features": list(self.X_train.columns),
            "params": self.best_params,
            "metrics": self.metrics
        }

        joblib.dump(artifact, self.model_path)
        joblib.dump(explainer, self.explainer_path)

        print(f"\nModel saved to {self.model_path}")
        print(f"Explainer saved to {self.explainer_path}")


    # ----------------------------------------------------------
    # FULL PIPELINE
    # ----------------------------------------------------------
    def run(self):
        self.load_data()
        self.prepare_data()
        self.train()
        self.evaluate()
        self.save()


# ----------------------------------------------------------
# RUN SCRIPT
# ----------------------------------------------------------
if __name__ == "__main__":
    param_distributions = {
                'n_estimators': randint(100, 600),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'min_child_weight': randint(1, 10),
                'gamma': uniform(0, 5),
                'reg_lambda': uniform(0, 5),
                'reg_alpha': uniform(0, 5),
                'base_score': uniform(0.1,0.8)
            }

    trainer = ModelTrainer(
        data_path="../data/raw/cs-training.csv",
        drop_columns=['SeriousDlqin2yrs','Unnamed: 0'],
        target=['SeriousDlqin2yrs'],
        model_params=param_distributions,
        model_path="../models/xgb_model.joblib",
        explainer_path="../models/xgb_shap_explainer.joblib",
        model_name="XGBoost_Model"
    )

    trainer.run()

