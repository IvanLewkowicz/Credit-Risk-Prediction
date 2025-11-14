import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

# ---------------------------
# TRAINING SCRIPT
# ---------------------------


def train_and_save_model(
    data_path: str, target: str, model_params, model_path: str, explainer_path: str
):
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # SHAP Explainer
    explainer = shap.TreeExplainer(model)

    # Save artifacts
    artifact = {"model": model, "features": list(X.columns)}
    joblib.dump(artifact, model_path)
    joblib.dump(explainer, explainer_path)
    print(f"Model saved to {model_path}")
    print(f"Explainer saved to {explainer_path}")


def generate_train_test_report(name, model, params, X_train, X_test, y_train, y_test, cv):
    print(f"\n>>> Training {name} ...")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=25,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=3,
    )

    search.fit(X_train, y_train)

    # Evaluate best model
    best_model = search.best_estimator_
    best_parameter = search.best_params_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "best_params": search.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    print(f"Classification report for {name}:\n", classification_report(y_test, y_pred))

    return best_model, best_parameter, metrics


def get_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test


def get_pos_weight(y_train):
    # Calculate class imbalance ratio
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    print(f"Class ratio: {neg}:{pos}, scale_pos_weight = {scale_pos_weight:.2f}")

    return scale_pos_weight


if __name__ == "__main__":
    train_and_save_model(
        data_path="dataset.csv",
        target="target_column",
        model_path="model_artifact.joblib",
        explainer_path="shap_explainer.joblib",
    )
