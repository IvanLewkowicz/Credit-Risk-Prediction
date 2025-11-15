# 💳 Credit Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-orange)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://www.docker.com/)

---

## Problem Description

Credit risk assessment is a critical task in financial services, aimed at evaluating the likelihood that a borrower will default on a loan. Accurately predicting the Probability of Default (PD) allows lenders to manage risk, set appropriate pricing, and comply with regulatory capital requirements. This project builds a machine learning system to predict PD, quantify Expected Loss (EL), and provide interpretable insights into model decisions, enhancing decision-making and transparency in credit lending.

---

## Exploratory Data Analysis (EDA)

- Conducted thorough data exploration on the [publicly available dataset from Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data).
- Analyzed feature value ranges, missing data patterns, and target distribution.
- Investigated feature importance to guide model development using correlation analysis and SHAP values.
- Data cleaning and feature engineering steps are included in the training script for reproducibility.

---

## Model Training

- Trained multiple models including logistic regression and XGBoost classifiers.
- Utilized extensive hyperparameter tuning for XGBoost using RandomizedSearchCV to optimize model performance.
- Metrics evaluated include Accuracy, Precision, Recall, F1, ROC-AUC, and Precision-Recall AUC.
- The final model achieved approximately 88% ROC-AUC on hold-out test data, demonstrating strong predictive power.

---

## Exported Training Script

- Training logic is encapsulated in `src/train.py`, a standalone script.
- Easily reproducible model training pipeline with CSV inputs.
- Saves trained model artifact and SHAP explainer for downstream inference and explanation.

---

## Reproducibility

- Dataset used for training is publicly available and instructions to download it are provided.
- All scripts and environment dependencies are captured in the repo.
- Training can be reproduced end-to-end with no errors following instructions.

---

## Model Deployment

- REST API developed with FastAPI and Uvicorn ASGI server serving predictions and explanations.
- API includes endpoints for:
  - Single prediction (`/predict`)
  - Expected Loss calculation (`/expected_loss`)
  - SHAP explainability (`/shap`)
  - Batch prediction endpoints
- Interactive frontend developed in Streamlit for real-time user input adjustment, predictions, and interpretability.

---

## Dependency and Environment Management

- Project dependencies are listed in `requirements.txt` generated from `pyproject.toml`.
- Instructions provided to create and activate a Python virtual environment for development.
- Includes all ML and web app dependencies such as `xgboost`, `scikit-learn`, `fastapi`, `streamlit`, `shap`, and `uvicorn`.

---

## Containerization

- Dockerfiles provided for both FastAPI backend and Streamlit frontend apps.
- Images built using Python 3.10 slim base, installing dependencies and copying source/model files.
- API container runs Uvicorn with multiple workers (`--workers 4`) for production readiness.
- Docker Compose orchestrates multi-container deployment, linking API and frontend with volume mounts for models.
- README contains step-by-step instructions for building, running, and managing containers.

---

## Cloud Deployment with Render

- Detailed deployment instructions included for [Render](https://render.com/), a popular cloud platform for easy container app deployment.
- Configurations provided for building Docker images and deploying with environment variables.
- Includes setup for continuous deployment from GitHub repo.
- Live URL example accessible after deployment to demonstrate working system online.

---

## Live Demo

A live demo of the deployed application is available at:  
[https://your-render-app.onrender.com](https://your-render-app.onrender.com)  
(Check your Render dashboard for the exact URL after deployment)

---

## Project Structure
```
credit-risk-prediction/
├── data/raw/cs-training.csv # Raw training data
├── models/ # Model artifacts and explainers
├── src/
│ ├── train.py # Model training pipeline
│ ├── predict.py # Model prediction and explanation logic
│ ├── api.py # FastAPI backend
│ └── app.py # Streamlit frontend
├── docker/
│ ├── Dockerfile.api # API container build
│ ├── Dockerfile.app # Streamlit container build
│ └── docker-compose.yml # Compose orchestration
├── requirements.txt # Python dependencies
├── README.md # This documentation
└── LICENSE # License information
```


---

## How to Use

### Locally

1. Clone repo and download dataset as instructed.
2. Create and activate Python virtual environment and install dependcies.
```
uv venv
uv sync
uv add --lock

```

2. Train the model: `python src/train.py`
3. Start API server: `uvicorn credit_risk_prediction.app:app --host 0.0.0.0 --port 8000 --workers 4`
4. Start frontend: `streamlit run credit_risk_prediction/streamlit_app.py --server.port=8501`
5. Open browser: frontend at http://localhost:8501 and API docs at http://localhost:8000/docs

### Using Docker

1. Build containers: `docker-compose build`
2. Run containers: `docker-compose up`
3. Access as above on mapped ports

### Using Render (Cloud)

1. Connect repo to Render and follow provided cloud deployment guide.
2. Configure environment variables and container settings.
3. Deploy and get live URL for external access.

---

## Contact

Author: Ivan Lewkowicz  
GitHub: [https://github.com/IvanLewkowicz/Credit-Risk-Prediction](https://github.com/IvanLewkowicz/Credit-Risk-Prediction)  
LinkedIn: [https://linkedin.com/in/ivan-lewkowicz](https://linkedin.com/in/ivan-lewkowicz)  
Email: your.email@example.com

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

Special thanks to open-source communities and libraries that made this project feasible:  
XGBoost, SHAP, FastAPI, Streamlit, Uvicorn, Docker, Render.

