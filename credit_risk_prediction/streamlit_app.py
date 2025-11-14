import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------
# CONFIG
# --------------------------------------
st.set_page_config(
    page_title="Credit Risk Scoring",
    layout="wide"
)

st.title("📊 Credit Risk Prediction & Explainability")


# --------------------------------------
# API URL
# --------------------------------------
st.sidebar.header("API Settings")
api_url = st.sidebar.text_input(
    "FastAPI URL:",
    value="http://api:8000"
)

# Utils
def call_api(endpoint: str, payload: dict):
    try:
        url = f"{api_url}{endpoint}"
        res = requests.post(url, json=payload)
        if res.status_code != 200:
            raise ValueError(res.text)
        return res.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# --------------------------------------
# Get model features from API
# --------------------------------------
@st.cache_data
def load_features():
    try:
        info = requests.get(f"{api_url}/model_info").json()
        return info.get("features", [])
    except:
        st.warning("⚠ Unable to load model features. Ensure /model_info exists.")
        return []


features = ["RevolvingUtilizationOfUnsecuredLines",
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


# --------------------------------------
# Sidebar Input Form
# --------------------------------------
st.sidebar.header("Customer Input")

if not features:
    st.stop()

user_input = {}
for feat in features:
    user_input[feat] = st.sidebar.number_input(feat, value=0.0)


# LGD / EAD (Only for EL endpoint)
st.sidebar.subheader("Expected Loss Parameters")
lgd_value = st.sidebar.number_input("LGD", value=0.45)
ead_value = st.sidebar.number_input("EAD", value=10000)


# --------------------------------------
# Prediction Section
# --------------------------------------
st.header("📈 Default Probability Prediction")

if st.button("Predict Default Risk"):
    response = call_api("/predict", user_input)

    if response:
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)

        col1.metric("Prediction (0=No Default, 1=Default)", response["prediction"])
        col2.metric("Probability of Default", f"{response['probability']:.4f}")
        col3.metric("Risk Level", response["risk_level"])

        st.success("Prediction completed.")


# --------------------------------------
# Expected Loss Section
# --------------------------------------
st.header("💰 Expected Loss Calculator")

el_input = {**user_input, "LGD": lgd_value, "EAD": ead_value}

if st.button("Calculate Expected Loss"):
    response = call_api("/expected_loss", el_input)

    if response:
        st.subheader("Expected Loss Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("PD", f"{response['probability']:.4f}")
        col2.metric("Expected Loss", f"{response['expected_loss']:.2f}")
        col3.metric("EL %", f"{response['el_percentage']:.2f}%")

        st.metric("Risk Level", response["risk_level"])

        st.success("Expected Loss calculated.")


# --------------------------------------
# SHAP Explanation Section
# --------------------------------------
st.header("🔍 SHAP Explainability")

if st.button("Generate SHAP Explanation"):
    response = call_api("/shap", user_input)

    if response:
        features = response["features"]
        shap_values = response["shap_values"]

        df = pd.DataFrame({
            "feature": features,
            "shap_value": shap_values
        }).sort_values("shap_value", key=abs, ascending=False)

        st.subheader("Top Feature Contributions")
        st.dataframe(df)

        fig = px.bar(
            df.head(15),
            x="shap_value",
            y="feature",
            orientation="h",
            title="Top 15 SHAP Contributions",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success("SHAP explanation generated.")
