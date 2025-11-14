import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import matplotlib.pyplot as plt
from datetime import datetime


API_URL = "http://localhost:8000"  # FastAPI URL


st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------
# CUSTOM STYLING
# -----------------------------------------------
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high {
        background-color: #ffcccc;
        color: #d32f2f;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #ff9800;
    }
    .risk-low {
        background-color: #d4edda;
        color: #28a745;
    }
    .el-box {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    </style>
""", unsafe_allow_html=True)


st.title("💳 Credit Risk Assessment Model")
st.markdown("*Predict Probability of Default (PD) with Expected Loss & Model Explainability*")


# -----------------------------------------------
# DEFINE FEATURES WITH METADATA
# -----------------------------------------------
FEATURE_CONFIG = {
    "RevolvingUtilizationOfUnsecuredLines": {
        "type": "slider",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "default": 0.3,
        "description": "Ratio of used credit on revolving accounts (0-1)"
    },
    "age": {
        "type": "slider",
        "min": 18,
        "max": 100,
        "step": 1,
        "default": 40,
        "description": "Age of the borrower in years"
    },
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "type": "slider",
        "min": 0,
        "max": 50,
        "step": 1,
        "default": 0,
        "description": "Number of times 30-59 days past due (not worse)"
    },
    "DebtRatio": {
        "type": "slider",
        "min": 0.0,
        "max": 5.0,
        "step": 0.1,
        "default": 0.3,
        "description": "Total debt to monthly income ratio"
    },
    "MonthlyIncome": {
        "type": "slider",
        "min": 0,
        "max": 50000,
        "step": 500,
        "default": 3000,
        "description": "Monthly income in dollars"
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "type": "slider",
        "min": 0,
        "max": 100,
        "step": 1,
        "default": 8,
        "description": "Number of open credit lines and loans"
    },
    "NumberOfTimes90DaysLate": {
        "type": "slider",
        "min": 0,
        "max": 50,
        "step": 1,
        "default": 0,
        "description": "Number of times 90 days late"
    },
    "NumberRealEstateLoansOrLines": {
        "type": "slider",
        "min": 0,
        "max": 50,
        "step": 1,
        "default": 1,
        "description": "Number of real estate loans or lines"
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "type": "slider",
        "min": 0,
        "max": 50,
        "step": 1,
        "default": 0,
        "description": "Number of times 60-89 days past due (not worse)"
    },
    "NumberOfDependents": {
        "type": "slider",
        "min": 0,
        "max": 20,
        "step": 1,
        "default": 0,
        "description": "Number of dependents"
    }
}


# -----------------------------------------------
# EXPECTED LOSS PARAMETERS CONFIG
# -----------------------------------------------
EL_PARAMS_CONFIG = {
    "LGD": {
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "default": 0.45,
        "description": "Loss Given Default (fraction of exposure lost at default)",
        "help": "Typical range: 0.25-0.75 depending on collateral and seniority"
    },
    "EAD": {
        "min": 0,
        "max": 100000,
        "step": 1000,
        "default": 10000,
        "description": "Exposure at Default (loan amount in dollars)",
        "help": "Amount of credit extended to the customer"
    }
}


# -----------------------------------------------
# SIDEBAR: FEATURE INPUT WITH SLIDERS
# -----------------------------------------------
st.sidebar.header("📊 Input Customer Profile")
st.sidebar.markdown("Adjust the sliders to explore different scenarios")

user_input = {}

# Create tabs for better organization
tab_demographics, tab_credit_history, tab_debt = st.sidebar.tabs(
    ["Demographics", "Credit History", "Debt Profile"]
)

with tab_demographics:
    user_input["age"] = st.slider(
        label=FEATURE_CONFIG["age"]["description"],
        min_value=FEATURE_CONFIG["age"]["min"],
        max_value=FEATURE_CONFIG["age"]["max"],
        value=FEATURE_CONFIG["age"]["default"],
        step=FEATURE_CONFIG["age"]["step"]
    )
    user_input["NumberOfDependents"] = st.slider(
        label=FEATURE_CONFIG["NumberOfDependents"]["description"],
        min_value=FEATURE_CONFIG["NumberOfDependents"]["min"],
        max_value=FEATURE_CONFIG["NumberOfDependents"]["max"],
        value=FEATURE_CONFIG["NumberOfDependents"]["default"],
        step=int(FEATURE_CONFIG["NumberOfDependents"]["step"])
    )
    user_input["MonthlyIncome"] = st.slider(
        label=FEATURE_CONFIG["MonthlyIncome"]["description"],
        min_value=FEATURE_CONFIG["MonthlyIncome"]["min"],
        max_value=FEATURE_CONFIG["MonthlyIncome"]["max"],
        value=FEATURE_CONFIG["MonthlyIncome"]["default"],
        step=int(FEATURE_CONFIG["MonthlyIncome"]["step"])
    )

with tab_credit_history:
    user_input["NumberOfTime30-59DaysPastDueNotWorse"] = st.slider(
        label=FEATURE_CONFIG["NumberOfTime30-59DaysPastDueNotWorse"]["description"],
        min_value=FEATURE_CONFIG["NumberOfTime30-59DaysPastDueNotWorse"]["min"],
        max_value=FEATURE_CONFIG["NumberOfTime30-59DaysPastDueNotWorse"]["max"],
        value=FEATURE_CONFIG["NumberOfTime30-59DaysPastDueNotWorse"]["default"],
        step=int(FEATURE_CONFIG["NumberOfTime30-59DaysPastDueNotWorse"]["step"])
    )
    user_input["NumberOfTime60-89DaysPastDueNotWorse"] = st.slider(
        label=FEATURE_CONFIG["NumberOfTime60-89DaysPastDueNotWorse"]["description"],
        min_value=FEATURE_CONFIG["NumberOfTime60-89DaysPastDueNotWorse"]["min"],
        max_value=FEATURE_CONFIG["NumberOfTime60-89DaysPastDueNotWorse"]["max"],
        value=FEATURE_CONFIG["NumberOfTime60-89DaysPastDueNotWorse"]["default"],
        step=int(FEATURE_CONFIG["NumberOfTime60-89DaysPastDueNotWorse"]["step"])
    )
    user_input["NumberOfTimes90DaysLate"] = st.slider(
        label=FEATURE_CONFIG["NumberOfTimes90DaysLate"]["description"],
        min_value=FEATURE_CONFIG["NumberOfTimes90DaysLate"]["min"],
        max_value=FEATURE_CONFIG["NumberOfTimes90DaysLate"]["max"],
        value=FEATURE_CONFIG["NumberOfTimes90DaysLate"]["default"],
        step=int(FEATURE_CONFIG["NumberOfTimes90DaysLate"]["step"])
    )

with tab_debt:
    user_input["DebtRatio"] = st.slider(
        label=FEATURE_CONFIG["DebtRatio"]["description"],
        min_value=FEATURE_CONFIG["DebtRatio"]["min"],
        max_value=FEATURE_CONFIG["DebtRatio"]["max"],
        value=FEATURE_CONFIG["DebtRatio"]["default"],
        step=FEATURE_CONFIG["DebtRatio"]["step"]
    )
    user_input["RevolvingUtilizationOfUnsecuredLines"] = st.slider(
        label=FEATURE_CONFIG["RevolvingUtilizationOfUnsecuredLines"]["description"],
        min_value=FEATURE_CONFIG["RevolvingUtilizationOfUnsecuredLines"]["min"],
        max_value=FEATURE_CONFIG["RevolvingUtilizationOfUnsecuredLines"]["max"],
        value=FEATURE_CONFIG["RevolvingUtilizationOfUnsecuredLines"]["default"],
        step=FEATURE_CONFIG["RevolvingUtilizationOfUnsecuredLines"]["step"]
    )
    user_input["NumberOfOpenCreditLinesAndLoans"] = st.slider(
        label=FEATURE_CONFIG["NumberOfOpenCreditLinesAndLoans"]["description"],
        min_value=FEATURE_CONFIG["NumberOfOpenCreditLinesAndLoans"]["min"],
        max_value=FEATURE_CONFIG["NumberOfOpenCreditLinesAndLoans"]["max"],
        value=FEATURE_CONFIG["NumberOfOpenCreditLinesAndLoans"]["default"],
        step=int(FEATURE_CONFIG["NumberOfOpenCreditLinesAndLoans"]["step"])
    )
    user_input["NumberRealEstateLoansOrLines"] = st.slider(
        label=FEATURE_CONFIG["NumberRealEstateLoansOrLines"]["description"],
        min_value=FEATURE_CONFIG["NumberRealEstateLoansOrLines"]["min"],
        max_value=FEATURE_CONFIG["NumberRealEstateLoansOrLines"]["max"],
        value=FEATURE_CONFIG["NumberRealEstateLoansOrLines"]["default"],
        step=int(FEATURE_CONFIG["NumberRealEstateLoansOrLines"]["step"])
    )


# -----------------------------------------------
# EXPECTED LOSS PARAMETERS SECTION
# -----------------------------------------------
st.sidebar.divider()
st.sidebar.header("💰 Expected Loss (EL) Parameters")
st.sidebar.markdown("Configure risk parameters: **EL = PD × LGD × EAD**")

el_params = {}

col1, col2 = st.sidebar.columns(2)

with col1:
    el_params["LGD"] = st.slider(
        label="Loss Given Default (LGD)",
        min_value=EL_PARAMS_CONFIG["LGD"]["min"],
        max_value=EL_PARAMS_CONFIG["LGD"]["max"],
        value=EL_PARAMS_CONFIG["LGD"]["default"],
        step=EL_PARAMS_CONFIG["LGD"]["step"],
        help=EL_PARAMS_CONFIG["LGD"]["help"]
    )

with col2:
    el_params["EAD"] = st.slider(
        label="Exposure at Default (EAD)",
        min_value=EL_PARAMS_CONFIG["EAD"]["min"],
        max_value=EL_PARAMS_CONFIG["EAD"]["max"],
        value=EL_PARAMS_CONFIG["EAD"]["default"],
        step=int(EL_PARAMS_CONFIG["EAD"]["step"]),
        help=EL_PARAMS_CONFIG["EAD"]["help"]
    )

# Show parameter descriptions
with st.sidebar.expander("ℹ️ About EL Parameters"):
    st.markdown("""
    **Probability of Default (PD)**
    - Predicted by our ML model
    - Probability the customer defaults within 1 year
    
    **Loss Given Default (LGD)**
    - What % of the loan is lost if default occurs
    - 0.45 = 45% loss (55% recovery through collateral/collections)
    - Depends on: collateral, seniority, industry
    
    **Exposure at Default (EAD)**
    - Amount of credit extended (loan principal)
    - Dollar amount at risk
    
    **Expected Loss = PD × LGD × EAD**
    - Expected monetary loss for this loan
    - Used for capital allocation & pricing
    """)


# -----------------------------------------------
# RESET BUTTON IN SIDEBAR
# -----------------------------------------------
if st.sidebar.button("🔄 Reset to Defaults", use_container_width=True):
    st.rerun()


# -----------------------------------------------
# MAIN CONTENT: PREDICTION & EXPLAINABILITY
# -----------------------------------------------
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("⚙️ Customer Profile Summary")
    with st.container():
        st.metric("Age", f"{int(user_input['age'])} years")
        st.metric("Monthly Income", f"${user_input['MonthlyIncome']:,.0f}")
        st.metric("Debt Ratio", f"{user_input['DebtRatio']:.2f}")
        st.metric("Credit Utilization", f"{user_input['RevolvingUtilizationOfUnsecuredLines']:.1%}")


# -----------------------------------------------
# PREDICTION API CALL
# -----------------------------------------------
if st.button("🔮 Predict Probability of Default", use_container_width=True):
    try:
        with st.spinner("Analyzing credit profile..."):
            # Call prediction endpoint
            response = requests.post(
                f"{API_URL}/predict",
                json={"__root__": user_input},
                timeout=10
            )
            result = response.json()

            # Call SHAP endpoint
            shap_resp = requests.post(
                f"{API_URL}/shap",
                json={"__root__": user_input},
                timeout=10
            ).json()

            # Extract results
            prediction = result.get("prediction", 0)
            probability = result.get("probability", 0.0)
            shap_values = shap_resp.get("shap_values", [])
            expected_value = shap_resp.get("expected_value", 0.0)
            features_list = shap_resp.get("features", list(user_input.keys()))

            # -----------------------------------------------
            # RISK CLASSIFICATION & MAIN METRICS
            # -----------------------------------------------
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if probability < 0.33:
                    risk_level = "🟢 LOW RISK"
                    risk_class = "risk-low"
                elif probability < 0.67:
                    risk_level = "🟡 MEDIUM RISK"
                    risk_class = "risk-medium"
                else:
                    risk_level = "🔴 HIGH RISK"
                    risk_class = "risk-high"

                st.markdown(f"""
                    <div class="metric-box {risk_class}">
                    <h3>Risk Classification</h3>
                    <h2>{risk_level}</h2>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class="metric-box">
                    <h3>Probability of Default</h3>
                    <h2>{probability:.2%}</h2>
                    <p>Expected default probability</p>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                    <div class="metric-box">
                    <h3>Model Decision</h3>
                    <h2>{"✅ Approve" if prediction == 0 else "❌ Decline"}</h2>
                    <p>Default: {prediction}</p>
                    </div>
                """, unsafe_allow_html=True)

            # -----------------------------------------------
            # EXPECTED LOSS CALCULATION & VISUALIZATION
            # -----------------------------------------------
            with col4:
                el = probability * el_params["LGD"] * el_params["EAD"]
                st.markdown(f"""
                    <div class="metric-box el-box">
                    <h3>Expected Loss</h3>
                    <h2>${el:,.2f}</h2>
                    <p>EL = PD × LGD × EAD</p>
                    </div>
                """, unsafe_allow_html=True)

            # -----------------------------------------------
            # EXPECTED LOSS DETAILED BREAKDOWN
            # -----------------------------------------------
            st.divider()
            st.subheader("💰 Expected Loss Analysis")

            el_col1, el_col2 = st.columns([1.5, 1])

            with el_col1:
                st.markdown("**Expected Loss Components**")

                # Create detailed EL breakdown table
                el_breakdown = pd.DataFrame({
                    "Component": ["Probability of Default (PD)", "Loss Given Default (LGD)", "Exposure at Default (EAD)", "Expected Loss (EL)"],
                    "Value": [f"{probability:.2%}", f"{el_params['LGD']:.2%}", f"${el_params['EAD']:,.0f}", f"${el:,.2f}"],
                    "Formula": ["Model prediction", "Risk parameter", "Loan amount", "PD × LGD × EAD"]
                })

                st.dataframe(el_breakdown, use_container_width=True, hide_index=True)

                # Risk-adjusted pricing suggestion
                annual_rate = 0.05  # Base rate 5%
                el_spread = (el / el_params["EAD"]) * 100 if el_params["EAD"] > 0 else 0
                suggested_rate = annual_rate + (el_spread / 12)  # Annualized

                st.markdown("**Risk-Adjusted Pricing Recommendation**")
                st.info(
                    f"Base Annual Rate: {annual_rate:.2%}\n\n"
                    f"EL as % of EAD: {el_spread:.2f}%\n\n"
                    f"Suggested Annual Rate: {suggested_rate:.2%}"
                )

            with el_col2:
                st.markdown("**EL Sensitivity**")

                # Create sensitivity heatmap data
                pd_range = np.linspace(0, probability * 2, 5)
                lgd_range = np.linspace(0.2, 0.8, 5)

                sensitivity_matrix = np.zeros((len(lgd_range), len(pd_range)))
                for i, lgd in enumerate(lgd_range):
                    for j, pd in enumerate(pd_range):
                        sensitivity_matrix[i, j] = pd * lgd * el_params["EAD"]

                fig, ax = plt.subplots(figsize=(8, 5))
                im = ax.imshow(sensitivity_matrix, cmap='RdYlGn_r', aspect='auto')

                ax.set_xticks(range(len(pd_range)))
                ax.set_yticks(range(len(lgd_range)))
                ax.set_xticklabels([f"{p:.1%}" for p in pd_range], fontsize=9)
                ax.set_yticklabels([f"{l:.1%}" for l in lgd_range], fontsize=9)
                ax.set_xlabel("Probability of Default", fontsize=10)
                ax.set_ylabel("Loss Given Default", fontsize=10)
                ax.set_title("Expected Loss Sensitivity ($)", fontsize=11, fontweight='bold')

                # Add text annotations
                for i in range(len(lgd_range)):
                    for j in range(len(pd_range)):
                        text = ax.text(j, i, f'${sensitivity_matrix[i, j]:,.0f}',
                                     ha="center", va="center", color="black", fontsize=8)

                plt.colorbar(im, ax=ax, label="Expected Loss ($)")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # -----------------------------------------------
            # SHAP VISUALIZATIONS
            # -----------------------------------------------
            st.divider()
            st.subheader("📈 Model Explainability (SHAP Analysis)")

            tab1, tab2 = st.tabs(["Feature Importance", "Decision Breakdown"])

            with tab1:
                st.markdown("**Which features have the most impact on this prediction?**")

                # SHAP Bar Plot
                df_input = pd.DataFrame([user_input])

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values=[shap_values],
                    features=df_input,
                    feature_names=features_list,
                    plot_type="bar",
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.caption("Mean absolute SHAP values - Higher values indicate stronger influence on prediction")

            with tab2:
                st.markdown("**How do individual features contribute to the prediction?**")

                # SHAP Force Plot
                try:
                    force_plot = shap.force_plot(
                        base_value=expected_value,
                        shap_values=shap_values,
                        features=df_input,
                        matplotlib=False
                    )
                    st.components.v1.html(force_plot.html(), height=300)
                    st.caption(
                        f"Base value (baseline): {expected_value:.2%} | "
                        f"Model output: {probability:.2%}"
                    )
                except Exception as e:
                    st.warning(f"Force plot unavailable: {str(e)}")

            # -----------------------------------------------
            # DETAILED FEATURE IMPACT TABLE
            # -----------------------------------------------
            st.divider()
            st.subheader("🔍 Feature Impact Details")

            impact_data = {
                "Feature": features_list,
                "Value": [f"{user_input[f]:.2f}" for f in features_list],
                "SHAP Impact": [f"{sv:.4f}" for sv in shap_values],
                "Direction": ["↑ Increases PD" if sv > 0 else "↓ Decreases PD" for sv in shap_values]
            }

            df_impact = pd.DataFrame(impact_data)
            df_impact["Abs. Impact"] = [abs(sv) for sv in shap_values]
            df_impact = df_impact.sort_values("Abs. Impact", ascending=False)
            df_impact = df_impact.drop("Abs. Impact", axis=1)

            st.dataframe(df_impact, use_container_width=True, hide_index=True)

            # -----------------------------------------------
            # SCENARIO ANALYSIS WITH EL IMPACT
            # -----------------------------------------------
            st.divider()
            st.subheader("🎯 What-If Scenarios")

            scenario_col1, scenario_col2, scenario_col3 = st.columns(3)

            with scenario_col1:
                if st.button("📉 Income +20%"):
                    scenario_input = user_input.copy()
                    scenario_input["MonthlyIncome"] *= 1.2

                    scenario_resp = requests.post(
                        f"{API_URL}/predict",
                        json={"__root__": scenario_input}
                    ).json()

                    new_prob = scenario_resp.get("probability", 0.0)
                    new_el = new_prob * el_params["LGD"] * el_params["EAD"]
                    el_change = new_el - el
                    pd_change = (new_prob - probability) / probability * 100

                    st.info(
                        f"📊 New PD: {new_prob:.2%} ({pd_change:+.1f}%)\n\n"
                        f"💰 New EL: ${new_el:,.2f} ({el_change:+.2f})"
                    )

            with scenario_col2:
                if st.button("📉 Debt Ratio -20%"):
                    scenario_input = user_input.copy()
                    scenario_input["DebtRatio"] *= 0.8

                    scenario_resp = requests.post(
                        f"{API_URL}/predict",
                        json={"__root__": scenario_input}
                    ).json()

                    new_prob = scenario_resp.get("probability", 0.0)
                    new_el = new_prob * el_params["LGD"] * el_params["EAD"]
                    el_change = new_el - el
                    pd_change = (new_prob - probability) / probability * 100

                    st.info(
                        f"📊 New PD: {new_prob:.2%} ({pd_change:+.1f}%)\n\n"
                        f"💰 New EL: ${new_el:,.2f} ({el_change:+.2f})"
                    )

            with scenario_col3:
                if st.button("💳 Late Payments: 0"):
                    scenario_input = user_input.copy()
                    scenario_input["NumberOfTime30-59DaysPastDueNotWorse"] = 0
                    scenario_input["NumberOfTime60-89DaysPastDueNotWorse"] = 0
                    scenario_input["NumberOfTimes90DaysLate"] = 0

                    scenario_resp = requests.post(
                        f"{API_URL}/predict",
                        json={"__root__": scenario_input}
                    ).json()

                    new_prob = scenario_resp.get("probability", 0.0)
                    new_el = new_prob * el_params["LGD"] * el_params["EAD"]
                    el_change = new_el - el
                    pd_change = (new_prob - probability) / probability * 100

                    st.info(
                        f"📊 New PD: {new_prob:.2%} ({pd_change:+.1f}%)\n\n"
                        f"💰 New EL: ${new_el:,.2f} ({el_change:+.2f})"
                    )

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to API. Make sure FastAPI is running at "
            f"{API_URL}"
        )
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


# -----------------------------------------------
# FOOTER
# -----------------------------------------------
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px; margin-top: 20px;'>
    💡 This model predicts Probability of Default (PD) and calculates Expected Loss (EL).
    Use SHAP explanations to understand model decisions. EL = PD × LGD × EAD for financial impact quantification.
    </div>
""", unsafe_allow_html=True)
