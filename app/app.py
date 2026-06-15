# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure page
st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon="💳",
    layout="wide"
)

# App title
st.title("💳 Credit Risk Scoring & Loan Approval System")

st.markdown(
    """
    Estimate borrower default risk, assign a risk category, and generate a loan approval recommendation.
    """
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load(
        "models/credit_risk_model.pkl"
    )
    return model

model = load_model()

st.success("Model loaded successfully.")

# Create borrower input form
st.header("Borrower Information")

col1, col2 = st.columns(2)

with col1:
    revolving_utilization = st.number_input(
        "Revolving Utilization of Unsecured Lines",
        min_value=0.0,
        max_value=2.0,
        value=0.30,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )

    debt_ratio = st.number_input(
        "Debt Ratio",
        min_value=0.0,
        max_value=5.0,
        value=0.35,
        step=0.01
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

with col2:
    late_30_59 = st.number_input(
        "Times 30-59 Days Past Due",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    late_60_89 = st.number_input(
        "Times 60-89 Days Past Due",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    late_90 = st.number_input(
        "Times 90+ Days Late",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    open_credit_lines = st.number_input(
        "Open Credit Lines and Loans",
        min_value=0,
        max_value=60,
        value=8,
        step=1
    )

    real_estate_loans = st.number_input(
        "Real Estate Loans or Lines",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

# Create prediction input
input_data = pd.DataFrame({
    "RevolvingUtilizationOfUnsecuredLines": [revolving_utilization],
    "age": [age],
    "NumberOfTime30-59DaysPastDueNotWorse": [late_30_59],
    "DebtRatio": [debt_ratio],
    "MonthlyIncome": [monthly_income],
    "NumberOfOpenCreditLinesAndLoans": [open_credit_lines],
    "NumberOfTimes90DaysLate": [late_90],
    "NumberRealEstateLoansOrLines": [real_estate_loans],
    "NumberOfTime60-89DaysPastDueNotWorse": [late_60_89],
    "NumberOfDependents": [dependents],
    "MissingIncome": [0]
})

# Add engineered features
input_data["TotalDelinquencies"] = (
    input_data["NumberOfTime30-59DaysPastDueNotWorse"]
    + input_data["NumberOfTime60-89DaysPastDueNotWorse"]
    + input_data["NumberOfTimes90DaysLate"]
)

input_data["HasDelinquencyHistory"] = (
    input_data["TotalDelinquencies"] > 0
).astype(int)

input_data["HighUtilizationFlag"] = (
    input_data["RevolvingUtilizationOfUnsecuredLines"] >= 0.80
).astype(int)

input_data["HasRealEstateLoan"] = (
    input_data["NumberRealEstateLoansOrLines"] > 0
).astype(int)

# Show model input
with st.expander("View Model Input"):
    st.dataframe(input_data)

# Generate prediction
if st.button("Generate Credit Risk Assessment"):

    default_probability = model.predict_proba(
        input_data
    )[0, 1]

    credit_score = int(
        850 - (default_probability * 550)
    )

    if default_probability < 0.05:
        risk_category = "Low Risk"
        loan_decision = "Approve"
    elif default_probability < 0.15:
        risk_category = "Medium Risk"
        loan_decision = "Manual Review"
    else:
        risk_category = "High Risk"
        loan_decision = "Reject"

    st.subheader("Credit Risk Assessment Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Default Probability",
        f"{default_probability:.2%}"
    )

    col2.metric(
        "Credit Score",
        credit_score
    )

    col3.metric(
        "Risk Category",
        risk_category
    )

    st.info(
        f"Recommended Loan Decision: **{loan_decision}**"
    )
