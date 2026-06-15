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
