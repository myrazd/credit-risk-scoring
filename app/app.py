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

# Sidebar information
st.sidebar.title("About This App")

st.sidebar.markdown(
    """
    This dashboard uses a trained XGBoost model to estimate borrower credit risk.

    It provides:

    - Probability of Default (PD)
    - Credit score estimate
    - Risk category
    - Loan approval recommendation
    - Expected loss estimate
    """
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "For portfolio and educational use only. Not financial advice."
)

# App title
st.title("💳 Credit Risk Scoring & Loan Approval System")

st.markdown(
    """
    Estimate borrower default risk, assign a risk category, and generate a loan approval recommendation.
    """
)

# Dashboard note
st.info(
    "Adjust borrower inputs below to generate a credit risk score, loan decision, and expected loss estimate."
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load(
        "models/credit_risk_model.pkl"
    )
    return model

model = load_model()

# Risk category guide
with st.expander("Risk Category Guide"):

    risk_guide = pd.DataFrame({
        "Risk Category": [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        "Probability of Default (PD)": [
            "< 5%",
            "5% - 15%",
            "> 15%"
        ],
        "Typical Decision": [
            "Approve",
            "Manual Review",
            "Reject"
        ]
    })

    st.dataframe(
        risk_guide,
        use_container_width=True
    )

# Credit score interpretation
with st.expander("Credit Score Guide"):

    score_guide = pd.DataFrame({
        "Credit Score Range": [
            "750 - 850",
            "650 - 749",
            "550 - 649",
            "300 - 549"
        ],
        "Interpretation": [
            "Excellent",
            "Good",
            "Fair",
            "High Risk"
        ]
    })

    st.dataframe(
        score_guide,
        use_container_width=True
    )

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

# Create missing income indicator
missing_income = int(monthly_income == 0)

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
    "MissingIncome": [missing_income]
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

# Reorder columns to match model training data
model_features = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "MissingIncome",
    "TotalDelinquencies",
    "HasDelinquencyHistory",
    "HighUtilizationFlag",
    "HasRealEstateLoan"
]

input_data = input_data[model_features]

# Show model input
with st.expander("View Model Input"):
    st.dataframe(input_data)

# Add loan amount input
loan_amount = st.number_input(
    "Requested Loan Amount",
    min_value=1000.0,
    max_value=50000.0,
    value=10000.0,
    step=1000.0
)

# Add LGD input
lgd = st.slider(
    "Loss Given Default (LGD)",
    min_value=0.10,
    max_value=1.00,
    value=0.60,
    step=0.05
)

# Generate prediction
if st.button("Generate Credit Risk Assessment"):

    default_probability = model.predict_proba(
        input_data
    )[0, 1]

    credit_score = max(
    300,
    min(
        850,
        int(850 - (default_probability * 550))
    )
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

    expected_loss = (
        default_probability
        * loan_amount
        * lgd
    )

    st.subheader("Credit Risk Assessment Results")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Probability of Default (PD)",
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

    col4.metric(
        "Loan Amount",
        f"RM{loan_amount:,.0f}"
    )

    col5.metric(
        "Expected Loss",
        f"RM{expected_loss:,.2f}"
    )

    st.info(
        f"Recommended Loan Decision: **{loan_decision}**"
    )

    # Display credit score progress
    st.progress(
        credit_score / 850
    )
    
    st.caption(
        "Credit score is scaled from 300 to 850, where higher scores indicate lower estimated credit risk."
    )

    # Decision explanation
    if risk_category == "Low Risk":
        st.success(
            """
            Low-risk borrower profile detected.
    
            Recommendation:
            - Loan can generally be approved.
            - Expected default risk is relatively low.
            - Portfolio exposure appears manageable.
            """
        )
    
    elif risk_category == "Medium Risk":
        st.warning(
            """
            Moderate-risk borrower profile detected.
    
            Recommendation:
            - Manual review is recommended.
            - Additional verification may be required.
            - Consider reviewing income stability and repayment history.
            """
        )
    
    else:
        st.error(
            """
            High-risk borrower profile detected.

            Recommendation:
            - Loan rejection is recommended.
            - Probability of Default (PD) is elevated.
            - Expected portfolio loss may be significant.
            """
        )

    # Borrower risk factor explanation
    st.subheader("Key Risk Factors")

    risk_factors = []

    if revolving_utilization >= 0.80:
        risk_factors.append(
            "High revolving credit utilization"
        )

    if late_30_59 > 0:
        risk_factors.append(
            "Recent 30-59 days past due history"
        )

    if late_60_89 > 0:
        risk_factors.append(
            "Recent 60-89 days past due history"
        )

    if late_90 > 0:
        risk_factors.append(
            "Severe 90+ days late history"
        )

    if debt_ratio >= 1.00:
        risk_factors.append(
            "Elevated debt ratio"
        )

    if monthly_income < 3000:
        risk_factors.append(
            "Lower monthly income"
        )

    if len(risk_factors) > 0:
        for factor in risk_factors:
            st.markdown(f"• {factor}")
    else:
        st.write(
            "No major high-risk indicators detected based on the input values."
        )
        

    # Show risk threshold explanation
    with st.expander("Decision Methodology"):
        st.markdown(
            """
            The recommendation is based on predicted Probability of Default (PD):
    
            - **Approve:** Probability of Default (PD) below 5%
            - **Manual Review:** Probability of Default (PD) between 5% and 15%
            - **Reject:** Probability of Default (PD) above 15%
    
            Expected loss is calculated using:
    
            **Expected Loss = Probability of Default (PD) × Loan Amount × LGD**
            """
        )


    # Create assessment report
    assessment_report = pd.DataFrame({
        "Metric": [
            "Probability of Default (PD)",
            "Credit Score",
            "Risk Category",
            "Loan Decision",
            "Loan Amount",
            "LGD",
            "Expected Loss"
        ],
        "Value": [
            f"{default_probability:.2%}",
            credit_score,
            risk_category,
            loan_decision,
            f"RM{loan_amount:,.0f}",
            f"{lgd:.0%}",
            f"RM{expected_loss:,.2f}"
        ]
    })

    st.download_button(
        label="Download Assessment Report",
        data=assessment_report.to_csv(
            index=False
        ),
        file_name="credit_risk_assessment.csv",
        mime="text/csv"
    )

# Project summary
st.markdown("---")

st.header("Model Overview")

summary_col1, summary_col2, summary_col3 = st.columns(3)

summary_col1.metric(
    "Final Model",
    "Feature Engineered XGBoost"
)

summary_col2.metric(
    "ROC-AUC",
    "0.869"
)

summary_col3.metric(
    "Training Records",
    "150,000"
)

st.caption(
    "This app is designed for portfolio and educational purposes. It is not intended for real-world lending decisions."
)
