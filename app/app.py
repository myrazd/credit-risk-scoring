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
    This dashboard estimates borrower default risk, assigns a risk category,
    and provides a loan approval recommendation based on a trained machine learning model.
    """
)
