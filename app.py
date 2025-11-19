"""
Streamlit app (precise) for your files:
 - Employee_model.pkl
 - Employee_Cleaned_Data.csv

Behavior (explicit):
 1. Loads Employee_Cleaned_Data.csv and expects columns:
      ['Age', 'Experience', 'Salary', 'Department', 'Feedback']
 2. Attempts to load Employee_model.pkl (must be a pickled sklearn-style model
    or sklearn Pipeline). If loading fails, trains a fallback model on the CSV.
 3. Detects whether Feedback is numeric (regression) or categorical (classification)
    and uses a suitable fallback estimator.
 4. Presents sidebar inputs for Age, Experience, Salary, Department and shows prediction.

Place files in same folder as this script and run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# ---- Config ----
DATA_FILE = "Employee_Cleaned_Data.csv"
MODEL_FILE = "Employee_model.pkl"
EXPECTED_COLUMNS = ["Age", "Experience", "Salary", "Department", "Feedback"]
st.set_page_config(page_title="Employee Feedback Predictor", layout="centered")

# ---- Load data ----
st.title("Employee Feedback Predictor (precise)")
st.write("Predict `Feedback` from Age, Experience, Salary and Department.")

if not os.path.exists(DATA_FILE):
    st.error(f"Could not find `{DATA_FILE}` in the working directory. Place the CSV next to this script.")
    st.stop()

df = pd.read_csv(DATA_FILE)
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"The CSV is missing expected columns: {missing}. Expected exactly: {EXPECTED_COLUMNS}")
    st.write("CSV columns detected:", list(df.columns))
    st.stop()

# Basic cleaning for training use
df = df[EXPECTED_COLUMNS].copy()
df = df.dropna().reset_index(drop=True)
if df.empty:
    st.error("CSV contained no non-missing rows after dropping NA.")
    st.stop()

# Distinguish problem type
target_col = "Feedback"
is_regression = pd.api.types.is_numeric_dtype(df[target_col])
problem_type = "regression" if is_regression else "classification"
st.write(f"Detected problem type: **{problem_type}** (column `{target_col}`)")

# Sidebar: inputs
st.sidebar.header("Employee features")
age_default = int(df["Age"].median()) if pd.api.types.is_numeric_dtype(df["Age"]) else 30
exp_default = int(df["Experience"].median()) if pd.api.types.is_numeric_dtype(df["Experience"]) else 2
sal_default = float(df["Salary"].median()) if pd.api.types.is_numeric_dtype(df["Salary"]) else 30000.0
dept_options = sorted(df["Department"].dropna().unique().tolist())

age = st.sidebar.number_input("Age", min_value=15, max_value=100, value=age_default, step=1)
experience = st.sidebar.number_input("Experience (years)", min_value=0, max_value=80, value=exp_default, step=1)
salary = st.sidebar.number_input("Salary", min_value=0.0, max_value=10_000_000.0, value=sal_default, step=100.0, format="%.2f")
department = st.sidebar.selectbox("Department", options=dept_options)

input_df = pd.DataFrame([{
    "Age": age,
    "Experience": experienc
