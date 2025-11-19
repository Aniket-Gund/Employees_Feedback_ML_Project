# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# -------------------
# Helper: safe OneHotEncoder (works for all sklearn versions)
from inspect import signature
def make_ohe():
    sig = signature(OneHotEncoder)
    if "sparse_output" in sig.parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
# -------------------

st.set_page_config(page_title="Employee Feedback Predictor", layout="centered")
st.title("Employee Feedback Predictor")

DATA_FILE = "Employee_Cleaned_Data.csv"
MODEL_FILE = "Employee_model.pkl"

# Load CSV (for valid department choices)
if not os.path.exists(DATA_FILE):
    st.error(f"Missing file: {DATA_FILE}")
    st.stop()

df = pd.read_csv(DATA_FILE)
required_cols = ["Age", "Experience", "Salary", "Department"]
for c in required_cols + ["Feedback"]:
    if c not in df.columns:
        st.error("CSV file must contain: Age, Experience, Salary, Department, Feedback")
        st.stop()

df = df.dropna().reset_index(drop=True)

# Sidebar Inputs
st.sidebar.header("Enter Employee Details")

age = st.sidebar.number_input("Age", value=int(df.Age.median()), min_value=15, max_value=100)
exp = st.sidebar.number_input("Experience", value=int(df.Experience.median()), min_value=0, max_value=80)
salary = st.sidebar.number_input("Salary", value=float(df.Salary.median()), min_value=0.0, step=100.0)
dept = st.sidebar.selectbox("Department", sorted(df.Department.unique()))

input_df = pd.DataFrame([{
    "Age": age,
    "Experience": exp,
    "Salary": salary,
    "Department": dept
}])

# Preprocessor
numeric = ["Age", "Experience", "Salary"]
categorical = ["Department"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric),
        ("cat", make_ohe(), categorical)
    ]
)

# Load Model
if not os.path.exists(MODEL_FILE):
    st.error(f"Missing model file: {MODEL_FILE}")
    st.stop()

try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------
# Prediction
# -------------------
try:
    pred = model.predict(input_df)[0]
except:
    # try with preprocessing
    X_trans = preprocessor.fit(df[numeric + categorical]).transform(input_df)
    pred = model.predict(X_trans)[0]

# -------------------
# Output (Clean & Minimal)
# -------------------
st.subheader("Input Values")
st.write(input_df)

st.subheader("Predicted Feedback")
st.success(str(pred))
