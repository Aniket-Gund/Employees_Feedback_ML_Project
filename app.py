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
    "Experience": experience,
    "Salary": salary,
    "Department": department
}])

st.subheader("Input preview")
st.table(input_df)

# ---- Preprocessor and pipeline builder (explicit) ----
numeric_features = ["Age", "Experience", "Salary"]
categorical_features = ["Department"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
    ],
    remainder="drop",
)

def try_load_model(path):
    if not os.path.exists(path):
        return None, f"{path} not found"
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, str(e)

# ---- Load or train fallback ----
model, load_err = try_load_model(MODEL_FILE)

fallback_used = False
if model is None:
    fallback_used = True
    st.sidebar.warning(f"Failed to load `{MODEL_FILE}`: {load_err}. Training fallback model on CSV.")
    X = df[numeric_features + categorical_features].copy()
    y = df[target_col].copy()
    # Train/test split
    stratify = y if (not is_regression and len(y.unique()) > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=stratify)
    # Choose estimator
    if is_regression:
        estimator = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        estimator = RandomForestClassifier(n_estimators=200, random_state=42)
    pipeline = Pipeline(steps=[("pre", preprocessor), ("est", estimator)])
    pipeline.fit(X_train, y_train)
    # quick metric
    y_pred = pipeline.predict(X_test)
    if is_regression:
        metric = r2_score(y_test, y_pred)
        st.sidebar.success(f"Fallback trained — R2 on test set: {metric:.3f}")
    else:
        metric = accuracy_score(y_test, y_pred)
        st.sidebar.success(f"Fallback trained — Accuracy on test set: {metric:.3f}")
    model = pipeline
else:
    st.sidebar.success(f"Loaded model from `{MODEL_FILE}` successfully.")

# ---- Prediction ----
st.subheader("Prediction")
try:
    # Try predicting directly (works if model is pipeline or accepts raw columns)
    prediction = model.predict(input_df)
except Exception:
    # Try transforming then predicting
    try:
        X_trans = preprocessor.transform(input_df)
        prediction = model.predict(X_trans)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

pred_value = prediction[0]
st.write(f"Predicted `{target_col}`: **{pred_value}**")

# If classifier with probabilities, show them
if (not is_regression) and hasattr(model, "predict_proba"):
    try:
        proba = model.predict_proba(input_df)
        classes = model.classes_
        proba_df = pd.DataFrame(proba, columns=[str(c) for c in classes])
        st.write("Prediction probabilities:")
        st.table(proba_df.T)
    except Exception:
        pass

# If pipeline with feature importances (fallback case), show top importances
st.subheader("Model details")
st.write(f"Model object type: `{type(model).__name__}`")
st.write("Fallback model used:" , fallback_used)

# Feature importances best-effort (works for RandomForest under pipeline named 'est')
try:
    if hasattr(model, "named_steps") and "est" in model.named_steps:
        est = model.named_steps["est"]
        if hasattr(est, "feature_importances_"):
            # build output feature names
            ohe = model.named_steps["pre"].named_transformers_.get("cat")
            cat_names = []
            if ohe is not None and hasattr(ohe, "get_feature_names_out"):
                cat_names = list(ohe.get_feature_names_out(categorical_features))
            feat_names = numeric_features + cat_names
            importances = est.feature_importances_
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
            st.write("Top feature importances:")
            st.dataframe(imp_df)
except Exception:
    pass

st.write("---")
st.markdown(
    """
**Notes (precise):**
- `app.py` expects exactly the columns listed at top of the file. If your CSV differs, modify `EXPECTED_COLUMNS`.
- If loading `Employee_model.pkl` fails due to binary compatibility (e.g. `numpy._core`), the app trains a local RandomForest fallback.
- For production, use a binary-identical environment to the one that produced `Employee_model.pkl` or export your model as ONNX / joblib from that environment.
"""
)
