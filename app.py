# app.py
"""
Streamlit app for Employee Feedback prediction.

Place these files in the same folder as this script:
 - Employee_Cleaned_Data.csv
 - Employee_model.pkl   (optional; app will attempt to load it)

Behavior:
 - Detects the problem type (classification vs regression) from the CSV Feedback column.
 - Attempts to load the provided pickled model.
 - If loading fails, trains a fallback RandomForest (classifier or regressor) on the CSV.
 - Accepts inputs: Age, Experience, Salary, Department and shows prediction + optional probabilities.
 - Uses a OneHotEncoder constructor that is compatible across sklearn versions.
"""

import os
import pickle
import inspect
from inspect import signature

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# fallback estimators
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Try importing OneHotEncoder in a version-safe way
try:
    from sklearn.preprocessing import OneHotEncoder
except Exception as e:
    raise RuntimeError("scikit-learn import failed. Make sure scikit-learn is installed.") from e

# ----------------- Config -----------------
DATA_FILE = "Employee_Cleaned_Data.csv"
MODEL_FILE = "Employee_model.pkl"
EXPECTED_COLUMNS = ["Age", "Experience", "Salary", "Department", "Feedback"]

st.set_page_config(page_title="Employee Feedback Predictor", layout="centered")
st.title("Employee Feedback Predictor")

st.write(
    "Drop `Employee_Cleaned_Data.csv` and optionally `Employee_model.pkl` into the folder with this script, then run `streamlit run app.py`."
)

# ----------------- Helpers -----------------
def make_onehot_encoder(handle_unknown="ignore", sparse_output=False, sparse=False):
    """
    Construct OneHotEncoder in a way that works across sklearn versions:
    - Use sparse_output if the constructor supports it (newer sklearn)
    - Otherwise use sparse (older sklearn)
    """
    try:
        sig = signature(OneHotEncoder)
        if "sparse_output" in sig.parameters:
            # newer sklearn
            return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_output)
        else:
            # older sklearn
            return OneHotEncoder(handle_unknown=handle_unknown, sparse=sparse)
    except Exception:
        # Last resort: construct with minimal args; handle_unknown supported in many versions.
        return OneHotEncoder(handle_unknown=handle_unknown)

@st.cache_data
def load_csv(path=DATA_FILE):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

def try_load_model(path=MODEL_FILE):
    """
    Attempt to open and pickle.load the model. Return (model, error_message_or_None).
    """
    if not os.path.exists(path):
        return None, f"{path} not found"
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, str(e)

def build_preprocessor(numeric_features, categorical_features):
    ohe = make_onehot_encoder(handle_unknown="ignore", sparse_output=False, sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", ohe, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor

# ----------------- Load and validate data -----------------
df = load_csv()
if df is None:
    st.error(f"CSV file `{DATA_FILE}` not found in working directory. Place it next to this script.")
    st.stop()

# Validate expected columns
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"CSV is missing expected columns: {missing}")
    st.write("Detected columns in CSV:", list(df.columns))
    st.stop()

# Keep only expected columns (defensive)
df = df[EXPECTED_COLUMNS].copy()
df = df.dropna().reset_index(drop=True)
if df.shape[0] == 0:
    st.error("CSV contains no rows after dropping NA. Please provide a non-empty CSV.")
    st.stop()

# Determine target type
target_col = "Feedback"
is_regression = pd.api.types.is_numeric_dtype(df[target_col])
problem_type = "regression" if is_regression else "classification"
st.write(f"Detected problem type: **{problem_type}** (column `{target_col}`)")

# ----------------- Sidebar inputs -----------------
st.sidebar.header("Employee features (input)")
age_default = int(df["Age"].median()) if pd.api.types.is_numeric_dtype(df["Age"]) else 30
exp_default = int(df["Experience"].median()) if pd.api.types.is_numeric_dtype(df["Experience"]) else 2
sal_default = float(df["Salary"].median()) if pd.api.types.is_numeric_dtype(df["Salary"]) else 30000.0
dept_options = sorted(df["Department"].dropna().unique().tolist())

age = st.sidebar.number_input("Age", min_value=15, max_value=100, value=age_default, step=1)
experience = st.sidebar.number_input("Experience (years)", min_value=0, max_value=80, value=exp_default, step=1)
salary = st.sidebar.number_input("Salary", min_value=0.0, max_value=10_000_000.0, value=sal_default, step=100.0, format="%.2f")
department = st.sidebar.selectbox("Department", options=dept_options)

attempt_to_load_model = st.sidebar.checkbox("Attempt to load Employee_model.pkl", value=True)

input_df = pd.DataFrame([{
    "Age": age,
    "Experience": experience,
    "Salary": salary,
    "Department": department
}])

st.subheader("Input preview")
st.table(input_df)

# ----------------- Preprocessor -----------------
numeric_features = ["Age", "Experience", "Salary"]
categorical_features = ["Department"]
preprocessor = build_preprocessor(numeric_features, categorical_features)

# ----------------- Load or train model -----------------
uploaded_model, load_err = (None, None)
if attempt_to_load_model:
    uploaded_model, load_err = try_load_model(MODEL_FILE)

fallback_used = False
model = None

if uploaded_model is not None:
    st.sidebar.success("Uploaded model loaded.")
    model = uploaded_model
else:
    if attempt_to_load_model:
        st.sidebar.warning(f"Could not load `{MODEL_FILE}`: {load_err}")
    # Train fallback model
    st.sidebar.info("Training fallback model on the CSV (so the app remains usable).")
    fallback_used = True

    X = df[numeric_features + categorical_features].copy()
    y = df[target_col].copy()

    # Basic fill for safety
    X = X.fillna(method="ffill").fillna(0)

    # Choose estimator by problem type
    if is_regression:
        estimator = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        estimator = RandomForestClassifier(n_estimators=200, random_state=42)

    pipeline = Pipeline(steps=[("pre", preprocessor), ("est", estimator)])

    # stratify only if classification and more than one class
    stratify = y if (not is_regression and len(y.unique()) > 1) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=stratify)
    except Exception:
        # fallback split without stratify
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    pipeline.fit(X_train, y_train)

    # quick metric
    y_pred = pipeline.predict(X_test)
    if is_regression:
        metric = r2_score(y_test, y_pred)
        st.sidebar.success(f"Fallback trained — R2: {metric:.3f}")
    else:
        try:
            metric = accuracy_score(y_test, y_pred)
            st.sidebar.success(f"Fallback trained — Accuracy: {metric:.3f}")
        except Exception:
            st.sidebar.success("Fallback trained.")

    model = pipeline

# ----------------- Prediction -----------------
st.subheader("Prediction")
prediction = None
prediction_proba = None

# Try direct predict
try:
    prediction = model.predict(input_df)
except Exception:
    # Try transform then predict
    try:
        X_trans = preprocessor.transform(input_df)
        prediction = model.predict(X_trans)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

pred_value = prediction[0]
st.success(f"Predicted `{target_col}`: {pred_value}")

# Show probabilities for classification if available
if not is_regression:
    proba_shown = False
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(input_df)
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps") and "est" in model.named_steps:
                # pipeline case: classifier inside pipeline
                est = model.named_steps["est"]
                classes = getattr(est, "classes_", None)
            if classes is not None:
                proba_df = pd.DataFrame(proba, columns=[str(c) for c in classes])
                st.write("Prediction probabilities (per class):")
                st.table(proba_df.T)
                proba_shown = True
        except Exception:
            proba_shown = False
    # Try pipeline est predict_proba if available
    if (not proba_shown) and hasattr(model, "named_steps") and "est" in model.named_steps:
        est = model.named_steps["est"]
        if hasattr(est, "predict_proba"):
            try:
                X_trans = model.named_steps["pre"].transform(input_df)
                proba = est.predict_proba(X_trans)
                classes = getattr(est, "classes_", None)
                if classes is not None:
                    proba_df = pd.DataFrame(proba, columns=[str(c) for c in classes])
                    st.write("Prediction probabilities (per class):")
                    st.table(proba_df.T)
            except Exception:
                pass

# ----------------- Model details -----------------
st.write("---")
st.subheader("Model details")
st.write("Model object type:", f"`{type(model).__name__}`")
st.write("Fallback model used:", fallback_used)

# Try to show feature importances for tree-based estimators in pipeline
try:
    est = None
    if hasattr(model, "named_steps") and "est" in model.named_steps:
        est = model.named_steps["est"]
        pre = model.named_steps["pre"]
    else:
        # model might be a bare estimator
        if hasattr(model, "feature_importances_"):
            est = model
            pre = None
    if est is not None and hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
        # build output feature names: numeric then expanded categorical names if possible
        feat_names = numeric_features.copy()
        if pre is not None:
            cat_transformer = None
            try:
                cat_transformer = pre.named_transformers_.get("cat", None)
            except Exception:
                cat_transformer = None
            if cat_transformer is not None and hasattr(cat_transformer, "get_feature_names_out"):
                try:
                    cat_names = list(cat_transformer.get_feature_names_out(categorical_features))
                except Exception:
                    cat_names = categorical_features
            else:
                cat_names = categorical_features
            feat_names.extend(cat_names)
        # If lengths mismatch, fall back to non-expanded names
        if len(feat_names) != len(importances):
            feat_names = numeric_features + categorical_features
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
        st.write("Feature importances:")
        st.dataframe(imp_df)
except Exception:
    # silently ignore feature importances failures
    pass

st.write("---")
st.markdown(
    """
**Notes & troubleshooting**
- If loading `Employee_model.pkl` raises errors like `No module named 'numpy._core'`, that indicates a binary-version mismatch (commonly numpy/scipy/scikit-learn compiled extension mismatch). Two solutions:
  1. Train / export the model in the environment you will run the app in, or
  2. Use a Docker image with pinned binary versions that match the training environment.
- The app trains a fallback RandomForest if the provided pickle can't be loaded; this keeps the UI usable for demos.
- To make the OneHotEncoder robust across sklearn versions, the code tries the `sparse_output` argument first and falls back to `sparse`.
"""
)
