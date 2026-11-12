"""Streamlit scoring app: enter applicant details, get default probability + SHAP explanation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from features import prepare

st.set_page_config(page_title="Credit Default Scorer", layout="centered")


@st.cache_resource
def load():
    b = joblib.load(Path(__file__).parent / "models" / "model.joblib")
    return b, shap.TreeExplainer(b["model"])


bundle, explainer = load()
st.title("Credit Default Risk Scorer")
st.caption("Enter applicant details. Bills and payments are entered as 6-month averages to keep the form short.")

c1, c2, c3 = st.columns(3)
limit = c1.number_input("Credit limit (NT$)", 10_000, 1_000_000, 150_000, 10_000)
age = c2.number_input("Age", 21, 79, 35)
sex = c3.selectbox("Sex", [1, 2], format_func=lambda v: {1: "Male", 2: "Female"}[v])
edu = c1.selectbox("Education", [1, 2, 3, 4], format_func=lambda v: {1: "Graduate", 2: "University", 3: "High school", 4: "Other"}[v], index=1)
mar = c2.selectbox("Marital status", [1, 2, 3], format_func=lambda v: {1: "Married", 2: "Single", 3: "Other"}[v])
avg_bill = c3.number_input("Avg monthly bill (NT$)", 0, 1_000_000, 50_000, 5_000)
avg_pay = c1.number_input("Avg monthly payment (NT$)", 0, 1_000_000, 5_000, 1_000)

st.subheader("Repayment status, last 6 months (-1 = paid on time, 1+ = months late)")
cols = st.columns(6)
pay = [col.number_input(f"M-{i + 1}", -2, 8, 0, key=f"p{i}") for i, col in enumerate(cols)]

row = {"LIMIT_BAL": limit, "SEX": sex, "EDUCATION": edu, "MARRIAGE": mar, "AGE": age}
row.update({f"PAY_{i + 1}": pay[i] for i in range(6)})
row.update({f"BILL_AMT{i}": avg_bill for i in range(1, 7)})
row.update({f"PAY_AMT{i}": avg_pay for i in range(1, 7)})
X = prepare(pd.DataFrame([row]))[bundle["features"]]

if st.button("Score applicant", type="primary"):
    pd_hat = float(bundle["model"].predict_proba(X)[0, 1])
    approve = pd_hat < bundle["threshold"]
    st.metric("Probability of default", f"{pd_hat:.1%}")
    (st.success if approve else st.error)(
        f"Recommendation: {'APPROVE' if approve else 'DECLINE'} (policy threshold {bundle['threshold']:.0%})")
    st.subheader("Why this score?")
    shap.plots.waterfall(explainer(X)[0], show=False)
    st.pyplot(plt.gcf())
