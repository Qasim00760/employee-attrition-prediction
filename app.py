"""Streamlit app: employee attrition risk with explanations.

Run locally:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src.analysis import (
    SATISFACTION,
    attrition_by,
    categorical_drivers,
    income_test,
    load,
    plot_attrition_by_role,
    plot_attrition_overview,
    plot_income,
    plot_satisfaction,
    quality_report,
)
from src.model import FEATURES, coefficients, explain, fit_final, plot_coefficients, plot_explanation

# Stratified 5-fold cross-validated results from the notebook (notebooks/employee_attrition.ipynb).
RESULTS = pd.DataFrame(
    [
        ["Always predict 'stays'", 0.500, 0.161, 0.000, 0.000, 0.839],
        ["Logistic Regression", 0.825, 0.586, 0.738, 0.368, 0.753],
        ["Random Forest", 0.808, 0.543, 0.511, 0.531, 0.848],
        ["XGBoost", 0.815, 0.583, 0.595, 0.515, 0.844],
    ],
    columns=["model", "ROC-AUC", "PR-AUC", "recall", "precision", "accuracy"],
)
THRESHOLDS = pd.DataFrame(
    [[0.3, 736, 207, 30, 0.873, 0.281], [0.4, 592, 190, 47, 0.802, 0.321], [0.5, 476, 175, 62, 0.738, 0.368],
     [0.6, 341, 150, 87, 0.633, 0.440], [0.7, 238, 128, 109, 0.540, 0.538]],
    columns=["threshold", "employees flagged", "leavers caught", "leavers missed", "recall", "precision"],
)

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")


@st.cache_data
def get_data():
    return load()


@st.cache_resource
def get_model(_df):
    return fit_final(_df)


df = get_data()
model = get_model(df)

st.title("Employee Attrition Prediction")
st.write(
    "Which employees are likely to leave, and why. Analysis of the IBM HR Analytics dataset (1,470 employees) "
    "and a Logistic Regression model that scores attrition risk and explains each score."
)

tab_score, tab_drivers, tab_model, tab_data = st.tabs(["Score an employee", "What drives attrition", "Model", "Data"])

# ---------------------------------------------------------------- scoring
with tab_score:
    st.subheader("Attrition risk for one employee")
    mode = st.radio("Start from", ["An employee in the dataset", "My own inputs"], horizontal=True)
    if mode == "An employee in the dataset":
        emp = st.selectbox("Employee number", df["EmployeeNumber"].tolist(),
                           format_func=lambda n: f"{n}  ({df.loc[df['EmployeeNumber'] == n, 'JobRole'].iloc[0]})")
        row = df[df["EmployeeNumber"] == emp].iloc[[0]].copy()
        st.caption(f"In the data this employee {'left' if row['left'].iloc[0] == 1 else 'stayed'}.")
    else:
        base = df.iloc[[0]].copy()
        c1, c2, c3, c4 = st.columns(4)
        base["Age"] = c1.slider("Age", 18, 60, 35)
        base["MonthlyIncome"] = c1.number_input("Monthly income ($)", 1000, 20000, 5000, step=100)
        base["JobRole"] = c1.selectbox("Job role", sorted(df["JobRole"].unique()), index=6)
        base["JobLevel"] = c1.select_slider("Job level", [1, 2, 3, 4, 5], value=2)
        base["OverTime"] = c2.radio("Works overtime", ["No", "Yes"], horizontal=True)
        base["MaritalStatus"] = c2.selectbox("Marital status", ["Married", "Single", "Divorced"])
        base["BusinessTravel"] = c2.selectbox("Business travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        base["DistanceFromHome"] = c2.slider("Distance from home (km)", 1, 29, 10)
        base["YearsAtCompany"] = c3.slider("Years at company", 0, 40, 5)
        base["YearsInCurrentRole"] = c3.slider("Years in current role", 0, 18, 3)
        base["YearsWithCurrManager"] = c3.slider("Years with current manager", 0, 17, 3)
        base["YearsSinceLastPromotion"] = c3.slider("Years since last promotion", 0, 15, 1)
        base["TotalWorkingYears"] = c3.slider("Total working years", 0, 40, 10)
        base["NumCompaniesWorked"] = c3.slider("Companies worked at", 0, 9, 2)
        for col, label in [("JobSatisfaction", "Job satisfaction"), ("EnvironmentSatisfaction", "Environment satisfaction"),
                           ("WorkLifeBalance", "Work-life balance"), ("JobInvolvement", "Job involvement")]:
            base[col] = c4.select_slider(label, [1, 2, 3, 4], value=3, format_func=lambda v: f"{v} {SATISFACTION[v]}")
        base["StockOptionLevel"] = c4.select_slider("Stock option level", [0, 1, 2, 3], value=1)
        base["Department"] = "Sales" if "Sales" in base["JobRole"].iloc[0] else (
            "Human Resources" if base["JobRole"].iloc[0] == "Human Resources" else "Research & Development")
        row = base
        st.caption("Inputs not shown here (daily, hourly and monthly rates, training, education and others) "
                   "keep the values of a sample employee.")

    risk = float(model.predict_proba(row[FEATURES])[0, 1])
    level = "High" if risk >= 0.6 else ("Medium" if risk >= 0.4 else "Low")
    m1, m2 = st.columns([1, 2])
    m1.metric("Attrition risk score", f"{risk * 100:.0f}%")
    m1.metric("Risk level", level)
    m1.caption("The model is trained with balanced class weights, so the score ranks risk rather than giving a "
               "calibrated probability. Scores of 50% or more are flagged.")
    m2.pyplot(plot_explanation(explain(model, row, df)))
    st.caption("Bars show how much each input moves this employee's score compared with the average employee. "
               "Red raises the risk of leaving, blue lowers it.")

# ---------------------------------------------------------------- drivers
with tab_drivers:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Employees", f"{len(df):,}")
    k2.metric("Left", f"{df['left'].sum()} ({df['left'].mean() * 100:.1f}%)")
    k3.metric("Attrition with overtime", f"{attrition_by(df, 'OverTime').set_index('OverTime').loc['Yes', 'attrition_%']}%")
    k4.metric("Attrition without overtime", f"{attrition_by(df, 'OverTime').set_index('OverTime').loc['No', 'attrition_%']}%")
    st.pyplot(plot_attrition_overview(df))
    left, right = st.columns(2)
    left.pyplot(plot_attrition_by_role(df))
    right.pyplot(plot_satisfaction(df))
    left, right = st.columns([2, 3])
    left.pyplot(plot_income(df))
    t = income_test(df)
    left.caption(f"Median monthly income: ${t['median_left']:,.0f} for leavers against ${t['median_stayed']:,.0f} "
                 f"for those who stayed (Mann-Whitney p = {t['mann_whitney_p']:.1e}).")
    right.markdown("**Largest gaps in attrition rate by group, with chi-square p-values**")
    right.dataframe(categorical_drivers(df), hide_index=True, width="stretch")

# ---------------------------------------------------------------- model
with tab_model:
    st.markdown(
        "Only 16% of employees left, so a model that always answers 'stays' is 84% accurate and useless. "
        "Models are compared on ROC-AUC, PR-AUC, recall and precision from stratified 5-fold cross-validation."
    )
    st.dataframe(RESULTS, hide_index=True, width="stretch")
    st.markdown(
        "Logistic Regression has the best ROC-AUC and PR-AUC, and each of its scores can be broken down exactly "
        "into the contribution of each input, so it is the final model. XGBoost is close behind."
    )
    st.markdown("**Choosing the threshold** (Logistic Regression, all 1,470 employees, out-of-fold)")
    st.dataframe(THRESHOLDS, hide_index=True, width="stretch")
    st.caption("A lower threshold catches more leavers but flags more people who would have stayed.")
    st.pyplot(plot_coefficients(coefficients(model)))
    st.caption(
        "Coefficients are measured with all other inputs held fixed. Tenure columns are closely related to each "
        "other, so a single tenure coefficient can point the opposite way from the simple attrition rate."
    )

# ---------------------------------------------------------------- data
with tab_data:
    st.dataframe(quality_report(df).astype(str).rename("value"), width="stretch")
    st.markdown(
        "- Source: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/"
        "ibm-hr-analytics-attrition-dataset) on Kaggle, a fictional dataset created by IBM data scientists.\n"
        "- `EmployeeCount`, `StandardHours` and `Over18` hold one value each and are dropped.\n"
        "- `EmployeeNumber` is an ID and is not used as an input."
    )
    st.dataframe(df.drop(columns=["left"]), hide_index=True, width="stretch")
