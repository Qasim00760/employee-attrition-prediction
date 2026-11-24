"""Loading, attrition summaries and charts for the IBM HR dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ibm_hr_attrition.csv"
TARGET = "Attrition"
CONSTANT = ["EmployeeCount", "StandardHours", "Over18"]
ID_COLUMNS = ["EmployeeNumber"]
NAVY, BLUE, RED = "#0f2a44", "#3d6d99", "#b2182b"
SATISFACTION = {1: "Low", 2: "Medium", 3: "High", 4: "Very high"}

sns.set_theme(style="whitegrid")


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["left"] = (df[TARGET] == "Yes").astype(int)
    return df


def quality_report(df: pd.DataFrame) -> pd.Series:
    constant = [c for c in df.columns if df[c].nunique() == 1]
    return pd.Series({
        "employees": len(df),
        "columns": df.shape[1] - 1,
        "missing values": int(df.isna().sum().sum()),
        "duplicate rows": int(df.duplicated().sum()),
        "duplicate employee numbers": int(df["EmployeeNumber"].duplicated().sum()),
        "columns with a single value": ", ".join(constant),
        "attrition rate (%)": round(df["left"].mean() * 100, 1),
    })


def attrition_by(df: pd.DataFrame, col: str, bins=None, labels=None) -> pd.DataFrame:
    key = pd.cut(df[col], bins=bins, labels=labels) if bins is not None else df[col]
    g = df.groupby(key, observed=True)["left"]
    out = pd.DataFrame({"employees": g.size(), "left": g.sum(), "attrition_%": (g.mean() * 100).round(1)})
    out.index.name = col
    return out.reset_index()


def chi_square(df: pd.DataFrame, col: str) -> float:
    return chi2_contingency(pd.crosstab(df[col], df["left"]))[1]


def income_test(df: pd.DataFrame) -> dict:
    left = df.loc[df["left"] == 1, "MonthlyIncome"]
    stayed = df.loc[df["left"] == 0, "MonthlyIncome"]
    return {"median_left": left.median(), "median_stayed": stayed.median(),
            "mann_whitney_p": mannwhitneyu(left, stayed).pvalue}


def categorical_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Attrition gap and chi-square p-value for each categorical or ordinal column."""
    cols = ["OverTime", "JobRole", "MaritalStatus", "BusinessTravel", "Department", "EducationField", "Gender",
            "JobLevel", "StockOptionLevel", "JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction",
            "WorkLifeBalance", "RelationshipSatisfaction", "Education", "PerformanceRating"]
    rows = []
    for c in cols:
        rate = df.groupby(c)["left"].mean() * 100
        rows.append({"column": c, "highest_group": str(rate.idxmax()), "highest_%": round(rate.max(), 1),
                     "lowest_group": str(rate.idxmin()), "lowest_%": round(rate.min(), 1),
                     "p_value": chi_square(df, c)})
    out = pd.DataFrame(rows)
    out["gap_points"] = (out["highest_%"] - out["lowest_%"]).round(1)
    return out.sort_values("gap_points", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- charts


def _finish(ax, title):
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    return ax.figure


def _rate_bars(ax, t, col, title, overall):
    ax.bar(t[col].astype(str), t["attrition_%"], color=[RED if v > overall * 1.5 else BLUE for v in t["attrition_%"]])
    ax.axhline(overall, color="#555", linestyle="--", linewidth=1)
    for i, (v, n) in enumerate(zip(t["attrition_%"], t["employees"])):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, max(t["attrition_%"].max() * 1.2, overall * 1.5))
    ax.set_ylabel("Attrition (%)")
    ax.set_xlabel("")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    sns.despine(ax=ax)


def plot_attrition_overview(df: pd.DataFrame):
    overall = df["left"].mean() * 100
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    specs = [
        ("OverTime", None, None, "Overtime"),
        ("MaritalStatus", None, None, "Marital status"),
        ("BusinessTravel", None, None, "Business travel"),
        ("Age", [17, 25, 30, 35, 45, 60], ["18-25", "26-30", "31-35", "36-45", "46-60"], "Age"),
        ("MonthlyIncome", [0, 2500, 5000, 10000, 20000], ["<2.5k", "2.5-5k", "5-10k", "10k+"], "Monthly income ($)"),
        ("YearsAtCompany", [-1, 1, 3, 5, 10, 40], ["0-1", "2-3", "4-5", "6-10", "11+"], "Years at company"),
    ]
    for ax, (col, bins, labels, title) in zip(axes.flat, specs):
        _rate_bars(ax, attrition_by(df, col, bins, labels), col, title, overall)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(f"Attrition rate by group (dashed line: overall {overall:.1f}%)", fontsize=13,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_attrition_by_role(df: pd.DataFrame):
    t = attrition_by(df, "JobRole").sort_values("attrition_%")
    overall = df["left"].mean() * 100
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(t["JobRole"], t["attrition_%"], color=[RED if v > overall * 1.5 else BLUE for v in t["attrition_%"]])
    for i, (v, n) in enumerate(zip(t["attrition_%"], t["employees"])):
        ax.text(v + 0.5, i, f"{v:.1f}%  (n={n})", va="center", fontsize=9)
    ax.axvline(overall, color="#555", linestyle="--", linewidth=1)
    ax.set_xlim(0, t["attrition_%"].max() * 1.3)
    ax.set_xlabel("Attrition (%)")
    return _finish(ax, "Attrition by job role")


def plot_satisfaction(df: pd.DataFrame):
    cols = ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "JobInvolvement"]
    rows = []
    for c in cols:
        t = attrition_by(df, c)
        for _, r in t.iterrows():
            rows.append({"measure": c.replace("Satisfaction", " satisfaction").replace("WorkLifeBalance", "Work-life balance")
                         .replace("JobInvolvement", "Job involvement").replace("Environment ", "Environment "),
                         "level": int(r[c]), "attrition_%": r["attrition_%"]})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    sns.barplot(data=t, x="measure", y="attrition_%", hue="level", palette="RdBu", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Attrition (%)")
    ax.legend(title="Level (1 = lowest)", frameon=False, ncol=4)
    return _finish(ax, "Attrition by satisfaction and involvement level")


def plot_income(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.boxplot(data=df, x=TARGET, y="MonthlyIncome", order=["No", "Yes"], hue=TARGET,
                palette={"No": "#9ecae1", "Yes": RED}, legend=False, width=0.5, fliersize=2, ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x / 1000:g}k"))
    ax.set_xticks([0, 1], ["Stayed", "Left"])
    ax.set_xlabel("")
    ax.set_ylabel("Monthly income")
    return _finish(ax, "Monthly income: employees who left vs stayed")
