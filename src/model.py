"""Attrition models, evaluation and per-employee explanations.

Only 16% of employees left, so a model that always answers "stays" is 84% accurate.
Models are compared on ROC-AUC, PR-AUC, recall and precision from stratified 5-fold
cross-validation instead of accuracy.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.analysis import CONSTANT, ID_COLUMNS, NAVY, BLUE, RED, TARGET

CATEGORICAL = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
NUMERIC = ["Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "HourlyRate",
           "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
           "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
           "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
           "YearsSinceLastPromotion", "YearsWithCurrManager"]
FEATURES = CATEGORICAL + NUMERIC
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def X_y(df: pd.DataFrame):
    return df[FEATURES], df["left"]


def logistic():
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
                              ("num", StandardScaler(), NUMERIC)])
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))])


def models() -> dict:
    one_hot = lambda: ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                                        remainder="passthrough")
    return {
        "Always predict 'stays'": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": logistic(),
        "Random Forest": Pipeline([("prep", one_hot()), ("model", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE))]),
        "XGBoost": Pipeline([("prep", one_hot()), ("model", XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=5.2, eval_metric="logloss", random_state=RANDOM_STATE))]),
    }


def _metrics(y, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {"ROC_AUC": roc_auc_score(y, proba), "PR_AUC": average_precision_score(y, proba),
            "recall": recall_score(y, pred, zero_division=0), "precision": precision_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred)}


def out_of_fold(model, df) -> np.ndarray:
    X, y = X_y(df)
    return cross_val_predict(model, X, y, cv=CV, method="predict_proba")[:, 1]


def compare_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y = df["left"]
    rows, probas = [], {}
    for name, model in models().items():
        p = out_of_fold(model, df)
        probas[name] = p
        rows.append({"model": name, **_metrics(y, p)})
    return pd.DataFrame(rows).round(3), probas


def threshold_table(y, proba, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        cm = confusion_matrix(y, pred)
        rows.append({"threshold": t, "flagged_employees": int(pred.sum()), "leavers_caught": int(cm[1, 1]),
                     "leavers_missed": int(cm[1, 0]), "recall": recall_score(y, pred),
                     "precision": precision_score(y, pred, zero_division=0)})
    return pd.DataFrame(rows).round(3)


def buggy_evaluation_demo(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the first version's mistake: train on raw data, test on scaled data."""
    from sklearn.model_selection import train_test_split
    X = pd.get_dummies(df.drop(columns=[TARGET, "left"] + CONSTANT), drop_first=True)
    y = df["left"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    scaled_te = StandardScaler().fit(X).transform(X_te)
    rows = []
    for label, Xt in [("Tested on the same kind of data it was trained on", X_te),
                      ("Tested on scaled data (first version's bug)", pd.DataFrame(scaled_te, columns=X.columns))]:
        p = rf.predict_proba(Xt)[:, 1]
        pred = rf.predict(Xt)
        rows.append({"evaluation": label, "accuracy": accuracy_score(y_te, pred), "ROC_AUC": roc_auc_score(y_te, p)})
    return pd.DataFrame(rows).round(3)


def fit_final(df: pd.DataFrame) -> Pipeline:
    X, y = X_y(df)
    return logistic().fit(X, y)


def coefficients(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    coef = model.named_steps["model"].coef_[0]
    out = pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).round(3).reset_index(drop=True)


def explain(model: Pipeline, row: pd.DataFrame, reference: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Each input's push on the log-odds of leaving for one employee, compared with the average employee.

    For a linear model this is exact: coefficient x (employee's value - average value), with one-hot
    columns added back up to their original field.
    """
    prep, lr = model.named_steps["prep"], model.named_steps["model"]
    dense = lambda m: m.toarray() if hasattr(m, "toarray") else m
    z = dense(prep.transform(row[FEATURES]))[0]
    z_mean = dense(prep.transform(reference[FEATURES])).mean(axis=0)
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    contrib = pd.Series((z - z_mean) * lr.coef_[0], index=names)
    field = {n: next((c for c in CATEGORICAL if n.startswith(c + "_")), n) for n in names}
    grouped = contrib.groupby(field).sum()
    out = pd.DataFrame({"input": grouped.index, "effect": grouped.values})
    out["value"] = out["input"].map(lambda c: row.iloc[0][c])
    out["direction"] = np.where(out["effect"] > 0, "raises risk", "lowers risk")
    return out.reindex(out["effect"].abs().sort_values(ascending=False).index).head(top).round(3).reset_index(drop=True)


# ---------------------------------------------------------------- charts


def plot_roc_pr(y, probas: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {"Logistic Regression": NAVY, "Random Forest": "#7f9bb5", "XGBoost": RED}
    for name, p in probas.items():
        if name not in colors:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=colors[name], linewidth=2, label=f"{name} ({roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, color=colors[name], linewidth=2, label=f"{name} ({average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1)
    axes[1].axhline(y.mean(), color="#999", linestyle="--", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)")
    axes[1].set(xlabel="Recall", ylabel="Precision")
    axes[0].set_title("ROC curve (AUC)", loc="left", fontweight="bold")
    axes[1].set_title("Precision-recall curve (average precision)", loc="left", fontweight="bold")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9, loc="lower right" if ax is axes[0] else "upper right")
        sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_confusion(y, proba, threshold=0.5):
    cm = confusion_matrix(y, (proba >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Stays", "Leaves"], yticklabels=["Stayed", "Left"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Logistic Regression, threshold {threshold}", loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_coefficients(coefs: pd.DataFrame, top: int = 15):
    c = coefs.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(c["feature"], c["coefficient"], color=[RED if v > 0 else BLUE for v in c["coefficient"]])
    ax.axvline(0, color="#555", linewidth=1)
    ax.set_xlabel("Coefficient (red raises the chance of leaving, blue lowers it)")
    return _finish(ax, f"Top {top} drivers in the Logistic Regression model")


def plot_explanation(expl: pd.DataFrame):
    e = expl.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh([f"{i} = {v}" for i, v in zip(e["input"], e["value"])], e["effect"],
            color=[RED if v > 0 else BLUE for v in e["effect"]])
    ax.axvline(0, color="#555", linewidth=1)
    ax.set_xlabel("Push on the chance of leaving (log-odds)")
    return _finish(ax, "Why this prediction")


def _finish(ax, title):
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    return ax.figure
