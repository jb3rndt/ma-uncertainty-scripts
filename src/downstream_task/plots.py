import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import ensemble
from sklearn.metrics import auc, confusion_matrix, log_loss, roc_curve


def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate")
    return fig


def plot_confusion_matrix(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Rain", "Rain"],
        yticklabels=["No Rain", "Rain"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.suptitle("Confusion Matrix")
    return fig


# https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
def plot_feature_importance(clf: ensemble.GradientBoostingClassifier, feature_names):
    feature_importance = clf.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(pos, feature_importance[sorted_idx], align="center")
    ax.set_yticks(pos, feature_names[sorted_idx])
    fig.suptitle("Feature Importance (MDI)")
    return fig


def plot_loss(clf: ensemble.GradientBoostingClassifier, X_test, y_test, n_estimators):
    fig, ax = plt.subplots()
    # compute test set deviance
    test_deviance = np.zeros((n_estimators,), dtype=np.float64)

    for i, y_proba in enumerate(clf.staged_predict_proba(X_test)):
        test_deviance[i] = 2 * log_loss(y_test, y_proba[:, 1])

    ax.plot(
        (np.arange(test_deviance.shape[0]) + 1)[::5],
        test_deviance[::5],
        "-",
    )
    return fig


def plot_results(results_folder: Path, downstream_tasks: Dict[str, Tuple]):
    sns.set_theme()

    if not (results_folder / "results.csv").exists():
        return

    df = pd.read_csv(results_folder / "results.csv")

    config = json.load(open(results_folder / "config.json"))
    if "dataset" not in config:
        return

    dataset = config.get("dataset")
    _, _, labels = downstream_tasks[dataset]
    legend_labels = {
        "cleaned": "Cleaned Data (Full Dataset)",
        "polluted": "Polluted Data (Full Dataset)",
        "filtered_dq": "Polluted Data Filtered by DQ",
        "filtered_dq_certainty": "Polluted Data Filtered by DQ * C",
    }

    fig, ax = plt.subplots()
    for i, (data, group) in enumerate(df.groupby("data")):
        mean_scores = group.groupby("threshold", dropna=False)["score"].mean()
        if all(mean_scores.index.isna()):
            ax.axhline(
                mean_scores.iloc[0],
                color=f"C{i}",
                linestyle="--",
                label=f"{legend_labels[str(data)]}",
            )
        else:
            mean_scores.plot(ax=ax, color=f"C{i}", label=f"{legend_labels[str(data)]}")

    ax.legend()
    ax.set_ylabel(labels["ylabel"])
    ax.set_xlabel(labels["xlabel"])
    fig.suptitle(labels["title"])
    fig.savefig(results_folder / "plot.pdf", bbox_inches="tight")
