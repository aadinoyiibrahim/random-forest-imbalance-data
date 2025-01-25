"""
Project: Random forest on imbalanced dataset

this fule contains the functions for plotting the results of the models

Author: Abdullahi A. Ibrahim
date: 21-01-2025
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
import seaborn as sns


def plot_roc_curves(results, X_test, y_test, figsize=(10, 6), title=None):
    """
    ROC curves here.
    """
    plt.figure(figsize=figsize)
    for name, metrics in results.items():
        y_pred_proba = metrics["Model"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics["AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="best", fontsize=20)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_cm(results, X_test, y_test, figsize=(18, 6)):

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, (name, metrics) in enumerate(results.items()):
        model = metrics["Model"]
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Class 0", "Class 1"],
            yticklabels=["Class 0", "Class 1"],
            ax=axes[i],
        )
        axes[i].set_title(f"{name}")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    plt.tight_layout()
    plt.show()


def plot_barplot(
    indices,
    train_accuracies,
    val_accuracies,
    test_accuracies,
    labels,
    bar_width=0.1,
    figsize=(12, 6),
):

    plt.figure(figsize=figsize)
    plt.bar(indices, train_accuracies, width=bar_width, label="training", alpha=0.8)
    plt.bar(
        indices + bar_width,
        val_accuracies,
        width=bar_width,
        label="validation",
        alpha=0.8,
    )
    plt.bar(
        indices + 2 * bar_width,
        test_accuracies,
        width=bar_width,
        label="test",
        alpha=0.8,
    )

    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(indices + bar_width, labels, rotation=0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_class_imbalance(income_counts):
    plt.figure(figsize=(8, 5))
    plt.bar(
        income_counts.keys(),
        income_counts.values(),
        color=["skyblue", "orange"],
        alpha=0.8,
    )
    plt.xlabel("Income", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks([0, 1], ["class 0 (<=50K)", "class 1 (>50K)"], fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
