"""
Model evaluation module implementing comprehensive metrics, comparisons, and publication-ready visualizations.
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from src.config import FIGURES_DIR, METRICS_DIR

def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes standard classification evaluation metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

def compare_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_prefix: str = "experiment_1"
) -> pd.DataFrame:
    """
    Evaluates multiple trained pipelines on test set and returns a comparison DataFrame.
    """
    records = []
    for name, item in models_dict.items():
        pipe = item["pipeline"] if isinstance(item, dict) else item
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["Model"] = name
        records.append(metrics)

    df_results = pd.DataFrame(records)
    # Reorder columns
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP"]
    df_results = df_results[[c for c in cols if c in df_results.columns]].sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)

    os.makedirs(METRICS_DIR, exist_ok=True)
    out_csv = os.path.join(METRICS_DIR, f"{save_prefix}_model_comparison.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\n[Evaluation] Model Comparison Results:\n{df_results.to_string(index=False)}")
    print(f"[Evaluation] Saved metrics table to: {out_csv}")
    return df_results

def plot_evaluation_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_name: str = "roc_pr_curves.png"
):
    """
    Generates ROC and Precision-Recall curves comparing all models.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, item in models_dict.items():
        pipe = item["pipeline"] if isinstance(item, dict) else item
        y_prob = pipe.predict_proba(X_test)[:, 1]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax2.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    # Baseline lines
    ax1.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    baseline_pr = y_test.mean()
    ax2.plot([0, 1], [baseline_pr, baseline_pr], color="grey", lw=1.5, linestyle="--", label=f"Random Baseline ({baseline_pr:.2%})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved ROC & PR Curves figure to: {out_path}")

def plot_confusion_matrices(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_name: str = "confusion_matrices.png"
):
    """
    Plots confusion matrices side by side.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, item) in zip(axes, models_dict.items()):
        pipe = item["pipeline"] if isinstance(item, dict) else item
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No Purchase (0)", "Purchase (1)"],
                    yticklabels=["No Purchase (0)", "Purchase (1)"])
        ax.set_title(f"Confusion Matrix: {name}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Confusion Matrices figure to: {out_path}")
