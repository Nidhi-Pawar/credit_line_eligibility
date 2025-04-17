import logging
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import os
from steps.config import ModelNameConfig


def evaluate_model(y_true, y_pred, y_prob, model_name: str):
    """
    Evaluate the models on standard metrics

    Args: 
    y_true (list): Ground truth values
    y_pred (list): Predicted values
    y_prob (list): Predicted probabilities
    model_name (str): Name of the model being evaluated

    Returns:
    dict: Dictionary containing the evaluation metrics
    
    """
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    model_name = model_name.__class__.__name__ 
    logging.info(f"Metrics calculated for {model_name}")

    # Log metrics to MLflow
    mlflow.log_metrics({
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1_score":f1,
        "auc":auc
    })

    # Print classification report in terminal and log it as an artifact in mlflow    
    print(f"\nClassification Report for {model_name}:\n")
    report = classification_report(y_true, y_pred)

    output_dir = (r"D:\Documents\GitHub\credit_line_eligibility\mlflow_outputs")
    os.makedirs(output_dir, exist_ok=True)  
    report_filename = f"{model_name}_classification_report.txt"
    report_path = os.path.join(output_dir, report_filename)
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)


    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig2 = plot_roc_curve(fpr, tpr, model_name)
    mlflow.log_figure(fig2, "roc_curve.png")
    plt.close(fig2)

    # PR Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    fig3 = plot_pr_curve(precision_vals, recall_vals, model_name)
    mlflow.log_figure(fig3, "pr_curve.png")
    plt.close(fig3)

    return accuracy, precision, recall, f1, auc, report


def plot_roc_curve(fpr, tpr, model_name):
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC)")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    return fig


def plot_pr_curve(precision, recall, model_name):
    fig = plt.figure()
    plt.plot(recall, precision, label=f"{model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.legend()
    return fig

