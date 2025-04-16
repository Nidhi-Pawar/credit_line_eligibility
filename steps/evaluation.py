import logging
import pandas as pd

from zenml.steps import step
from zenml.client import Client
import mlflow

from src.model_evaluation import evaluate_model, plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)


def evaluation(model:ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series):
    """
    Evaluates the model on ingested data
    
    Args:
        df (pd.DataFrame): the ingested data
    """
    try:
       if mlflow.active_run() is not None:
        mlflow.end_run()

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] 

        accuracy, precision, recall, f1, auc, report = evaluate_model(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=model
        )
        print(f"accuracy: {accuracy:.4f},\n precision_score:{precision:.4f},\n recall_score:{recall:.4f},\n f1_score:{f1:4f},\n auc_score:{auc:.4f}")
        print(f"Classification Report:\n{report}")

        if mlflow.active_run() is not None:
            mlflow.end_run()

        
    except Exception as e:
        logging.error("Error in evaluating mode: {}".format(e))
        raise e