import logging
import mlflow
import pandas as pd
from zenml.steps import step
from zenml.client import Client

from src.models.base_trainer import HyperParameterTuner
from src.models.lgbm_model import LightGBM
from src.models.catboost_model import CatBoost
from src.models.xgb_model import XGBoost
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set an experiment tracker for ML FLow
experiment_tracker = Client().active_stack.experiment_tracker

if experiment_tracker is None:
    raise ValueError("No active experiment tracker found. Please ensure your ZenML stack is configured correctly.")

@step(experiment_tracker=experiment_tracker.name)  

def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                config: ModelNameConfig) -> ClassifierMixin:
    
    """
    Train a model on the ingested data.
    
    Args:
    X_train (pd.DataFrame): The training features.
    X_test (pd.DataFrame): The testing features.
    y_train (pd.Series): The training labels.
    y_test (pd.Series): The testing labels.
    config (ModelNameConfig): The model configuration.
    """

    logging.info(f"Shapes of training data: X_train: {X_train.shape}, y_train: {y_train.shape}") 
    
    try:
        model = None
        tuner = None 

        if config.model_name == "CatBoost":
            model = CatBoost()
        elif config.model_name == "XGBoost":
            model = XGBoost()
        elif config.model_name == "LightGBM":
            model = LightGBM()
        else:
            raise ValueError(f"Model {config.model_name} not supported")

        tuner = HyperParameterTuner(model, X_train, y_train)

        if config.fine_tuning:
            best_params = tuner.optimize(n_trials=30)
            mlflow.log_params(best_params)

            trained_model = model.train(X_train, y_train, **best_params)   
            mlflow.sklearn.log_model(trained_model, artifact_path="model")
            
            
        else:
            trained_model = model.train(X_train, y_train)

            mlflow.sklearn.log_model(trained_model, artifact_path="model")
            

        return trained_model
            
    
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise e
