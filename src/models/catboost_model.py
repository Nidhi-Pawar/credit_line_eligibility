from catboost import CatBoostClassifier
from .base_trainer import Model
import logging 
from sklearn.base import BaseEstimator, ClassifierMixin

class SklearnCompatibleCatBoost(CatBoostClassifier, BaseEstimator, ClassifierMixin):
    """
    Since CatBoost is not compatible with sklearn's API, we create a wrapper class to make it compatible.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CatBoost(Model):

    def train(self, X_train, y_train, **kwargs):
        """
        Trains CatBoostClassifier model on the data

        Args:
        X_train (array-like): Training data
        y_train (array-like): Training labels
        **kwargs: Best parameters

        Returns:
        CatBoostClassifier: Model trained on best parameters provided by optuna trials
        
        """
        try:
            catboost_model = SklearnCompatibleCatBoost(**kwargs, eval_metric= "AUC",loss_function= "Logloss", 
                                                scale_pos_weight=0.24, verbose=0, random_seed=42)
            cbm = catboost_model.fit(X_train, y_train)
            logging.info("CatBoost Classifier model trained successfully")
            return cbm   
        except Exception as e:
            logging.error(f"Error in CatBoost: {e}")
            return e
        
    def extratrain(self, X_train, y_train, **kwargs):
        """
        CatBoostClassifier model to run optuna trials

        Returns:
        model: trained model

        """
        try:
            catboost_model = CatBoostClassifier(**kwargs, eval_metric= "AUC",loss_function= "Logloss", random_seed=42, verbose=0)
            cbm = catboost_model.fit(X_train, y_train)
            return cbm   
        except Exception as e:
            logging.error(f"Error in CatBoost: {e}")
            return e
    
    def optimize(self, trial, X_train, y_train):
        """
        Defines parameters for optuna trials on CatBoostClassifer model

        Args:
        trial (optuna.trial.Trial): Optuna trial object
        X_train (array): Training data
        y_train (array): Training labels

        Returns:
        dict: CatBoostClassifier model parameters
        score: Best score for the model
        
        """

        X_train, X_valid, y_train, y_valid = super().optimize(trial, X_train, y_train)
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        }
        model = self.extratrain(X_train,y_train, **params)   
        return model.score(X_valid, y_valid) 
    


    



        
