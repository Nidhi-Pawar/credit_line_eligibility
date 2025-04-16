import logging 
from abc import ABC, abstractmethod

import optuna 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.core_utils import config_loader as config

from sklearn.ensemble import RandomForestClassifier 
import lightgbm  as lgbm
# import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from optuna_integration.xgboost import XGBoostPruningCallback

class Model(ABC):
    """
    Abstract class for model selection
    
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.DataFrame): Target data
        """
        pass 

    @abstractmethod 
    def optimize(self, trial, X_train, y_train):
        """
        Optimizes the model hyperparameters
        Args:
            trial: optuna.trial object
            X_train: training data
            y_train: target data
        """
        X_train_resampled, y_train_resampled = resample(X_train, y_train, 
                                                            n_samples=70000, 
                                                            stratify=y_train,
                                                            random_state=42)
            
        X_train_sample, X_valid, y_train_sample, y_valid = train_test_split(X_train_resampled, y_train_resampled,
                                                                   test_size=0.2, 
                                                                   stratify=y_train_resampled,
                                                                   random_state=42)
            
        return X_train_sample, X_valid, y_train_sample, y_valid
    

class RandomForest(Model):
    """
    Random Forest model

    """
    def train(self, X_train, y_train, **kwargs):
        try:
            rfmodel = RandomForestClassifier(**kwargs)
            rfmodel.fit(X_train, y_train)
            logging.info("Random Forest model trained successfully")
            return rfmodel 
        except Exception as e:
            logging.error(f"Error training Random Forest model: {e}")
            return e
    
    def optimize(self, trial, X_train, y_train):
        X_train, X_valid, y_train, y_valid = super().optimize(trial, X_train, y_train)
        params = {
            "n_estimators" : trial.suggest_int("n_estimators", 50, 450, step=50),
            "max_depth" : trial.suggest_int("max_depth", 3, 15),
            "min_samples_split" : trial.suggest_int("min_samples_split", 2, 10),
            "max_features" : trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state" : 42
            }
        model = self.train(X_train,y_train, **params, class_weight='balanced')
        # m = model.fit(X_train, y_train)
        return model.score(X_valid, y_valid)


class XGBoost(Model):
    """
    XGBoost Forest model

    """
    def train(self, X_train, y_train, **kwargs):
        try:
            xgbmodel = XGBClassifier(**kwargs)
            xgbmodel.fit(X_train, y_train)
            logging.info("XGBoost model trained successfully")
            return xgbmodel 
        except Exception as e:
            logging.error(f"Error training XGBoost model: {e}")
            return e
    
    def optimize(self, trial, X_train, y_train):
        X_train, X_valid, y_train, y_valid = super().optimize(trial, X_train, y_train)
        params = {
        "objective": "binary:logistic",
        "tree_method": "gpu_hist",  
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "n_estimators": 500,  
        "eval_metric": "logloss",
        # "early_stopping_rounds":20,
        # "callback": XGBoostPruningCallback(trial, "logloss"),
    }
        
        # logging.info(f"XGBoost parameters: {params}")
        model = self.train(X_train, y_train, **params) #, evals=[(X_valid, "validation")]
                           
        wrapped_model = Pipeline([("classifier", model)])
        # m = wrapped_model.fit(X_train, y_train)
        return wrapped_model.score(X_valid, y_valid)
         

class LightGBM(Model):
    """
    LightGBM model

    """
    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        try:
            train_data = lgbm.Dataset(X_train, label=y_train)
            val_data = lgbm.Dataset(X_val, label=y_val)

            lgbmodel = lgbm(valid_sets=[train_data, val_data],
                            valid_names=["train", "valid_0"],
                            callbacks=[lgbm.early_stopping(50,verbose=False), lgbm.log_evaluation(0)],
                            **kwargs)
            lgbmodel.fit(train_data)
            logging.info("LightGBM model trained successfully")
            return lgbmodel 
        except Exception as e:
            logging.error(f"Error training LightGBM model: {e}")
            return e
    
    def optimize(self, trial, X_train, y_train):
        X_train, X_valid, y_train, y_valid = super().optimize(trial, X_train, y_train)

        
        params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": 'gbdt', 
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 400, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    }

        model = self.train(X_train, y_train, X_valid, y_valid,**params)
        # preds = model.predict(X_valid)
        return model.score(X_valid, y_valid)



class HyperParameterTuner:
    """
    Hyperparameter tuning class for the model
    """
    def __init__(self, model: Model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def optimize(self, n_trials = 5):
        """
        Optimize the hyperparameters of the model
        Args:
            trial: optuna.trial object
            X_train: training data
            y_train: target data
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.X_train, self.y_train), n_trials = n_trials)
        return study.best_trial.params