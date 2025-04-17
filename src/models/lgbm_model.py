import lightgbm as lgbm
from lightgbm import LGBMClassifier 
from .base_trainer import Model
import logging
import warnings

class LightGBM(Model):
    """
    LightGBM model

    """
    def train (self, X_train, y_train, **kwargs):
        """
        Trains LGBMClassifier model on the data

        Args:
        X_train (array-like): Training data
        y_train (array-like): Training labels
        **kwargs: Best parameters

        Returns:
        LGBMClassifier: Model trained on best parameters provided by optuna trials
        
        """
        try:
            warnings.filterwarnings("ignore", category=UserWarning)
            lgbm_model = LGBMClassifier(**kwargs, objective= "binary", metric= "auc", scale_pos_weight=0.24, verbose=-1, verbose_eval=False)
            lgbm_model.fit(X_train, y_train)
            logging.info("LightGBM model trained successfully")
            return lgbm_model
            
        except Exception as e:
            logging.error(f"Error in LightGBM: {e}")


    def extratrain(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        LightGBM model to run optuna trials

        Returns:
        model: trained model
        
        """
        try:
            warnings.filterwarnings("ignore", category=UserWarning)
            lgbmodel = LGBMClassifier(**kwargs, objective= "binary", metric= "auc", random_state=42, verbose=-1, scale_pos_weight= 0.2334, )
            lgbmodel.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=["train", "valid_0"],
                callbacks=[
                    lgbm.early_stopping(50, verbose=False),
                    lgbm.log_evaluation(0),
                ]
                
            )
            return lgbmodel 
        except Exception as e:
            logging.error(f"Error training LightGBM model: {e}")
            return e

    
    def optimize(self, trial, X_train, y_train):
        """
        Defines parameters for optuna trials on LightGBM model

        Args:
        trial (optuna.trial.Trial): Optuna trial object
        X_train (array): Training data
        y_train (array): Training labels

        Returns:
        dict: LightGBM model parameters
        score: Best score for the model
        
        """
        X_train, X_valid, y_train, y_valid = super().optimize(trial, X_train, y_train)
        
        params = {
        "verbose_eval":False,
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

        model = self.extratrain(X_train, y_train, X_valid, y_valid, **params)
        return model.score(X_valid, y_valid)
    
