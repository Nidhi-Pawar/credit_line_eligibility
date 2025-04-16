from .base_trainer import Model
from xgboost import XGBClassifier
from optuna_integration.xgboost import XGBoostPruningCallback
import logging 

class XGBoost(Model):
    """
    XGBoost Forest model

    """
    def train(self, X_train, y_train,**kwargs):
        """
        This train function is used to train the XGBClassifier model to find best parameters during optuna trails on sampled data, 
        and then to train the classifier with the best parameters found on entire data

        Args:
        X_train (array-like): Training data
        y_train (array-like): Training labels
        **kwargs: Best parameters

        Returns:
        XGBClassifier: Model trained on best parameters provided by optuna trials
        
        """        
        try:
            xgbmodel = XGBClassifier(**kwargs, eval_metric='logloss', scale_pos_weight=0.24, objective="binary:logistic", tree_method='gpu_hist')
            xgb = xgbmodel.fit(X_train, y_train)
            logging.info("XGBoost model trained successfully")
            return xgb
        except Exception as e:
            logging.error(f"Error training XGBoost model: {e}")
            return e
        
    def extratrain(self, X_train, y_train, **kwargs):
        """
        XGBoostClassifier model to run optuna trials

        Returns:
        model: trained model

        """
        try:
            xgbmodel = XGBClassifier(**kwargs, eval_metric='logloss', scale_pos_weight=0.24, objective="binary:logistic", tree_method='gpu_hist')
            xgb = xgbmodel.fit(X_train, y_train)
            # logging.info("XGBoost model trained successfully")
            return xgb
        except Exception as e:
            logging.error(f"Error training XGBoost model: {e}")
            return e
        
    
    def optimize(self, trial, X_train, y_train):
        """
        Defines parameters for optuna trials on XGBClassifier model

        Args:
        trial (optuna.trial.Trial): Optuna trial object
        X_train (array): Training data
        y_train (array): Training labels

        Returns:
        dict: XGBClassifier model parameters
        score: Best score for the model
        
        """
        X_train, X_valid, y_train, y_valid = super().optimize(trial, X_train, y_train)
        params = {
            # "objective": "binary:logistic",
            # "tree_method": "gpu_hist",  
            "verbosity":0,
            "verbose":-1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-3, 1.0),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0),
            # "early_stopping_rounds":20,
            # "callbacks":[XGBoostPruningCallback(trial, "validation_0-logloss")],
    }
        
        # logging.info(f"XGBoost parameters: {params}")
        model = self.train(X_train, y_train, **params) #, evals=[(X_valid, "validation")]
        return model.score(X_valid, y_valid)