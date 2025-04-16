import logging
from abc import ABC, abstractmethod
from models.catboost_model import RandomForestModel
import xgboost as xgb
import lightgbm as lgbm 

class Model(ABC):
    """
    Abstract class for model selection
    
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class RandomForest(Model):
    """
    Random Forest model
    
    """    
    def train(self, X_train, y_train, **kwargs):
        """
        Trains a Random Forest model 

        Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Target data
        **kwargs: Additional parameters for the model
        Returns:
        None
        
        """
        try:
            rfmodel = RandomForestModel(**kwargs)
            rfmodel.fit(X_train, y_train)
            logging.info("Random Forest model trained successfully")
            return rfmodel 
        except Exception as e:
            logging.error(f"Error training Random Forest model: {e}")
            return e
        
class XGBoost(Model):
        def train(self, X_train, y_train, **kwargs):
            """
            Trains a XGBoost model 

            Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.DataFrame): Target data
            **kwargs: Additional parameters for the model
            Returns:
            None

            """
            try:
                xgbmodel = xgb.XGBClassifier(**kwargs)
                xgbmodel.fit(X_train, y_train)
                logging.info("XGBoost model trained successfully")
                return xgbmodel 
            except Exception as e:
                logging.error(f"Error training XGBoost model: {e}")
                return e
            
class LightGBM(Model):
    def train(self, X_train, y_train, **kwargs):
        """
        Trains a LightGBM model

        Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Target data
        **kwargs: Additional parameters for the model

        Returns:
        None

        """
        try:
            lgbmodel = lgbm.LGBMClassifier(**kwargs)
            lgbmodel.fit(X_train, y_train)
            logging.info("LightGBM model trained successfully")
            return lgbmodel
        except Exception as e:
            logging.error(f"Error training LightGBM model: {e}")
            return e

        