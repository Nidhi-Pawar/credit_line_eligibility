from abc import ABC, abstractmethod
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

class BaseTrainer(ABC):
    def __init__(self, model_name: str, config:dict):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.best_params = None

        @abstractmethod  
        def define_model(self, trial):  
            """
            Define the model and its hyperparameters
            """
            pass

        @abstractmethod
        def objective(self, trail, X_train, y_train):
            """
            Objective function for the hyperparameter optimization
            Args: 
                trail: optuna.trial object
                X_train: training data
                y_train: target data
            Returns:
            loss value

            """
            X_train_resampled, y_train_resampled = resample(X_train, y_train, 
                                                            n_samples=self.config['optuna_sample_rows'].get('n_samples'), 
                                                            stratify=y_train,
                                                            random_state=42)
            
            X_train_sample, X_valid, y_train_sample, y_valid = train_test_split(X_train_resampled, y_train_resampled,
                                                                   test_size=0.2, 
                                                                   stratify=y_train_resampled,
                                                                   random_state=42)
            
            return X_train_sample, X_valid, y_train_sample, y_valid


            pass

        @abstractmethod
        def train(self, trail, X_train, y_train, X_valid, y_valid):
            pass
