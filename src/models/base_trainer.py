from abc import ABC, abstractmethod
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna

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
                                                            n_samples=100000, 
                                                            stratify=y_train,
                                                            random_state=42)
            
        X_train_sample, X_valid, y_train_sample, y_valid = train_test_split(X_train_resampled, y_train_resampled,
                                                                   test_size=0.3, 
                                                                   stratify=y_train_resampled,
                                                                   random_state=42)
            
        return X_train_sample, X_valid, y_train_sample, y_valid

    
    def extratrain(self, X_train, y_train, X_val, y_val):
        """
        extra method to train the model 
        """
        pass


    
class HyperParameterTuner:
    """
    Hyperparameter tuning class for the model
    """
    def __init__(self, model: Model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
    
    def optimize(self, n_trials):
        """
        Optimize the hyperparameters of the model
        Args:
            trial: optuna.trial object
            X_train: training data
            y_train: target data
        """
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.X_train, self.y_train), n_trials = n_trials)
        return study.best_trial.params