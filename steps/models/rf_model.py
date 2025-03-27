from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1 
import optuna 
from .base_trainer import BaseTrainer
import yaml 


def load_config(path=r"D:\Documents\GitHub\credit_line_eligibility\steps\config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

class RandomForestModel(BaseTrainer):
    def __init__(self, model_name, config):
        super().__init__(model_name, config)

    def define_model(self, trial)-> dict:
        """
        Define the model and its hyperparameters
        Args:
            trial: optuna.trial object
        Returns:
            params: dict containing hyperparameters for the model
        """
        params = {
            "n_estimators" : trial.suggest_int("n_estimators", 50, 450, step=50),
            "max_depth" : trial.suggest_int("max_depth", 3, 15),
            "min_samples_split" : trial.suggest_int("min_samples_split", 2, 10),
            "max_features" : trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state" : 42
            }
        return params
    
    def objective(self, trial, X_train, y_train, X_valid, y_valid) -> float:
        """
        Run the optimization for the hyperparameters
        Args:
            trial: optuna.trial object
            X_train: training data
            y_train: target data
            X_valid: validation data
            y_valid: validation target data

        Returns:
            f1_score: float

        """

        X_train, X_valid, y_train, y_valid = super().objective(trial, X_train, y_train)
        params = self.define_model(trial)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_valid)[:, 1]
        f1_score = f1(y_valid, preds)
        return f1_score

    def train(self, X_train, y_train, X_valid, y_valid) -> RandomForestClassifier:
        """
        Train the model with the best hyperparameters
        Args:
            X_train: training data
            y_train: target data
            X_valid: validation data
            y_valid: validation target data
        Returns:
            model: trained model
        """
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_valid, y_valid), n_trials=30)
        best_params = study.best_params
        self.model = RandomForestClassifier(**best_params)
        self.model.fit(X_train, y_train)
        return self.model 

    



        
