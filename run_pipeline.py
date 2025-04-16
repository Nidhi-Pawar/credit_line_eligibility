import os 
from pathlib import Path
from zenml.client import Client
import mlflow

from pipelines.training_pipeline import train_pipeline

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":

    tracking_uri = Client().active_stack.experiment_tracker.get_tracking_uri()
    
    if tracking_uri.startswith('C:\\'):
        tracking_uri = 'file:///' + tracking_uri.replace('\\', '/')
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")

    data_path = Path(__file__).parent / "data" / "credit_eligibility.csv"
    abs_path = data_path.absolute()
    if not abs_path.exists():
        raise FileNotFoundError(f"Data file not found at {abs_path}")
    train_pipeline(data_path=str(abs_path))


# zenml experiment-tracker register mlflow_credit_tracker --flavor=mlflow
# zenml model-deployer register credit_mlflow --flavor=mlflow
# zenml stack register mlflow_credit_stack -a default -o default -d credit_mlflow -e mlflow_credit_tracker --set

# mlflow ui --backend-store-uri "__"

#zenml stack delete -y mlflow_credit3_stack