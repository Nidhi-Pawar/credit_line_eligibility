from zenml.pipelines import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluation import evaluation
from steps.config import ModelNameConfig


@pipeline(enable_cache= True)
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, y_train, X_test, y_test = clean_data(df)
    config = ModelNameConfig() 
    model = train_model(X_train, y_train, X_test, y_test, config)
    evaluation(model, X_test, y_test)

