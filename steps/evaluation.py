import logging 
import pandas as pd 
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluates the model on the given data.
    Args:
        df: The data to evaluate the model on.
    """
    pass 