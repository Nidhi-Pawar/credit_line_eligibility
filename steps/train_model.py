import logging 
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    """
    Trains the model on the given data.
    Args:
        df: The data to train the model on.
    """
    pass