import logging
import pandas as pd
from zenml.steps import step
from src.data_cleaning import DataCleaning, DataPreprocessStrategy, DataDivideStrategy
from src.core_utils.config_loader import load_config  

from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_test"],]:
    """
        Cleans the data and splits it into training and testing datasets
        
        Args: 
            df: raw data
        Returns:
            Split dataset with standardized values
    """
    try:
        config = load_config() 
        preprocessing_strategy = DataPreprocessStrategy(config)

        data_cleaning = DataCleaning(df, preprocessing_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, y_train, X_test, y_test = data_cleaning.handle_data()


        # Convert scaled arrays back to DataFrames
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)  
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)

        return X_train, y_train, X_test, y_test
    
    except Exception as e:
        logging.error("Error in cleaning data:{}".format(e))
        raise e
