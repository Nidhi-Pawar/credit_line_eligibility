import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class to define strategies for data handling

    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame):
        pass

# class DataPreprocessStrategy(DataStrategy):

    # def handle_data(self, data) -> pd.DataFrame:
    
            
        
