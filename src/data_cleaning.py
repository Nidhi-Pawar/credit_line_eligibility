import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataStrategy(ABC):
    """
    Abstract class to define strategies for data handling

    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame):
        pass

class DataPreprocessStrategy(DataStrategy):
    """
    Abstract Class defining strategies for handling data preprocessing
    
    """
    def __init__(self, config):
        self.config = config

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data = self.convert_column_values(data)
        data = self.handle_missing_values(data)
        data = self.handle_outliers(data)
        data = self.drop_columns(data)
        data = self.frequency_encoding(data)

        return data

    def convert_column_values(self, data):
            """
            Convert column values to appropriate data types

            """
            try:
                if "emp_length" in data.columns:
                    data['emp_length'] = data['emp_length'].str.extract(r'(\d+)').astype(float)
                if "term" in data.columns:
                    data['term'] = data['term'].str.extract(r'(\d+)').astype(float)
                if "loan_status" in data.columns:
                    data['loan_status'] = data['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
                logging.info('Column values converted successfully')
                return data
                
            except Exception as e:
                logging.error('Error in converting column values')
                raise e
        
    def handle_missing_values(self, data):
            """
            Handle missing values in the data

            """
            try:
                if "mort_acc" in data.columns and data["mort_acc"].isnull().sum() > 0:
                    median = data["mort_acc"].median()
                    data["mort_acc"]=data["mort_acc"].fillna(median)

                data.dropna(inplace=True)
                logging.info('Missing values handled successfully')
                return data
            
            except Exception as e:
                logging.error('Error in handling missing values')
                raise e
            
    def handle_outliers(self, data):
            """
            Handle outliers in the data
            """
            try:
                # Handling outliers using IQR method
                cols = self.config["columns"].get("iqr_cols",[])
                mask = pd.Series(True, index=data.index)

                for col in cols:
                    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
                    ll = data[col].quantile(0.25) - 1.5 * IQR
                    ul = data[col].quantile(0.75) + 1.5 * IQR
                    mask &= (data[col] >= ll) & (data[col] <= ul)
                data = data[mask]

                # Handling outliers using quantile method
                cols = self.config["columns"].get("quantile_cols",[])
                for col in cols:
                    lower = data[col].quantile(0.01)  # 1st percentile
                    upper = data[col].quantile(0.99)  # 99th percentile
                    data = data[(data[col] >= lower) & (data[col] <= upper)]
                
                # Handling outliers using PowerTransformer
                cols = self.config["columns"].get("pt_cols",[])
                pt = PowerTransformer(method='yeo-johnson')
                for col in cols:
                    data[col] = pt.fit_transform(data[[col]])
                logging.info('Outliers handled successfully')
                return data
            
            except Exception as e:
                logging.error('Error in handling outliers')
                raise e
            
    def drop_columns(self, data):
            """
            Dropping unnecessary columns from the data
            """
            try:
                data.drop(columns=self.config["columns"].get("drop_cols",[]), inplace=True)
                
                index_to_drop = data[data['home_ownership'].isin(['ANY', 'NONE', 'OTHER'])].index
                data = data.drop(index_to_drop)

                logging.info('Columns dropped successfully')
                return data
            
            except Exception as e:
                logging.error('Error in dropping columns')
                raise e
            
    def frequency_encoding(self, data):
            """
            Frequency encoding for categorical columns
            """
            try:
                cols = self.config["columns"].get("freq_cols",[])
                for col in cols:
                    freq_encoding = data[col].value_counts().to_dict()
                    data[col] = data[col].map(freq_encoding)

                logging.info('Frequency encoding done successfully')
                return data
            
            except Exception as e:
                logging.error('Error in frequency encoding')
                raise e          
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into training and testing sets.
    
    """
    def handle_data(self, data:pd.DataFrame):
        try:
            X = data.drop(columns=['loan_status'],axis=1)
            y = data['loan_status']

            # Split data before standardization
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            logging.info('Data divided successfully')

            # Standardize the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # Fit & transform on training data
            X_test_scaled = scaler.transform(X_test)  # Only transform test data (NO fitting)
            logging.info('Data standardized successfully')
 
        except Exception as e:
            logging.error('Error in data division')
            raise e
        
        return X_train_scaled, y_train, X_test_scaled, y_test
        
class DataBalancingStrategy(DataStrategy):
    """
    Strategy for balancing the data
    
    """
    def handle_data(self, X_train_scaled, y_train):
        try:
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            logging.info('Data balanced successfully')
        
        except Exception as e:
            logging.error('Error in data balancing')
            raise e 
        
        return X_train_resampled, y_train_resampled
        

class DataCleaning:
    """
    Class to clean the data using the strategy pattern

    """
    def __init__(self,data:pd.DataFrame, data_strategy:DataStrategy):
        self.data = data
        self.data_strategy = data_strategy

    def handle_data(self):
        """
        Handles data using the strategy patterns
        
        """
        try:
            return self.data_strategy.handle_data(self.data)
        
        except Exception as e:
            logging.error("Error in handling data:{}".format(e))
            raise e

        
if __name__ == "__main__":
    import pandas as pd
    from data_cleaning import DataCleaning, DataPreprocessStrategy
    from core_utils.config_loader import load_config  # Assuming you have a config loader

    # Load config
    config = load_config()

    # Load a sample CSV file (or your real one)
    df = pd.read_csv(r"D:\Documents\GitHub\credit_line_eligibility\data\credit_eligibility.csv")

    # Run preprocessing
    cleaner = DataCleaning(df, DataPreprocessStrategy(config))
    cleaned_df = cleaner.handle_data()

    # Check the output
    print("Data cleaning successful")
    print(cleaned_df.head())
    print(cleaned_df.shape)