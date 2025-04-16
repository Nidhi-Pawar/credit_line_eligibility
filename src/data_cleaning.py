import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.core_utils import config_loader

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataStrategy(ABC):
    """
    Abstract class to define strategies for df handling

    """
    @abstractmethod
    def handle_data(self, df: pd.DataFrame):
        pass

class DataPreprocessStrategy(DataStrategy):
    """
    Abstract Class defining strategies for handling df preprocessing
    
    """
    def __init__(self, config: config_loader):
        self.config = config

    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.convert_column_values(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        df = self.drop_columns(df)
        df = self.frequency_encoding(df)

        return df

    def convert_column_values(self, df):
            """
            Converts column values to appropriate data types

            """
            try:
                if "emp_length" in df.columns:
                    df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
                if "term" in df.columns:
                    df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
                if "loan_status" in df.columns:
                    df['loan_status'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

                top_3_purposes = ['debt_consolidation', 'credit_card', 'home_improvement']
                df['purpose']=df['purpose'].apply(lambda x: x if x in top_3_purposes else 'others')

                logging.info('Column values processed successfully')
                return df
                
            except Exception as e:
                logging.error('Error in converting column values')
                raise e
        
    def handle_missing_values(self, df):
            """
            Handle missing values in the df

            """
            try:
                if "mort_acc" in df.columns and df["mort_acc"].isnull().sum() > 0:
                    median = df["mort_acc"].median()
                    df["mort_acc"]=df["mort_acc"].fillna(median)

                df.dropna(inplace=True)
                logging.info('Missing values handled successfully')
                return df
            
            except Exception as e:
                logging.error('Error in handling missing values')
                raise e
            
    def handle_outliers(self, df):
            """
            Handle outliers in the df
            """
            try:
                # Handling outliers using IQR method
                cols = self.config["columns"].get("iqr_cols",[])
                mask = pd.Series(True, index=df.index)

                for col in cols:
                    IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
                    ll = df[col].quantile(0.25) - 1.5 * IQR
                    ul = df[col].quantile(0.75) + 1.5 * IQR
                    mask &= (df[col] >= ll) & (df[col] <= ul)
                df = df[mask]

                # Handling outliers using quantile method
                cols = self.config["columns"].get("quantile_cols",[])
                for col in cols:
                    lower = df[col].quantile(0.01)  # 1st percentile
                    upper = df[col].quantile(0.99)  # 99th percentile
                    df = df[(df[col] >= lower) & (df[col] <= upper)]
                
                # Handling outliers using PowerTransformer
                cols = self.config["columns"].get("pt_cols",[])
                pt = PowerTransformer(method='yeo-johnson')
                for col in cols:
                    df[col] = pt.fit_transform(df[[col]])
                logging.info('Outliers handled successfully')
                return df
            
            except Exception as e:
                logging.error('Error in handling outliers')
                raise e
            
    def drop_columns(self, df):
            """
            Dropping unnecessary columns from the df
            """
            try:
                df.drop(columns=self.config["columns"].get("drop_cols",[]), inplace=True)
                
                index_to_drop = df[df['home_ownership'].isin(['ANY', 'NONE', 'OTHER'])].index
                df = df.drop(index_to_drop)

                logging.info('Columns dropped successfully')
                return df
            
            except Exception as e:
                logging.error('Error in dropping columns')
                raise e
            
    def frequency_encoding(self, df):
            """
            Frequency encoding for categorical columns
            """
            try:
                cols = self.config["columns"].get("freq_cols",[])
                for col in cols:
                    freq_encoding = df[col].value_counts().to_dict()
                    df[col] = df[col].map(freq_encoding)

                logging.info('Frequency encoding done successfully')
                return df
            
            except Exception as e:
                logging.error('Error in frequency encoding')
                raise e          
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing df into training and testing sets.
    
    """
    def handle_data(self, df:pd.DataFrame):
        try:
            X = df.drop(columns=['loan_status'],axis=1)
            y = df['loan_status']

            # Split df before standardization
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            logging.info('df divided successfully')

            # Standardize the df
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # Fit & transform on training df
            X_test_scaled = scaler.transform(X_test)  # Only transform test df (NO fitting)
            logging.info('df standardized successfully')
 
        except Exception as e:
            logging.error('Error in df division')
            raise e
        
        return X_train_scaled, y_train, X_test_scaled, y_test
        

class DataCleaning:
    """
    Class to clean the df using the strategy pattern

    """
    def __init__(self, df:pd.DataFrame, df_strategy:DataStrategy):
        self.df = df
        self.df_strategy = df_strategy

    def handle_data(self):
        """
        Handles df using the strategy patterns
        
        """
        try:
            return self.df_strategy.handle_data(self.df)
        
        except Exception as e:
            logging.error("Error in handling df:{}".format(e))
            raise e
        


        
# if __name__ == "__main__":
#     import pandas as pd
#     from data_cleaning import DataCleaning, DataPreprocessStrategy
#     from core_utils.config_loader import load_config 

#     config = load_config()

#     df = pd.read_csv(r"D:\Documents\GitHub\credit_line_eligibility\df\credit_eligibility.csv")
#     cleaner = DataCleaning(df, DataPreprocessStrategy(config))
#     cleaned_df = cleaner.handle_data()

#     print("df cleaning successful")
#     print(cleaned_df.head())
#     print(cleaned_df.shape)