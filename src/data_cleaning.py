import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

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
    def handle_data(self, data) -> pd.DataFrame:
        try:
            d = {'10+ years':10, '9 years':9, '8 years':8, '7 years':7, '6 years':6,
                '5 years':5, '4 years':4, '3 years':3, '2 years':2,  '1 year':1,
                '< 1 year':0 }
            data['emp_length']=data['emp_length'].replace(d)
            d = {' 36 months':36, ' 60 months':60}
            data['term']=data['term'].replace(d)

            d = {'Fully Paid':1, 'Charged Off':0}
            data['loan_status']=data['loan_status'].replace(d)

            mort_avg = data.groupby('total_acc')['mort_acc'].mean()
            def mort(total_acc, mort_acc):
                if np.isnan(mort_acc):
                    return mort_avg[total_acc].round()
                else:
                    return mort_acc
            data['mort_acc'] = data.apply(lambda x: mort(x['total_acc'],x['mort_acc']), axis=1)
            data.dropna(inplace=True)

            def iqr_limits(series, multiplier=1.5):
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_limit = Q1 - multiplier * IQR
                upper_limit = Q3 + multiplier * IQR
                return lower_limit, upper_limit
            
            # 1. IQR method
            cols = ['loan_amnt', 'int_rate', 'installment']
            mask = pd.Series(True, index=data.index)
            # Calculate the upper and lower limits
            for col in cols:
                ll, ul = iqr_limits(data[col], multiplier=1.5)
                
                # Update mask: Mark False where outliers exist in any column
                mask &= (data[col] >= ll) & (data[col] <= ul)
                outliers = data[(data[col] < ll) | (data[col] > ul)]
                
            # Apply the mask to filter out rows with outliers
            data = data[mask]

            #2. Quantile capping
            cols = ['open_acc', 'revol_util', 'total_acc']

            for col in cols:
                lower = data[col].quantile(0.01)  # 1st percentile
                upper = data[col].quantile(0.99)  # 99th percentile
                
                data = data[(data[col] >= lower) & (data[col] <= upper)]
                outliers = data[(data[col] < ll) | (data[col] > ul)]
    
            #3. Yeo-Johnson transformation
            from sklearn.preprocessing import PowerTransformer
            cols = ['annual_inc', 'revol_bal', 'dti']
            pt = PowerTransformer(method='yeo-johnson')

            for col in cols:
                data[col] = pt.fit_transform(data[[col]])
                
            #Dropping columns

            data.drop(columns=['initial_list_status', 'emp_title', 'title','earliest_cr_line', 
                            'issue_d', 'grade', 'sub_grade', 'installment','pub_rec_bankruptcies', 
                            'application_type', 'address'], inplace=True)

            # Get the indices of rows to drop
            index_to_drop = data[data['home_ownership'].isin(['ANY', 'NONE'])].index
            data = data.drop(index_to_drop)
            

            # Frequency encoding
            cols = ['purpose', 'home_ownership', 'verification_status']
            for col in cols:
                freq_encoding = data[col].value_counts().to_dict()
                data[col] = data[col].map(freq_encoding)
              
        except Exception as e:
            logging.error('Error in data preprocessing')
            raise e 
        
        return data
        

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

            # Standardize the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # Fit & transform on training data
            X_test_scaled = scaler.transform(X_test)  # Only transform test data (NO fitting)
 
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

        
