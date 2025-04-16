import logging 
import pandas as pd
from zenml import step
from pathlib import Path
import os

class IngestData:
    def __init__(self, data_path: str):
        """
        Ingest data from a given path.
        Args:
            data_path: Path to the data file.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting data from the data path.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        try:
            path = Path(self.data_path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found at {path.absolute()}")
            if not os.access(path, os.R_OK):
                raise PermissionError(f"No read permissions for file {path.absolute()}")
            return pd.read_csv(path)
        
        except pd.errors.EmptyDataError:
            logging.error("The CSV file is empty")
            raise
        except pd.errors.ParserError:
            logging.error("Error parsing CSV file - check file format")
            raise
        except Exception as e:
            logging.error(f"Error reading data file: {str(e)}")
            raise e

@step 
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path
    Args:
        data_path: Path to the data file.
    Returns:
        pd.DataFrame: The data as a pandas DataFrame.
            
    """
    
    try:
        logging.info(f"Starting data ingestion from {data_path}")
        ingest_data_step = IngestData(data_path)
        data = ingest_data_step.get_data()
        logging.info(f"Successfully ingested data with shape {data.shape}")
        return data
    
    except Exception as e:
        logging.error(f"Error ingesting data: {e}")
        raise e




        