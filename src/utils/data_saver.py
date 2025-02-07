## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES
import os
import json
import pandas as pd

from ..utils.logger import LoggerSetup

# LOGGING SETUP
dataSaver_logger = LoggerSetup(logger_name = "data_saver.py", log_filename_prefix = "data_saver").get_logger()

class DataSaver:
    """
    A class for saving raw data in JSON format.
    
    This class provides functionality to save data from a Pandas DataFrame or a list to a JSON file.
    """
    
    @staticmethod
    def data_saver(data, file_path):
        """
        Saves a Pandas DataFrame or a list to a JSON file.
        
        Arguments:
        -----------
        data (pd.DataFrame or list)  : The data to be saved.
        file_path (str)              : The path to the JSON file where data will be saved.
        
        Raises:
        --------
        ValueError                   : If the input data is neither a DataFrame nor a list.
        """
        
        try:
            
            if isinstance(data, pd.DataFrame):
                data.to_json(file_path, orient = 'records', indent = 4)
                dataSaver_logger.info("Data Saved into JSON Format")
                
            elif isinstance(data, list):
                
                with open(file_path, 'w', encoding = 'utf-8') as f:
                    json.dump(data, f, indent = 4)
                    dataSaver_logger.info("Data Saved into JSON Format")
            
            else:
                dataSaver_logger.error("Input data must be a Pandas DataFrame or a list.")
                
        except Exception as e:
            dataSaver_logger.error(f"Error Occurred in Data Saving: {repr(e)}")
            
            
    @staticmethod
    def data_reader(file_path):
        """
        Reads a JSON file and returns its contents as a DataFrame.

        Arguments:
        ----------
        file_path (str) : The path to the JSON file to be read.

        Returns:
        --------
        pd.DataFrame or None : The contents of the JSON file as a DataFrame, or None if an error occurs.
        """
        try:
            if not os.path.exists(file_path):
                dataSaver_logger.error(f"File not found: {file_path}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):  # Ensure data is a list of records
                dataSaver_logger.error(f"Data loaded successfully from {file_path}")
                return pd.DataFrame(data)

            dataSaver_logger.error("Invalid JSON format. Expected a list.")
            return pd.DataFrame(None)

        except json.JSONDecodeError:
            dataSaver_logger.error(f"Invalid JSON format in {file_path}. Cannot parse.")
            return None

        except Exception as e:
            dataSaver_logger.error(f"Error loading file {file_path}: {repr(e)}")
            return None
'''

import json
import os
import pandas as pd
from typing import Union, List, Dict
from src.utils.logger import LoggerSetup

saver_logger = LoggerSetup(logger_name="data_saver.py", log_filename_prefix="DataSaver").get_logger()

class DataSaver:
    """
    A class for handling data reading and saving operations.
    Primarily handles JSON files for social media data.
    """
    
    def __init__(self):
        """
        Initializes the DataSaver class.
        """
        saver_logger.info("DataSaver instance created.")
        
    def data_reader(self, file_path: str) -> Union[List[Dict], None]:
        """
        Read data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file to read
            
        Returns:
            Union[List[Dict], None]: List of dictionaries containing the data, or None if reading fails
        """
        try:
            if not os.path.exists(file_path):
                saver_logger.error(f"File not found: {file_path}")
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            saver_logger.info(f"Successfully read data from {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            saver_logger.error(f"Invalid JSON format in {file_path}: {str(e)}")
            return None
        except Exception as e:
            saver_logger.error(f"Error reading data from {file_path}: {str(e)}")
            return None
            
    def data_saver(self, data: pd.DataFrame, file_path: str) -> bool:
        """
        Save DataFrame to a JSON file.
        
        Args:
            data (pd.DataFrame): DataFrame to save
            file_path (str): Path where to save the JSON file
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert DataFrame to JSON and save
            data_json = data.to_dict(orient='records')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_json, f, indent=4, ensure_ascii=False)
                
            saver_logger.info(f"Successfully saved data to {file_path}")
            return True
            
        except Exception as e:
            saver_logger.error(f"Error saving data to {file_path}: {str(e)}")
            return False

'''import pandas as pd
import os
import json
import logging
from typing import Optional, Union, Dict, List
from src.utils.logger import LoggerSetup

saver_logger = LoggerSetup(logger_name="data_saver.py", log_filename_prefix="DataSaver").get_logger()

class DataSaver:
    """
    A class for handling data reading and saving operations for social media data and images.
    Supports CSV, JSON, and Excel file formats for data.
    """
    
    SUPPORTED_DATA_FORMATS = {
        'csv': {'read': pd.read_csv, 'save': 'to_csv'},
        'json': {'read': None, 'save': 'to_json'},  # Custom JSON handling
        'excel': {'read': pd.read_excel, 'save': 'to_excel'},
    }
    
    def __init__(self, base_dir: str):
        """
        Initialize the DataSaver with a base directory for all operations.
        
        Args:
            base_dir (str): Base directory for reading and saving data
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        saver_logger.info(f"DataSaver initialized with base directory: {base_dir}")

    def _read_json_file(self, file_path: str, encoding: str = 'utf-8') -> List[Dict]:
        """
        Read a JSON file and handle both single objects and arrays of objects.
        
        Args:
            file_path (str): Path to the JSON file
            encoding (str): File encoding
            
        Returns:
            List[Dict]: List of dictionaries containing the JSON data
        """
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
            # Convert single object to list if necessary
            if isinstance(data, dict):
                data = [data]
            return data

    def read_data(self, 
                  file_path: str, 
                  file_format: str = None,
                  encoding: str = 'utf-8',
                  **kwargs) -> Optional[pd.DataFrame]:
        """
        Read data from a file into a pandas DataFrame.
        
        Args:
            file_path (str): Path to the input file
            file_format (str, optional): Format of the file ('csv', 'json', 'excel')
            encoding (str): File encoding (default: 'utf-8')
            **kwargs: Additional arguments to pass to the reading function
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing the read data, or None if reading fails
        """
        try:
            full_path = os.path.join(self.base_dir, file_path)
            
            if not os.path.exists(full_path):
                saver_logger.error(f"File not found: {full_path}")
                return None
                
            if file_format is None:
                file_format = os.path.splitext(file_path)[1][1:].lower()
                
            if file_format not in self.SUPPORTED_DATA_FORMATS:
                saver_logger.error(f"Unsupported file format: {file_format}")
                return None
            
            if file_format == 'json':
                # Custom JSON handling
                json_data = self._read_json_file(full_path, encoding)
                df = pd.json_normalize(json_data)
            else:
                read_func = self.SUPPORTED_DATA_FORMATS[file_format]['read']
                df = read_func(full_path, encoding=encoding, **kwargs)
                
            saver_logger.info(f"Successfully read data from {full_path}")
            return df
            
        except Exception as e:
            saver_logger.error(f"Error reading data from {file_path}: {str(e)}")
            return None
            
    def save_data(self,
                  data: pd.DataFrame,
                  file_path: str,
                  file_format: str = None,
                  encoding: str = 'utf-8',
                  **kwargs) -> bool:
        """
        Save DataFrame to a file.
        
        Args:
            data (pd.DataFrame): DataFrame to save
            file_path (str): Path where to save the file
            file_format (str, optional): Format to save the file in ('csv', 'json', 'excel')
            encoding (str): File encoding (default: 'utf-8')
            **kwargs: Additional arguments to pass to the saving function
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            full_path = os.path.join(self.base_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            if file_format is None:
                file_format = os.path.splitext(file_path)[1][1:].lower()
                
            if file_format not in self.SUPPORTED_DATA_FORMATS:
                saver_logger.error(f"Unsupported file format: {file_format}")
                return False
            
            if file_format == 'json':
                # Handle JSON saving with proper formatting
                json_data = data.to_dict(orient='records')
                with open(full_path, 'w', encoding=encoding) as f:
                    json.dump(json_data, f, indent=4)
            else:
                save_method = getattr(data, self.SUPPORTED_DATA_FORMATS[file_format]['save'])
                save_method(full_path, encoding=encoding, **kwargs)
                
            saver_logger.info(f"Successfully saved data to {full_path}")
            return True
            
        except Exception as e:
            saver_logger.error(f"Error saving data to {file_path}: {str(e)}")
            return False'''