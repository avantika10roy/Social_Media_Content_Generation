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
                    json.dump(data, f, indent = 4, ensure_ascii = False)
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
                dataSaver_logger.info(f"Data loaded successfully from {file_path}")
                return pd.DataFrame(data)

            dataSaver_logger.error("Invalid JSON format. Expected a list.")
            return pd.DataFrame(None)

        except json.JSONDecodeError:
            dataSaver_logger.error(f"Invalid JSON format in {file_path}. Cannot parse.")
            return None

        except Exception as e:
            dataSaver_logger.error(f"Error loading file {file_path}: {repr(e)}")
            return None
