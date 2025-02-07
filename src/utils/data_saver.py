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
        Reads a JSON file and returns its contents as a dictionary.
        
        Arguments:
        -----------
        file_path (str)    : The path to the JSON file to be read.
        
        Returns:
        --------
        dict               : The contents of the JSON file.
        
        Raises:
        --------
        FileNotFoundError  : If the specified file does not exist.
        """
            
        try:      
            with open(file_path, 'r', encoding = 'utf-8') as f:
                dataSaver_logger.info("Data Loaded Successfully")
                return json.load(f)
            
        except Exception as e:
            dataSaver_logger.error(f"The file {file_path} does not exist. Error Occurred: {repr(e)}")
