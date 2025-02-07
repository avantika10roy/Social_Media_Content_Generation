## IT WILL CLEAN THE RAW DATA AND SAVE IT TO data/cleaned_data/ and withing their corresponding sub-folders

import os
import pandas as pd
from config.config import Config
from src.data_cleaner.data_cleaner import DataCleaner
from src.utils.data_saver import DataSaver
from src.utils.logger import LoggerSetup 

# Logger Setup
data_cleaning_logger = LoggerSetup(logger_name="data_cleaning.py", log_filename_prefix="DataCleaning").get_logger()

class DataCleaning:
    """
    A class for loading, cleaning, and saving raw data.
    """
    
    def __init__(self):
        """Initializes the DataCleaning class."""
        self.data_cleaner = DataCleaner()
        #base_dir = "/Users/soumik/Movies/social_media_content_generation"
        self.saver = DataSaver()
        self.cleaned_data_dir = "data/cleaned_data/"
        
        # Ensure directories exist
        os.makedirs(self.cleaned_data_dir, exist_ok=True)

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw data from a JSON file.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Raw data not found in {file_path}")
            
            raw_data = self.saver.data_reader(file_path)
            #raw_data = self.saver.read_data(file_path)
            
            if not isinstance(raw_data, list):
                raise ValueError("Invalid JSON format. Expected a list of records.")
            
            return pd.DataFrame(raw_data)
        except Exception as e:
            data_cleaning_logger.error(f"Error loading file: {e}")
            return pd.DataFrame()

    def save_clean_data(self, cleaned_data: pd.DataFrame, file_path: str):
        """
        Save cleaned data to a JSON file.
        """
        try:
            self.saver.data_saver(cleaned_data, file_path)
            #self.saver.save_data(cleaned_data,file_path)
            data_cleaning_logger.info(f"Cleaned data saved to {file_path}")
        except Exception as e:
            data_cleaning_logger.error(f"Error saving cleaned data: {e}")

    def clean_save_data(self, file_path: str, platform: str, output_file_path: str):
        """
        Load, clean, and save raw data.
        """
        try:
            raw_data = self.load_raw_data(file_path)
            if raw_data.empty:
                data_cleaning_logger.warning("Raw data is empty, skipping cleaning.")
                return
            
            if platform.lower() == "instagram":
                cleaned_data = self.data_cleaner.instagram_clean_data(raw_data, platform)
            elif platform.lower() == "linkedin":
                cleaned_data = self.data_cleaner.linkdln_clean_data(raw_data, platform)
            elif platform.lower() == "facebook":
                cleaned_data = self.data_cleaner.facebook_cleaner(raw_data,platform)
            else:
                data_cleaning_logger.warning(f"Unsupported platform: {platform}")
                return
            
            self.save_clean_data(cleaned_data, output_file_path)
        except Exception as e:
            data_cleaning_logger.error(f"Error in cleaning and saving data: {e}")

# Example Usage
if __name__ == "__main__":
    data_cleaner = DataCleaning()
    
    #raw_file_path = Config.LINKEDIN_RAW_POST_DATA_PATH
    #cleaned_file_path = Config.LINKEDIN_CLEANED_POST_DATA_PATH

    #raw_file_path = Config.INSTAGRAM_RAW_POST_DATA_PATH
    #cleaned_file_path = Config.INSTAGRAM_CLEANED_POST_DATA_PATH
    
    raw_file_path = Config.FACEBOOK_RAW_POST_DATA_PATH
    cleaned_file_path = Config.FACEBOOK_CLEANED_POST_DATA_PATH

    data_cleaning_logger.info(f"Raw Data Path: {raw_file_path}")
    data_cleaning_logger.info(f"Cleaned Data Path: {cleaned_file_path}")
    data_cleaning_logger.info(f"Does Raw Data Path Exist? {os.path.exists(raw_file_path)}")
    data_cleaning_logger.info(f"Does Cleaned Data Directory Exist? {os.path.exists(os.path.dirname(cleaned_file_path))}")
    
    data_cleaner.clean_save_data(raw_file_path, "facebook", cleaned_file_path)


