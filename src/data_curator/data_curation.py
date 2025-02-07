import pandas as pd
from ..utils.logger import LoggerSetup
from ..utils.data_saver import DataSaver
from config.config import Config

# LOGGING SETUP
dataCurator_logger = LoggerSetup(logger_name="data_curation.py", log_filename_prefix="data_curator").get_logger()

class DataCuration:
    """
    This Class is used for Data Curation of the Texts from various social media platforms.
    """

    def __init__(self, linkedin_cleaned_data_path: str, instagram_cleaned_data_path: str, facebook_cleaned_data_path: str):
        """
        Initializes the DataCuration class with paths to cleaned JSON files.

        Arguments:
        ----------
            linkedin_cleaned_data_path (str)  : Path to LinkedIn cleaned data JSON file.
            facebook_cleaned_data_path (str)  : Path to Facebook cleaned data JSON file.
            instagram_cleaned_data_path (str) : Path to Instagram cleaned data JSON file.

        Raises:
        -------
            Logs an error if the JSON file path is invalid.
        """
        try:
            self.linkedin_cleaned_data_path = linkedin_cleaned_data_path
            self.facebook_cleaned_data_path = facebook_cleaned_data_path
            self.instagram_cleaned_data_path = instagram_cleaned_data_path

            dataCurator_logger.info("Data Curator Class Initialized Successfully")

        except Exception as e:
            dataCurator_logger.error(f"Error Occurred in Initializing the Class: {repr(e)}")

    def data_curation(self) -> pd.DataFrame:
        """
        Reads and merges data from LinkedIn, Instagram, and Facebook cleaned JSON files.

        Returns:
        --------
            pd.DataFrame  : A merged DataFrame containing data from all three sources.

        Raises:
        -------
            Logs an error if any file reading operation fails.
        """
        try:
            linked_cleaned_data     = DataSaver.data_reader(self.linkedin_cleaned_data_path)
            facebook_cleaned_data   = DataSaver.data_reader(self.facebook_cleaned_data_path)
            instagram_cleaned_data  = DataSaver.data_reader(self.instagram_cleaned_data_path)

            dataCurator_logger.info("Data successfully read from all sources")

            if not isinstance(linked_cleaned_data, list):
                dataCurator_logger.warning("LinkedIn data is not in list format. Converting to list.")
                linked_cleaned_data = list(linked_cleaned_data) if linked_cleaned_data else []
        
            if not isinstance(facebook_cleaned_data, list):
                dataCurator_logger.warning("Facebook data is not in list format. Converting to list.")
                facebook_cleaned_data = list(facebook_cleaned_data) if facebook_cleaned_data else []
        
            if not isinstance(instagram_cleaned_data, list):
                dataCurator_logger.warning("Instagram data is not in list format. Converting to list.")
                instagram_cleaned_data = list(instagram_cleaned_data) if instagram_cleaned_data else []

            combined_data = linked_cleaned_data + facebook_cleaned_data + instagram_cleaned_data

            if not combined_data.empty:
                DataSaver.data_saver(combined_data, Config.CURATED_POST_DATA_PATH)
                dataCurator_logger.info("Data Curation Completed Successfully")
            else:
                dataCurator_logger.warning("No valid data found to curate.")

            return combined_data

        except Exception as e:
            dataCurator_logger.error(f"Error Occurred in data_curation function: {repr(e)}")
            return pd.DataFrame()
        
       
# EXAMPLE USAGE 
# curator = DataCuration(Config.LINKEDIN_CLEANED_POST_DATA_PATH, 
#                        Config.INSTAGRAM_CLEANED_POST_DATA_PATH, 
#                        Config.FACEBOOK_CLEANED_POST_DATA_PATH
#                        )
# curator.data_curation()