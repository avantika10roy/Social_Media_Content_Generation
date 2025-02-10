# ------------------------Done by Amrit Bag---------------------------

# Dependencies
import os
import json
import pandas as pd
from config.config import Config
from src.utils.logger import LoggerSetup
from src.utils.data_saver import DataSaver
from src.data_cleaner.data_cleaner import DataCleaner
from src.data_curator.data_curation import DataCuration
from src.data_preprocesser.text_preprocessing import TextPreprocessing
from src.utils.download_from_drive import DownloadData

# DOWNLOADING DATA
downloader = DownloadData(client_secrets_path = Config.CLIENT_SECRET_CREDENTIALS)
downloader.download_google_drive_folder(Config.INSTAGRAM_RAW_IMAGE_DATA_LINK, Config.INSTAGRAM_RAW_IMAGE_DATA_PATH)
downloader.download_google_drive_folder(Config.FACEBOOK_RAW_IMAGE_DATA_LINK, Config.FACEBOOK_RAW_IMAGE_DATA_PATH)
downloader.download_google_drive_folder(Config.LINKEDIN_RAW_IMAGE_DATA_LINK, Config.LINKEDIN_RAW_IMAGE_DATA_PATH)


# Set up logger
logger = LoggerSetup(logger_name="run.py", log_filename_prefix="RunScript").get_logger()

def main():
    try:
        logger.info("Starting data cleaning process...")
        cleaner = DataCleaner()

        # Load raw data
        raw_facebook_data  = Config.FACEBOOK_RAW_POST_DATA_PATH
        raw_instagram_data = Config.INSTAGRAM_RAW_POST_DATA_PATH
        raw_linkdin_data   = Config.LINKEDIN_RAW_POST_DATA_PATH
        # Raw data converted into dataframe
        raw_facebook_data_df  = pd.read_json(raw_facebook_data)  
        raw_instagram_data_df = pd.read_json(raw_instagram_data)
        raw_linkdin_data_df   = pd.read_json(raw_linkdin_data)
       
        # Cleaned Data path
        cleaned_fb_data        = Config.FACEBOOK_CLEANED_POST_DATA_PATH
        cleaned_instagram_data = Config.INSTAGRAM_CLEANED_POST_DATA_PATH
        cleaned_linkedin_data  = Config.LINKEDIN_CLEANED_POST_DATA_PATH
        

        # Data Cleaning
        cleaned_facebook  = (cleaner.facebook_cleaner(raw_facebook_data_df, 'facebook'))
        cleaned_instagram =  (cleaner.instagram_clean_data(raw_instagram_data_df,'instagram'))
        cleaned_linkedin  =  (cleaner.linkdln_clean_data(raw_linkdin_data_df,'linkedin'))


        # Save cleaned data
       
        DataSaver.data_saver(cleaned_facebook,cleaned_fb_data)
        DataSaver.data_saver(cleaned_instagram,cleaned_instagram_data)
        DataSaver.data_saver(cleaned_linkedin,cleaned_linkedin_data)
        logger.info("Data cleaning process completed successfully.")

        #         Data Curation
        #     --------------------
        # Curated Data Path
        curated_data_path = Config.CURATED_POST_DATA_PATH

        # Data curation
        curation = DataCuration(cleaned_linkedin_data, cleaned_instagram_data, cleaned_fb_data)
        logger.info("Starting data curation process...")

        curated_data = curation.text_curation()

        # Curated Data saving

        DataSaver.data_saver(curated_data,curated_data_path)

        logger.info("Data curation process completed successfully.")


        #     Data Preprocessing
        #  ------------------------

        # Preprocessed Data path
        preprocessed_data_path = Config.PREPROCESSED_DATA_PATH
        text_preprocessor      = TextPreprocessing()
        #print(type(text_preprocessor))

        preprocessed_df        = text_preprocessor.preprocess(curated_data = curated_data, 
                                                              text_column  = 'post_content')
        DataSaver.data_saver(preprocessed_df,preprocessed_data_path)
        
        
        # Data preprocessing
        
       
    
    except Exception as e:
        logger.error(f"Error in data cleaning process: {e}")

if __name__ == "__main__":
    main()