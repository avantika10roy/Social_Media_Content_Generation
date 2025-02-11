# ------------------------Done by Amrit Bag---------------------------#

# Dependencies
import pandas as pd
from config.config import Config
from src.utils.logger import LoggerSetup          
from src.utils.data_saver import DataSaver
from src.data_cleaner.data_cleaner import DataCleaner
from src.data_cleaner.image_cleaner import ImageCleaner
from src.data_curator.data_curation import DataCuration
from src.data_preprocesser.text_preprocessing import TextPreprocessing
from src.data_preprocesser.image_preprocessing import ImagePreprocessor

"""
# To download Data From Drive run this lines
downloader = DownloadData(client_secrets_path = Config.CLIENT_SECRET_CREDENTIALS)
downloader.download_google_drive_folder(Config.INSTAGRAM_RAW_IMAGE_DATA_LINK, Config.INSTAGRAM_RAW_IMAGE_DATA_PATH)
downloader.download_google_drive_folder(Config.FACEBOOK_RAW_IMAGE_DATA_LINK, Config.FACEBOOK_RAW_IMAGE_DATA_PATH)
downloader.download_google_drive_folder(Config.LINKEDIN_RAW_IMAGE_DATA_LINK, Config.LINKEDIN_RAW_IMAGE_DATA_PATH)
"""

# Set up logger
logger = LoggerSetup(logger_name="run.py", log_filename_prefix="RunScript").get_logger()

def main():
    try:
        #   Data Cleaning
        # -----------------
        logger.info("Starting data cleaning process...")
        cleaner                = DataCleaner()
        imageclean             = ImageCleaner()

        # Setting The raw data Path
        raw_facebook_data      = Config.FACEBOOK_RAW_POST_DATA_PATH
        raw_instagram_data     = Config.INSTAGRAM_RAW_POST_DATA_PATH
        raw_linkdin_data       = Config.LINKEDIN_RAW_POST_DATA_PATH

        # Raw json data converted into dataframe
        raw_facebook_data_df   = pd.read_json(path_or_buf = raw_facebook_data)  
        raw_instagram_data_df  = pd.read_json(path_or_buf = raw_instagram_data)
        raw_linkdin_data_df    = pd.read_json(path_or_buf = raw_linkdin_data)
       
        # Setting Cleaned Data path
        cleaned_facebook_data  = Config.FACEBOOK_CLEANED_POST_DATA_PATH
        cleaned_instagram_data = Config.INSTAGRAM_CLEANED_POST_DATA_PATH
        cleaned_linkedin_data  = Config.LINKEDIN_CLEANED_POST_DATA_PATH
        

        # Data Cleaning
        
        # Facebook Cleaning
        
        # Text Cleaning
        cleaned_facebook       = (cleaner.facebook_cleaner(facebook_data = raw_facebook_data_df ,
                                                            platform     ='facebook'))
        
        DataSaver.data_saver(data             = cleaned_facebook,
                             file_path        = cleaned_facebook_data)
        
        # Image cleaning
        
        fb_data                 = DataSaver.data_reader(cleaned_facebook_data)
        cleaned_fb_image        = imageclean.filter_and_copy_images(data     = fb_data,
                                                                    platform = 'facebook')
        DataSaver.data_saver(data             = cleaned_fb_image,
                             file_path        = cleaned_facebook_data)

        # Instagram Cleaning
        
        # Text Cleaning
        cleaned_instagram       = (cleaner.instagram_clean_data(insta_data = raw_instagram_data_df,
                                                               platform    ='instagram'))
        DataSaver.data_saver(data      = cleaned_instagram,
                             file_path = cleaned_instagram_data)
        
        # Image cleaning

        insta_data              = DataSaver.data_reader(cleaned_instagram_data)
        cleaned_instagram_image = imageclean.filter_and_copy_images(data     = insta_data,
                                                                    platform = 'instagram')
        DataSaver.data_saver(cleaned_instagram_image,cleaned_instagram_data)
        
        # Linkedin Cleaning
        
        # Text Cleaning
        
        cleaned_linkedin       = (cleaner.linkdln_clean_data(linkdln_data  = raw_linkdin_data_df,
                                                             platform      = 'linkedin'))
        DataSaver.data_saver(data        = cleaned_linkedin,
                             file_path   = cleaned_linkedin_data)
        
        # Image Cleaning
        
        linkedin_data          = DataSaver.data_reader(cleaned_linkedin_data)
        cleaned_linkedin_image = imageclean.filter_and_copy_images(data      = linkedin_data,
                                                                   platform  = 'linkedin')
        DataSaver.data_saver(cleaned_linkedin_image,cleaned_linkedin_data)

        logger.info("Data cleaning process completed successfully.")
        
        
        # Save cleaned data
       

        #         Data Curation
        #     --------------------

        # Setting Curated Data Path
        curated_data_path        = Config.CURATED_POST_DATA_PATH

        # Text Data curation
        curation = DataCuration(linkedin_cleaned_data_path   = cleaned_linkedin_data,
                                 instagram_cleaned_data_path = cleaned_instagram_data, 
                                 facebook_cleaned_data_path  = cleaned_facebook_data)
        logger.info("Starting data curation process...")

        curated_data             = curation.text_curation()

        # Curated Data saving

        DataSaver.data_saver(curated_data,curated_data_path)

        logger.info("Data curation process completed successfully.")

        # Image Data Curation
        curation.image_curation(json_path                    = Config.CURATED_POST_DATA_PATH, 
                                curated_images_dir           = Config.CURATED_IMAGE_DATA_PATH
                                )


        #     Data Preprocessing
        #  ------------------------

        # Setting Preprocessed Data path
        
        preprocessed_data_path = Config.PREPROCESSED_DATA_PATH
        text_preprocessor      = TextPreprocessing()
        
        
        # Text Data preprocessing

        preprocessed_df        = text_preprocessor.preprocess(curated_data = curated_data, 
                                                              text_column  = 'post_content')
        DataSaver.data_saver(preprocessed_df,preprocessed_data_path)

        # Image Preprocessing

        preprocessor          =ImagePreprocessor(raw_data_path     = Config.CURATED_IMAGE_DATA_PATH ,
                                                 cleaned_data_path = Config.PREPROCESSED_IMAGE_DATA_PATH
                                                 )
        
        preprocessor.preprocess_images()
        
        
    
    
    except Exception as e:
        logger.error(f"Error in data cleaning process: {e}")

if __name__ == "__main__":
    main()