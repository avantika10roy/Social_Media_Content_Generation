import pandas as pd
from config.config import Config
from src.utils.logger import LoggerSetup
from src.utils.data_saver import DataSaver
from src.data_cleaner.data_cleaner import DataCleaner
# from src.utils.download_from_drive import DownloadData
from src.data_curator.data_curation import DataCuration
from src.data_preprocesser.text_preprocessing import TextPreprocessing
from src.data_preprocesser.image_preprocessing import ImagePreprocessor
from src.data_cleaner.image_cleaner import ImageCleaner

# Set up logger
# logger = LoggerSetup(logger_name="data_processor.py", log_filename_prefix="data_processor").get_logger()

# downloader = DownloadData(client_secrets_path = Config.CLIENT_SECRET_CREDENTIALS)
# downloader.download_google_drive_folder(Config.LINKEDIN_RAW_IMAGE_DATA_LINK, Config.LINKEDIN_RAW_IMAGE_DATA_PATH)
# downloader.download_google_drive_folder(Config.INSTAGRAM_RAW_IMAGE_DATA_LINK, Config.INSTAGRAM_RAW_IMAGE_DATA_PATH)
# downloader.download_google_drive_folder(Config.FACEBOOK_RAW_IMAGE_DATA_LINK, Config.FACEBOOK_RAW_IMAGE_DATA_PATH)

text_cleaner = DataCleaner()
image_cleaner = ImageCleaner()


linkedin_cleaned_text  = text_cleaner.linkdln_clean_data(DataSaver.data_reader(Config.LINKEDIN_RAW_POST_DATA_PATH), 'linkedin')
instagram_cleaned_text = text_cleaner.instagram_clean_data(DataSaver.data_reader(Config.INSTAGRAM_RAW_POST_DATA_PATH), 'instagram')
facebook_cleaned_text  = text_cleaner.facebook_cleaner(DataSaver.data_reader(Config.FACEBOOK_RAW_POST_DATA_PATH), 'facebook')

DataSaver.data_saver(linkedin_cleaned_text,Config.LINKEDIN_CLEANED_POST_DATA_PATH)
DataSaver.data_saver(instagram_cleaned_text,Config.INSTAGRAM_CLEANED_POST_DATA_PATH)
DataSaver.data_saver(facebook_cleaned_text,Config.FACEBOOK_CLEANED_POST_DATA_PATH)

DataSaver.data_saver(image_cleaner.filter_and_copy_images(linkedin_cleaned_text, 'linkedin'), Config.LINKEDIN_CLEANED_POST_DATA_PATH)
DataSaver.data_saver(image_cleaner.filter_and_copy_images(instagram_cleaned_text, 'instagram'), Config.INSTAGRAM_CLEANED_POST_DATA_PATH)
DataSaver.data_saver(image_cleaner.filter_and_copy_images(facebook_cleaned_text, 'facebook'), Config.FACEBOOK_CLEANED_POST_DATA_PATH)


curator = DataCuration(Config.LINKEDIN_CLEANED_POST_DATA_PATH, 
                       Config.INSTAGRAM_CLEANED_POST_DATA_PATH,
                       Config.FACEBOOK_CLEANED_POST_DATA_PATH)

curator.text_curation()

curator.image_curation(json_path          = Config.CURATED_POST_DATA_PATH, 
                       curated_images_dir = Config.CURATED_IMAGE_DATA_PATH
                       )
