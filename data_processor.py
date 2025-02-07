# Done by Amrit Bag

# Dependencies 
import os 
from src.data_cleaner import data_cleaner
from src.data_preprocesser.text_preprocessing import TextPreprocessing
#from src.data_preprocesser.image_preprocessing import *
#from src.data_curator import *
from config.config import Config
from src.utils.download_from_drive import DownloadData



downloader = DownloadData(client_secrets_path = Config.CLIENT_SECRET_CREDENTIALS)
downloader.download_google_drive_folder(Config.INSTAGRAM_RAW_IMAGE_DATA_LINK, Config.INSTAGRAM_RAW_IMAGE_DATA_PATH)
downloader.download_google_drive_folder(Config.LINKEDIN_RAW_IMAGE_DATA_LINK,  Config.LINKEDIN_RAW_IMAGE_DATA_PATH)
downloader.download_google_drive_folder(Config.FACEBOOK_RAW_IMAGE_DATA_LINK,  Config.FACEBOOK_RAW_IMAGE_DATA_PATH)