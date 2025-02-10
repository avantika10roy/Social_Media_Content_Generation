## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from ..utils.logger import LoggerSetup

download_logger = LoggerSetup(logger_name = "download_from_drive.py", log_filename_prefix = "drive_downloader").get_logger()

class DownloadData:
    
    def __init__(self, client_secrets_path = "client_secrets.json"):
        
        """
        Initialize Google Drive authentication.
        
        Arguments:
        ----------
        client_secrets_path: Path to client_secrets.json file.
        """
        try:
            self.client_secrets_path = client_secrets_path
            self.authenticate()
        
            download_logger.info("Download Class Initialization Successful")
            
        except Exception as e:
            download_logger.error(f"Download Class Initialization Unsuccessfull - {repr(e)}")

    def authenticate(self):
        
        """Authenticate using Google Drive API with client_secrets.json."""
        
        if not os.path.exists(self.client_secrets_path):
            download_logger.warning(f"Missing client_secrets.json at: {self.client_secrets_path}")
            raise FileNotFoundError(f"Missing client_secrets.json at: {self.client_secrets_path}")

        try:
            self.gauth = GoogleAuth()
        
            self.gauth.LoadClientConfigFile(self.client_secrets_path)
            self.gauth.LocalWebserverAuth() 
        
            self.drive = GoogleDrive(self.gauth)
            
            download_logger.info("Authentication Successful")
            
        except Exception as e:
            download_logger.error(f"Authentication Unsuccessful - {repr(e)}")

    def get_folder_id(self, folder_url):
        
        """
        Extracts the folder ID from a Google Drive folder URL.

        This function takes a Google Drive folder URL and extracts the unique folder ID,
        which is used to interact with the folder via the Google Drive API.

        The URL should be in the format:
        https://drive.google.com/drive/folders/{folder_id}
        
        Arguments:
        ----------
        folder_url  : str
            The URL of the Google Drive folder from which to extract the folder ID.
        
        Returns:
        --------
        str         : The unique folder ID extracted from the given URL.

        Raises:
        -------
        ValueError: 
            If the URL does not contain the valid format for a Google Drive folder URL.
        """
        
        if "drive.google.com" in folder_url:
            download_logger.info("Folder found in Google Drive")
            return folder_url.split("/folders/")[1].split("?")[0]
        
        else:
            download_logger.warning("Invalid Google Drive folder URL")
            raise ValueError("Invalid Google Drive folder URL.")

    def download_google_drive_folder(self, folder_url, local_save_path):
        """
        Downloads a Google Drive folder and saves it to a local directory.
        
        Arguments:
        -----------
        folder_url: Google Drive folder link.
        local_save_path: Path where the folder will be saved locally.
        
        """
        
        folder_id = self.get_folder_id(folder_url)
        file_list = self.drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

        if not os.path.exists(local_save_path):
            os.makedirs(local_save_path)

        for file in file_list:
            file_path = os.path.join(local_save_path, file['title'])
            download_logger.info(f"Downloading: {file['title']}...")

            file.GetContentFile(file_path)

        download_logger.info(f"Download complete! Files saved to {local_save_path}")


# # EXAMPLE: 
# downloader = DownloadData(client_secrets_path = Config.CLIENT_SECRET_CREDENTIALS)
# downloader.download_google_drive_folder(Config.INSTAGRAM_RAW_IMAGE_DATA_LINK, Config.INSTAGRAM_RAW_IMAGE_DATA_PATH)