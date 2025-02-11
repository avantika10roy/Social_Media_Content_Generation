## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

import os
import json
import shutil
import pandas as pd
from ..utils.logger import LoggerSetup
from ..utils.data_saver import DataSaver
from config.config import Config

# LOGGING SETUP
dataCurator_logger = LoggerSetup(logger_name = "data_curation.py", log_filename_prefix = "data_curator").get_logger()

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
            self.linkedin_cleaned_data_path   = linkedin_cleaned_data_path
            self.facebook_cleaned_data_path   = facebook_cleaned_data_path
            self.instagram_cleaned_data_path  = instagram_cleaned_data_path

            dataCurator_logger.info("Data Curator Class Initialized Successfully")

        except Exception as e:
            dataCurator_logger.error(f"Error Occurred in Initializing the Class: {repr(e)}")

    def text_curation(self) -> pd.DataFrame:
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

            dataframes              = [df for df in 
                                       [linked_cleaned_data, 
                                        facebook_cleaned_data, 
                                        instagram_cleaned_data
                                        ] 
                                       if df is not None
                                       ]

            if not dataframes:
                dataCurator_logger.error("No valid data found. Returning an empty DataFrame.")
                
                return pd.DataFrame()

            combined_data           = pd.concat(dataframes, 
                                                ignore_index = True)

            if not combined_data.empty:
                DataSaver.data_saver(combined_data, Config.CURATED_POST_DATA_PATH)
                dataCurator_logger.info("Data Curation Completed Successfully")
            
            else:
                dataCurator_logger.warning("No valid data to save after merging.")

            return combined_data

        except Exception as e:
            dataCurator_logger.error(f"Error Occurred in data_curation function: {repr(e)}")
            
            return pd.DataFrame()
        
        
    def image_curation(self, json_path: str, curated_images_dir: str):
        """
        Copies images from their original local paths to a curated_images directory 
        and updates the JSON file with the new image paths.

        Arguments:
        ----------
            json_path (str)          : Path to the curated JSON file.
            curated_images_dir (str) : Directory where images will be copied.

        Returns:
        --------
            None
        
        """
    
        try:
            os.makedirs(curated_images_dir, exist_ok = True)

            with open(json_path, "r", encoding = "utf-8") as f:
                posts_data = json.load(f)
            f.close()

            for post in posts_data:
                if "image_paths" in post and post["image_paths"]:

                    if isinstance(post["image_paths"], str):
                        image_paths  = [img.strip() for img in post["image_paths"].split(",")]
                    else:
                        image_paths  = post["image_paths"]

                    updated_paths    = []

                    for img_path in image_paths:
                        if os.path.exists(img_path):
                            filename = os.path.basename(img_path)
                            new_path = os.path.join(curated_images_dir, filename)

                            shutil.copy(img_path, new_path)
                            updated_paths.append(new_path.replace("\\", "/"))

                            dataCurator_logger.info(f"Image curated successfully: {new_path}")

                        else:
                            dataCurator_logger.warning(f"Image not found - {img_path}")

                    post["image_paths"] = updated_paths

            with open(json_path, "w", encoding = "utf-8") as f:
                json.dump(posts_data, f, indent = 4, ensure_ascii = False)

            dataCurator_logger.info("Images copied and JSON updated successfully!")
            
            # return posts_data

        except Exception as e:
            dataCurator_logger.error(f"Error in updating image paths: {repr(e)}")

        
        
        
       
# EXAMPLE USAGE 
# curator = DataCuration(Config.LINKEDIN_CLEANED_POST_DATA_PATH, 
#                        Config.INSTAGRAM_CLEANED_POST_DATA_PATH, 
#                        Config.FACEBOOK_CLEANED_POST_DATA_PATH
#                        )
# curator.data_curation()

# curator.image_curation(json_path          = Config.CURATED_POST_DATA_PATH, 
#                        curated_images_dir = Config.CURATED_IMAGE_DATA_PATH
#                        )