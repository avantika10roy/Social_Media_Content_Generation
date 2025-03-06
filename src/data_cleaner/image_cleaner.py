# -------- DONE BY JIT ---------

# DEPENDENCIES
import os
import shutil
import pandas as pd
from PIL import Image
from ..utils.logger import LoggerSetup

image_cleaner_log = LoggerSetup(logger_name = "ImageCleanerLog", log_filename_prefix = "ImageCleaner").get_logger()

class ImageCleaner:
    def __init__(self, keywords: list =[]):
        """
        Initialize the ImageCleaner class.
        
        Arguments:
        keywords (list): List of keywords to filter out posts. Default keywords provided.
        """
        image_cleaner_log.info("Initialized ImageCleaner")
        self.default_keywords = ['celebrated', 'celebration', 'throwback', 'lunch', 'unforgettable', 'dinner', 'anniversary', 'outing', 'enjoyed']
        self.keywords = keywords or self.default_keywords
    
    def is_gif_image(self, image_path: str) -> bool:
        """
        Check if the given image file is actually a GIF saved as a JPG or PNG
        
        Arguments:
        image_path (str): Path to the image file.
        
        Returns:
        bool: True if the image is a GIF, otherwise False.
        """
        try:
            with Image.open(image_path) as img:
                if img.format == 'GIF':
                    return True
        except Exception as e:
            image_cleaner_log.error(f"Error checking image format {image_path}: {repr(e)}")
        return False
    
    def copy_images(self, data: pd.DataFrame, destination_folder: str) -> pd.DataFrame:
        """
        Copy images from their paths to the specified destination folder and update image paths in the DataFrame.
        
        Arguments:
        data (DataFrame): Post data.
        destination_folder (str): The destination folder where images will be copied.
        
        Returns:
        DataFrame: Updated DataFrame with new image paths.
        """
        os.makedirs(destination_folder, exist_ok=True)
        
        updated_image_paths = []
        
        for _, row in data.iterrows():
            image_paths = row["image_paths"]
            new_paths = []
            
            if isinstance(image_paths, str):
                image_paths = image_paths.split(", ")
            elif not isinstance(image_paths, list):
                updated_image_paths.append([])
                continue
            
            for image_path in image_paths:
                image_path = image_path.strip()
                abs_image_path = os.path.abspath(image_path)
                
                if os.path.exists(abs_image_path) and os.path.isfile(abs_image_path):
                    if self.is_gif_image(abs_image_path):
                        image_cleaner_log.info(f"Skipping GIF images: {abs_image_path}")
                        continue
                    
                    filename = os.path.basename(abs_image_path)
                    dest_path = os.path.join(destination_folder, filename)
                    
                    try:
                        shutil.copy2(abs_image_path, dest_path)
                        new_paths.append(dest_path)
                        image_cleaner_log.info(f"Copied: {abs_image_path} -> {dest_path}")
                    except Exception as e:
                        image_cleaner_log.error(f"Error copying {abs_image_path}: {repr(e)}")
                else:
                    image_cleaner_log.warning(f"Warning: File not found - {abs_image_path}")
            
            updated_image_paths.append(new_paths)
        
        data["image_paths"] = updated_image_paths
        
        # Remove rows where image_paths is an empty list
        data = data[data["image_paths"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        
        return data
    
    def filter_and_copy_images(self, data: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        Filter images and copy only the cleaned images to the respective platform folder.
        
        Arguments:
        data (DataFrame): Data containing posts and image paths.
        platform (str): The platform name (linkedin, facebook, instagram).
        
        Returns:
        DataFrame: The cleaned dataset with updated image paths.
        """
        pattern = '|'.join(self.keywords)
        
        invalid_posts = data[(data['post_heading'].str.contains(pattern, case=False, na=False)) 
                             | (data['post_content'].str.contains(pattern, case=False, na=False))]
        
        cleaned_data = data.drop(invalid_posts.index)
        destination_folder = f"./data/cleaned_data/{platform}_cleaned_images/"
        cleaned_data = self.copy_images(cleaned_data, destination_folder)
        
        return cleaned_data


