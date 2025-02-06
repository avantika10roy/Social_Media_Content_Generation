import emoji
import re
import pandas as pd
import logging
from src.utils.logger import LoggerSetup
import os
import shutil

cleaner_logger = LoggerSetup(logger_name="data_cleaner.py", log_filename_prefix="DataCleaner").get_logger()

class DataCleaner():
    """
    A class for preprocessing text data, including extracting emojis, extracting hashtags,
    and cleaning text by removing emojis, hashtags, HTML tags, extra spaces, and standardizing case.
    """
    
    HASHTAG_PATTERN = re.compile(r"#\w+")
    HTML_TAG_PATTERN = re.compile(r'<[^>]*>')
    NEWLINE_PATTERN = re.compile(r'\n')
    EXTRA_DOTS_PATTERN = re.compile(r'\.{2,}')
    EXTRA_SPACES_PATTERN = re.compile(r'\s+')

    def __init__(self):
        """
        Initializes the DataCleaner class.
        """
        cleaner_logger.info("DataCleaner instance created.")
        
    def extract_emojis(self, text: pd.Series):
        """
        Extracts distinct emojis from each text entry in a pandas DataFrame.
        """
        try:
            cleaner_logger.info("Extracting emojis from text.")
            emoji_list = text.apply(emoji.distinct_emoji_list)
            return emoji_list
        except Exception as e:
            cleaner_logger.error(f"Error extracting emojis: {e}")
            return text

    def extract_hashtags(self, text: pd.Series):
        """
        Extracts hashtags from each text entry in a pandas DataFrame.
        """
        try:
            cleaner_logger.info("Extracting hashtags from text.")
            hashtags_list = text.apply(lambda x: self.HASHTAG_PATTERN.findall(x))
            return hashtags_list
        except Exception as e:
            cleaner_logger.error(f"Error extracting hashtags: {e}")
            return text
        
    def clean_text(self, text: pd.Series) -> pd.DataFrame:
        """
        Cleans text by performing various operations.
        """
        try:
            cleaner_logger.info("Cleaning text data.")
            removed_emoji = text.apply(lambda x: emoji.replace_emoji(x,replace=''))  # Remove emojis
            removed_hashtags = removed_emoji.str.replace(self.HASHTAG_PATTERN, '', regex=True).str.strip()  # Remove hashtags
            removed_html_tags = removed_hashtags.str.replace(self.HTML_TAG_PATTERN, '', regex=True)  # Remove HTML tags
            removed_n = removed_html_tags.str.replace(self.NEWLINE_PATTERN, '', regex=True)  # Remove '\n'
            removed_dots = removed_n.str.replace(self.EXTRA_DOTS_PATTERN, '', regex=True)  # Remove extra dots
            cleaned_text = removed_dots.str.replace(self.EXTRA_SPACES_PATTERN, ' ', regex=True)  # Remove extra spaces
            
            cleaner_logger.info("Text cleaning complete.")
            return cleaned_text
        
        except Exception as e:
            cleaner_logger.error(f"Error cleaning text: {e}")
            return text
        
    def split_at_first_delimiter(self, gen_data: pd.DataFrame) -> pd.DataFrame:
        """
        Split each text entry in a pandas DataFrame into heading and content at the first delimiter.
    
        Arguments:
        ----------
        text (pd.DataFrame) : DataFrame of text to split
        
        Returns:
        ---------
        pd.DataFrame        : DataFrame of tuples (heading, content)
        """
        def split_text(single_text: str) -> tuple:
            match = re.search(r'[.!?]', single_text)
            if match:
                split_index = match.start() + 1
                heading = single_text[:split_index].strip()
                content = single_text[split_index:].strip()
                content = re.sub(r'^[.!?\s]+', '', content)
                return heading, content
            else:
                return single_text.strip(), ""

        try:
            cleaner_logger.info("Splitting text at first delimiter for DataFrame.")
            gen_data[['post_heading','post_content']] = gen_data['post_contents'].apply(split_text).apply(pd.Series)
            return gen_data
        except Exception as e:
            cleaner_logger.error(f"Error splitting text in DataFrame: {e}")
            return gen_data
        
        
    def instagram_clean_data(self, insta_data: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        General method to clean data for specified platform.
        """
        try:
            #merged_data = insta_data[]
            insta_data['emoji'] = self.extract_emojis(insta_data['post_heading'])
            self.extract_hashtags(insta_data)
            self.split_at_first_delimiter(insta_data)
            cleaned_insta_data = self.clean_text(insta_data)
            
            cleaned_insta_data = cleaned_insta_data.drop(columns=['image_urls','post_id'], axis=1)
            cleaned_insta_data['platform'] = 'instagram'
        
            cleaner_logger.info(f"{platform} data cleaned")
            return cleaned_insta_data
        
        except Exception as e:
            cleaner_logger.error(f"Error in {platform} cleaner: {e}")

            return insta_data
        
    def linkdln_clean_data(self, linkdln_data: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        General method to clean data for specified platform.
        """
        try:
            linkdln_data['emoji'] = self.extract_emojis(linkdln_data['post_contents'])
            linkdln_data['hashtags'] = self.extract_hashtags(linkdln_data['post_contents'])
            linkdln_data['post_contents'] = self.clean_text(linkdln_data['post_contents'])
            self.split_at_first_delimiter(linkdln_data)
            
            
            linkdln_data = linkdln_data.drop(columns=['image_URLs','post_contents'], axis=1)
            linkdln_data['platform'] = 'linkedin'
            
            cleaner_logger.info(f"{platform} data cleaned")
        
            return linkdln_data
        
        except Exception as e:
            cleaner_logger.error(f"Error in {platform} cleaner: {e}")
            return linkdln_data
        

'''class ImageDataCleaning():

    def __init__(self,img_dir,clean_dir,inv_image_dir):
        cleaner_logger.info("Image datacleaner instance created")
        self.image_dir = img_dir
        self.clean_image_dir = clean_dir
        self.invalid_image_dir = inv_image_dir

        os.makedirs(self.clean_image_dir, exist_ok=True)
        os.makedirs(self.invalid_image_dir, exist_ok=True)

    def is_image_valid(self,data: pd.DataFrame):
        try:
            keywords = ['celebrated', 'celebration', 'throwback', 'celebrated', 'lunch', 'dinner','anniversary','outing','party','enjoyed']
            pattern = '|'.join(keywords)

            wrong_images = data[data['post_content'].str.contains(pattern, case=False, na=False)|| [data['post_heading'].str.contains(pattern, case=False, na=False)]

            wrong_images_path = wrong_images['image_paths'].tolist()
            
            for path in wrong_images_path:
                full_path = os.path.join(self.image_dir,path)

                if os.path.exists(full_path):
                    self._move_invalid_image(full_path)
                    cleaner_logger.info(f"Moved invalid image {full_path}")

                else:
                    cleaner_logger.warning(f"image {full_path} not found")

            valid_data = data[~data['post_content'].str.contains(pattern, case=False, na=False)]

            self._move_valid_images(valid_data)

            return valid_data
        except Exception as e:
            cleaner_logger.error(f"error during image cleaning")
            return data


    def remove_duplicates(self):
        """Remove duplicate images based on their hash values."""
        hashes = set()
        duplicate_images = []

        for image_file in os.listdir(self.image_dir):
            image_path = os.path.join(self.image_dir, image_file)

            if not self.is_image_valid(image_path):
                continue

            image_hash = self.calculate_image_hash(image_path)
            if image_hash in hashes:
                duplicate_images.append(image_path)
            else:
                hashes.add(image_hash)

        for duplicate in duplicate_images:
            print(f"Deleting duplicate image: {duplicate}")
            os.remove(duplicate)


    def _move_valid_images(self, data: pd.DataFrame) -> None:
        """
        Move valid images to the cleaned image directory.
        
        Args:
            data (pd.DataFrame): DataFrame containing valid image entries
        """
        try:
            for _, row in data.iterrows():
                image_path = row['image_urls']
                original_path = os.path.join(self.image_dir, image_path)
                
                if os.path.exists(original_path):
                    # Create subdirectories in clean_image_dir if needed
                    clean_dest = os.path.join(
                        self.clean_image_dir, 
                        os.path.dirname(image_path)
                    )
                    os.makedirs(clean_dest, exist_ok=True)
                    
                    # Move valid image
                    clean_final_path = os.path.join(
                        self.clean_image_dir, 
                        image_path
                    )
                    shutil.copy(original_path, clean_final_path)
                    cleaner_logger.info(f"Moved valid image: {image_path}")
                else:
                    cleaner_logger.warning(f"Valid image not found: {original_path}")
                    
        except Exception as e:
            cleaner_logger.error(f"Error moving valid images: {str(e)}")
            raise

    def _move_invalid_image(self, image_path: str):
        """Move the invalid image to the invalid images directory."""
        try:
            # Define the new path where the invalid image will be moved
            invalid_image_path = os.path.join(self.invalid_image_dir, os.path.basename(image_path))
            # Ensure the directory exists for the invalid image
            os.makedirs(self.invalid_image_dir, exist_ok=True)
            # Move the image
            shutil.move(image_path, invalid_image_path)
            cleaner_logger.info(f"Moved invalid image {image_path} to {invalid_image_path}")
        except Exception as e:
            cleaner_logger.error(f"Error moving invalid image {image_path}: {e}")'''
    
import os
import shutil
import pandas as pd
from PIL import Image
import hashlib
import logging

# Configure logger
cleaner_logger = logging.getLogger("ImageCleaner")
cleaner_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
cleaner_logger.addHandler(handler)

class ImageDataCleaning:
    def __init__(self, img_dir, clean_dir, inv_image_dir):
        cleaner_logger.info("Image data cleaner instance created")
        self.image_dir = img_dir
        self.clean_image_dir = clean_dir
        self.invalid_image_dir = inv_image_dir
        
        # Create directories if they don't exist
        os.makedirs(self.clean_image_dir, exist_ok=True)
        os.makedirs(self.invalid_image_dir, exist_ok=True)

    def calculate_image_hash(self, image_path):
        """Calculate the hash of an image."""
        with Image.open(image_path) as img:
            return hashlib.md5(img.tobytes()).hexdigest()

    def is_image_valid(self, data: pd.DataFrame):
        """
        Filter out images based on keywords in the post content.
        
        Args:
            data (pd.DataFrame): DataFrame containing image metadata.
        
        Returns:
            pd.DataFrame: DataFrame containing only valid images.
        """
        try:
            if 'image_urls' not in data.columns or 'post_content' not in data.columns:
                cleaner_logger.error("Required columns 'image_urls' or 'post_content' are missing in the data.")
                return data
        
            keywords = ['celebrated', 'celebration', 'throwback', 'lunch', 'dinner', 
                        'anniversary', 'outing', 'party', 'enjoyed']
            pattern = '|'.join(keywords)
            
            # Filter out invalid images based on keywords
            invalid_images = data[data['post_content'].str.contains(pattern, case=False, na=False)]
            valid_images = data[~data['post_content'].str.contains(pattern, case=False, na=False)]
            
            # Move invalid images to the invalid directory
            for _, row in invalid_images.iterrows():
                image_path = row['image_urls']
                full_path = os.path.join(self.image_dir, image_path)
                if os.path.exists(full_path):
                    self._move_invalid_image(full_path)
                    cleaner_logger.info(f"Moved invalid image {full_path}")
                else:
                    cleaner_logger.warning(f"Image {full_path} not found")
            
            return valid_images
        except Exception as e:
            cleaner_logger.error(f"Error during image validation: {e}")
            return data

    def remove_duplicates(self):
        """
        Remove duplicate images based on their hash values and move valid images to the clean directory.
        """
        hashes = set()
        duplicate_images = []
        
        # Iterate over all images in the image directory
        for image_file in os.listdir(self.image_dir):
            image_path = os.path.join(self.image_dir, image_file)
            
            # Skip non-image files or invalid paths
            if not os.path.isfile(image_path):
                continue
            
            try:
                # Calculate the hash of the image
                image_hash = self.calculate_image_hash(image_path)
                
                # Check if the hash is already in the set (i.e., duplicate)
                if image_hash in hashes:
                    duplicate_images.append(image_path)
                else:
                    hashes.add(image_hash)
            except Exception as e:
                cleaner_logger.error(f"Error calculating hash for image {image_path}: {e}")
        
        # Delete duplicate images
        for duplicate in duplicate_images:
            try:
                cleaner_logger.info(f"Deleting duplicate image: {duplicate}")
                os.remove(duplicate)
            except Exception as e:
                cleaner_logger.error(f"Error deleting duplicate image {duplicate}: {e}")

    def _move_valid_images(self, data: pd.DataFrame):
        """
        Move valid images to the cleaned image directory.
        
        Args:
            data (pd.DataFrame): DataFrame containing valid image entries.
        """
        try:
            for _, row in data.iterrows():
                image_path = row['image_urls']
                original_path = os.path.join(self.image_dir, image_path)
                
                if os.path.exists(original_path):
                    # Create subdirectories in clean_image_dir if needed
                    clean_dest = os.path.join(
                        self.clean_image_dir, 
                        os.path.dirname(image_path)
                    )
                    os.makedirs(clean_dest, exist_ok=True)
                    
                    # Move valid image
                    clean_final_path = os.path.join(
                        self.clean_image_dir, 
                        image_path
                    )
                    shutil.copy(original_path, clean_final_path)
                    cleaner_logger.info(f"Moved valid image: {image_path}")
                else:
                    cleaner_logger.warning(f"Valid image not found: {original_path}")
                    
        except Exception as e:
            cleaner_logger.error(f"Error moving valid images: {str(e)}")
            raise

    def _move_invalid_image(self, image_path: str):
        """
        Move the invalid image to the invalid images directory.
        
        Args:
            image_path (str): Path to the invalid image.
        """
        try:
            # Define the new path where the invalid image will be moved
            invalid_image_path = os.path.join(self.invalid_image_dir, os.path.basename(image_path))
            # Ensure the directory exists for the invalid image
            os.makedirs(self.invalid_image_dir, exist_ok=True)
            # Move the image
            shutil.move(image_path, invalid_image_path)
            cleaner_logger.info(f"Moved invalid image {image_path} to {invalid_image_path}")
        except Exception as e:
            cleaner_logger.error(f"Error moving invalid image {image_path}: {e}")
