#-----------------Done by Jit Nandi and Soumik Sengupta-----------------
#-----------------Refactored by Avantika Roy------------------

# DEPENDENCIES
import os
import sys
import time
import json
import random
import instaloader
from instaloader.exceptions import ConnectionException, ProfileNotExistsException

# Add the parent directory to the system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import LoggerSetup
from config.config import Config

instagram_logger = LoggerSetup(logger_name = "instagram_scraper", log_filename_prefix = "InstagramDataScraper").get_logger()

cf = Config()

class InstagramDataScraper:
    """
    A class to scrape Instagram posts for a given user, download images, extract metadata,
    and save post details in a JSON file.
    """
    def __init__(self, user: str = cf.INSTAGRAM_USERNAME, image_dir: str = cf.INSTAGRAM_RAW_IMAGE_DATA_PATH, data_path: str = cf.INSTAGRAM_RAW_POST_DATA_PATH):
        """
            Initializes the scraper with a username, directory for images, and data file path.

            Arguments:
            ----------
                - user      : Username to scrape data.
                - image_dir : Path to directory where images will be stored.
                - data_path : Path to file where raw JSON data is to be stored.
        """
        self.user = user                          # Stores Instagram username.
        self.image_dir = image_dir                # Directory to store downloaded images.
        self.data_path = data_path                # JSON file path
        self.loader = instaloader.Instaloader()   # Instaloader instance used for scraping

        instagram_logger.info(f"Initialized InstagramDataScrapper for user: {self.user}")

    def scrape(self):
        """
        Scrapes Instagram posts for a given user, downloads images, and saves metadata.
        """
        try:
            instagram_logger.info(f"Starting scraping for user: {self.user}")
            
            profile = self.get_profile()
            if not profile:
                return

            # Ensure the directory exists for saving images, create if it does not exist
            os.makedirs(self.image_dir, exist_ok=True)
            
            # Process posts and download images
            post_data = self.process_posts(profile)

            # Save metadata to JSON
            self.save_metadata(post_data)

        except ProfileNotExistsException:
            instagram_logger.error(f"Error: Profile '{self.user}' does not exist.")
        
        except ConnectionException:
            instagram_logger.error("Error: Unable to connect to Instagram. Check your internet connection.")
        
        except Exception as e:
            instagram_logger.error(f"An unexpected error occurred: {str(e)}")

    def get_profile(self):
        """
        Retrieves the Instagram profile for the specified user.

        Returns:
        --------
        - Profile object if successful, otherwise None.
        """
        try:
            return instaloader.Profile.from_username(self.loader.context, self.user)
        
        except ConnectionException:
            instagram_logger.error("Connection failed: Instagram is blocking access.")
            return None

    def process_posts(self, profile):
        """
        Processes Instagram posts for the given profile and downloads images.

        Arguments:
        ----------
        - profile : `instaloader.Profile` object representing the Instagram account.

        Returns:
        --------
        - post_data (list) : List of dictionaries containing post details.

        Raises:
        -------
        - Exception: If an error occurs while downloading posts.
        """
        post_data = []

        for i, post in enumerate(profile.get_posts()):
            time.sleep(random.uniform(5, 10))  # Avoid triggering Instagram's rate limits
            
            try:
                post_info = self.download_post_images(post)
                post_data.append(post_info)
                instagram_logger.info(f"Successfully downloaded post {i+1}: {', '.join(post_info['image_paths'])}")
            
            except Exception as e:
                instagram_logger.error(f"Error downloading post {i+1}: {str(e)}")

        return post_data

    def download_post_images(self, post):
        """
        Downloads images from a single Instagram post.

        Arguments:
        ----------
        - post : `instaloader.Post` object containing Instagram post data.

        Returns:
        --------
        - Dictionary containing post ID, caption, image file paths, and image URLs.
        """
        image_paths = []
        image_urls  = []

        # If the post contains multiple images (GraphSidecar)
        if post.typename == "GraphSidecar":
            for j, sidecar in enumerate(post.get_sidecar_nodes()):
                image_path, image_url = self.download_image(post.shortcode, 
                                                            sidecar.display_url, 
                                                            post.date_local, j+1)
                image_paths.append(image_path)
                image_urls.append(image_url)
        else:
            image_path, image_url = self.download_image(post.shortcode, 
                                                        post.url, 
                                                        post.date_local)
            image_paths.append(image_path)
            image_urls.append(image_url)

        return {
            "post_id"       : post.shortcode,
            "post_contents" : post.caption,
            "image_paths"   : image_paths,
            "image_urls"    : image_urls
        }

    def download_image(self, shortcode, url, date_local, index=None):
        """
        Downloads a single image from Instagram.

        Arguments:
        ----------
        - shortcode   : Unique identifier for the Instagram post.
        - url         : Direct URL of the image.
        - date_local  : Post date used for metadata.
        - index       : Image index (if multiple images exist in a post).

        Returns:
        --------
        - Tuple containing the local image path and image URL.
        """
        image_filename = f"post_{shortcode}{'_' + str(index) if index else ''}.jpg"
        image_path     = os.path.join(self.image_dir, image_filename)

        self.loader.download_pic(image_path, url, date_local)

        return image_path, url

    def save_metadata(self, post_data):
        """
        Saves scraped post metadata into a JSON file.

        Arguments:
        ----------
        - post_data : List of dictionaries containing post details.

        Returns:
        --------
        - None (saves JSON file).
        """
        with open(self.data_path, mode = "w", encoding = "utf-8") as json_file:
            json.dump(post_data, json_file, indent = 4, ensure_ascii = False)

        instagram_logger.info(f"All images saved in '{self.image_dir}'")
        instagram_logger.info(f"Post details saved in '{self.data_path}'")
