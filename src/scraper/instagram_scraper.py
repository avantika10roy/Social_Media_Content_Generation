import instaloader
import os
import json
import time
import random
from config import config
from ..utils.logger import LoggerSetup
from config.config import Config

instgram_logger = LoggerSetup(logger_name = "instagram_scraper", log_filename_prefix = "InstagramDataScrapper").get_logger()

cf = Config()
class InstagramDataScraper():
    """
    A class to scrape Instagram posts for a given user, download images, extract metadata,
    and save post details in a JSON file.
    """
    def __init__(self, user: str = cf.INSTAGRAM_USERNAME, image_dir: str = cf.INSTAGRAM_IMAGE_DATA_PATH, data_path: str = cf.INSTAGRAM_POST_DATA_PATH):
        """
        Initializes the scraper with a username, directory for images, and data file path.
        """
        self.user = user
        self.image_dir = image_dir
        self.data_path = data_path
        self.loader = instaloader.Instaloader()
        instgram_logger.info(f"Initialized InstagramDataScrapper for user: {self.user}")

    def instagram_scraper(self):
        """Scrapes Instagram posts, downloads images, and saves metadata."""
        try:
            instgram_logger.info(f"Starting scraping for user: {self.user}")
            try:
                profile = instaloader.Profile.from_username(self.loader.context, self.user)
            except instaloader.exceptions.ConnectionException:
                instgram_logger.error("Connection failed: Instagram is blocking access.")
                return
            posts = profile.get_posts()
            os.makedirs(self.image_dir, exist_ok=True)
            
            post_data = []
            
            for i, post in enumerate(posts):
                time.sleep(random.uniform(5, 10))
                image_paths = []
                image_urls = []
                
                try:
                    if post.typename == "GraphSidecar":
                        for j, sidecar in enumerate(post.get_sidecar_nodes()):
                            image_filename = f"post_{post.shortcode}_{j+1}.jpg"
                            image_path = os.path.join(self.image_dir, image_filename)
                            self.loader.download_pic(image_path, sidecar.display_url, post.date_local)
                            image_paths.append(image_path)
                            image_urls.append(sidecar.display_url)
                    else:
                        image_filename = f"post_{post.shortcode}.jpg"
                        image_path = os.path.join(self.image_dir, image_filename)
                        self.loader.download_pic(image_path, post.url, post.date_local)
                        image_paths.append(image_path)
                        image_urls.append(post.url)
                    
                    post_info = {
                        "post_id": post.shortcode,
                        "post_contents": post.caption,
                        "image_paths": image_paths,
                        "image_urls": image_urls
                    }
                    
                    post_data.append(post_info)
                    instgram_logger.info(f"Successfully downloaded post {i+1}: {', '.join(image_paths)}")
                except Exception as e:
                    instgram_logger.error(f"Error downloading post {i+1}: {str(e)}")
            
            with open(self.data_path, mode="w", encoding="utf-8") as json_file:
                json.dump(post_data, json_file, indent=4, ensure_ascii=False)
            
            instgram_logger.info(f"All images saved in '{self.image_dir}'")
            instgram_logger.info(f"Post details saved in '{self.data_path}'")
        
        except instaloader.exceptions.ProfileNotExistsException:
            instgram_logger.error(f"Error: Profile '{self.user}' does not exist.")
        except instaloader.exceptions.ConnectionException:
            print("Error: Unable to connect to Instagram. Check your internet connection.")
        except Exception as e:
            instgram_logger.error(f"An unexpected error occurred: {str(e)}")
