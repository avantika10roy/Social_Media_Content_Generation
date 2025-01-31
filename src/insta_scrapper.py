# DONE BY JIT
import instaloader
import re
import os
import json

class InstaDataScrapper():
    """
    A class to scrape Instagram data for a given user, including downloading images and extracting post information.
    
    Attributes:
    user (str): The Instagram username whose data will be scraped.
    image_dir (str): Directory to store downloaded images. Default is 'instagram_images'.
    data_path (str): Path to save post data as a JSON file. Default is 'instagram_post_data.json'.
    loader (instaloader.Instaloader): Instaloader instance to interact with Instagram API.
    
    Methods:
    extract_hashtags(caption: str) -> list:
        Extracts hashtags from a given caption.
    
    remove_hashtags(caption: str) -> str:
        Removes hashtags from a given caption and returns the cleaned caption.
    
    insta_scrapper():
        Scrapes posts from the specified Instagram profile, downloads images, 
        extracts post information (e.g., likes, comments, hashtags), 
        and saves the data in a JSON file.
    """
    def __init__(self, user: str, image_dir: str = 'instagram_images', data_path: str='instagram_post_data.json'):
        """
        Initializes the InstaDataScrapper with the specified Instagram username, image directory, and data file path.

        Parameters:
        user (str): The Instagram username to scrape data for.
        image_dir (str): Directory to store downloaded images. Default is 'instagram_images'.
        data_path (str): Path to save post data as a JSON file. Default is 'instagram_post_data.json'.
        """

        self.user = user
        self.image_dir = image_dir
        self.data_path = data_path
        self.loader = instaloader.Instaloader()

    def extract_hashtags(self, caption):
        """
        Extracts hashtags from a given caption.

        Parameters:
        caption (str): The caption text from which hashtags will be extracted.

        Returns:
        list: A list of hashtags found in the caption.
        """
        if caption:
            return re.findall(r"#\w+", caption)
        return []
        
    def remove_hashtags(self, caption):
        """
        Removes hashtags from a given caption and returns the cleaned caption.

        Parameters:
        caption (str): The caption text from which hashtags will be removed.

        Returns:
        str: The cleaned caption without hashtags.
        """
        if caption:
            return re.sub(r"#\w+", "", caption).strip()
        return ""

    def insta_scrapper(self):
        """
        Scrapes posts from the specified Instagram profile, downloads images, 
        extracts post information (e.g., likes, comments, hashtags), 
        and saves the data in a JSON file.

        This method:
        - Fetches posts from the specified Instagram user.
        - Downloads images (or sidecar images for multiple-image posts).
        - Extracts relevant data such as the caption, number of likes, comments, and hashtags.
        - Saves the data in a JSON file for further use.

        The downloaded images will be stored in the directory specified by `image_dir`.
        The post details will be saved in a JSON file at the location specified by `data_path`.
        """
        profile = instaloader.Profile.from_username(self.loader.context, self.user)
        posts = profile.get_posts()
        os.makedirs(self.image_dir, exist_ok=True)

        post_data = []

        for i, post in enumerate(posts):
            image_paths = []

            if post.typename == "GraphSidecar":
                for j, sidecar in enumerate(post.get_sidecar_nodes()):
                    image_filename = f"post_{post.shortcode}_{j+1}.jpg"
                    image_path = os.path.join(self.image_dir, image_filename)
                    self.loader.download_pic(image_path, sidecar.display_url, post.date_local)
                    image_paths.append(image_path)

            else:
                image_filename = f"post_{post.shortcode}.jpg"
                image_path = os.path.join(self.image_dir, image_filename)
                self.loader.download_pic(image_path, post.url, post.date_local)
                image_paths.append(image_path)

            hashtags = self.extract_hashtags(post.caption)
            clean_caption = self.remove_hashtags(post.caption)

            post_info = {
                "post_id": post.shortcode,
                "caption": clean_caption,
                "likes": post.likes,
                "comments": post.comments,
                "hashtags": hashtags,
                "image_paths": image_paths
            }

            post_data.append(post_info)

            print(f"Downloaded Post {i+1}: {', '.join(image_paths)}")

        with open(self.data_path, mode="w", encoding="utf-8") as json_file:
            json.dump(post_data, json_file, indent=4, ensure_ascii=False)

        print(f"All images saved in '{self.image_dir}'")
        print(f"Post details saved in '{self.data_path}'")
pp = InstaDataScrapper(user = 'itobuztechnologies')
pp.insta_scrapper()