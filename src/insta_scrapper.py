import instaloader
import re
import os
import json
import time
import random

class InstaDataScrapper():
    """
    A class to scrape Instagram posts for a given user, download images, extract metadata,
    and save post details in a JSON file.
    """
    def __init__(self, user: str, image_dir: str = '../data/instagram_images', data_path: str = '../data/instagram_post_data.json'):
        """
        Initializes the scraper with a username, directory for images, and data file path.

        :param user: Instagram username to scrape.
        :param image_dir: Directory to save images.
        :param data_path: Path to save post details in JSON format.
        """
        self.user = user
        self.image_dir = image_dir
        self.data_path = data_path
        self.loader = instaloader.Instaloader()
    
    def extract_hashtags(self, caption):
        """
        Extracts hashtags from a given caption.

        :param caption: Post caption as a string.
        :return: List of hashtags.
        """
        if caption:
            return re.findall(r"#\w+", caption)
        return []
        
    def remove_hashtags(self, caption):
        """
        Removes hashtags from a caption.

        :param caption: Post caption as a string.
        :return: Caption without hashtags.
        """
        if caption:
            return re.sub(r"#\w+", "", caption).strip()
        return ""
    
    def split_at_first_delimiter(self, text):
        """
        Splits text at the first occurrence of a delimiter (., !, ?).
        The first sentence is used as the heading, and the rest as content.

        :param text: Input text.
        :return: Tuple containing (heading, content).
        """
        match = re.search(r'[.!?]', text)
        if match:
            split_index = match.start() + 1
            heading = text[:split_index].strip()
            content = text[split_index:].strip()
            content = re.sub(r'^[.!?\s]+', '', content)
            return heading, content
        else:
            return text.strip(), ""

    def insta_scrapper(self):
        """
        Scrapes Instagram posts, downloads images, and saves metadata.
        """
        try:
            profile = instaloader.Profile.from_username(self.loader.context, self.user)
            posts = profile.get_posts()
            os.makedirs(self.image_dir, exist_ok=True)
            
            post_data = []
            
            for i, post in enumerate(posts):
                time.sleep(random.uniform(3, 7))
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
                    
                    hashtags = self.extract_hashtags(post.caption)
                    clean_caption = self.remove_hashtags(post.caption)
                    caption_text, content_text = self.split_at_first_delimiter(clean_caption)
                    
                    post_info = {
                        "post_id": post.shortcode,
                        "post_heading": caption_text,
                        "post_content": content_text,
                        "hashtags": hashtags,
                        "image_paths": image_paths,
                        "image_urls": image_urls
                    }
                    
                    post_data.append(post_info)
                    print(f"Downloaded Post {i+1}: {', '.join(image_paths)}")
                except Exception as e:
                    print(f"Error downloading post {i+1}: {str(e)}")
            
            with open(self.data_path, mode="w", encoding="utf-8") as json_file:
                json.dump(post_data, json_file, indent=4, ensure_ascii=False)
            
            print(f"All images saved in '{self.image_dir}'")
            print(f"Post details saved in '{self.data_path}'")
        
        except instaloader.exceptions.ProfileNotExistsException:
            print(f"Error: Profile '{self.user}' does not exist.")
        except instaloader.exceptions.ConnectionException:
            print("Error: Unable to connect to Instagram. Check your internet connection.")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")