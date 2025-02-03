## ----- DONE BY PRIYAM PAL AND SUBHAS MUKHERJEE -----

# DEPENDENCIES

import re
import os
import time
import emoji
import requests
import pandas as pd
from bs4 import BeautifulSoup

from config import LINKEDIN_POST_DATA_PATH
from config import LINKEDIN_LOGIN_PAGE_LINK
from config import LINKEDIN_IMAGE_DATA_PATH
from config import LINKEDIN_POST_DATA_FILENAME

from .logger import LoggerSetup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# LOGGING SETUP
scraper_logger = LoggerSetup(logger_name = "web_scraper.py", log_filename_prefix = "linkedin").get_logger()

class SocialMediaScraper:
    """
    A class to handle social media scraping operations.
    
    This class provides functionality for scraping various social media platforms,
    with initial implementation focusing on LinkedIn.
    
    Attributes:
    -----------
        username (str)     : The username for social media login
        password (str)     : The password for social media login
        profile_url (str)  : The URL to scrape
        driver             : Selenium WebDriver instance
        wait               : WebDriverWait instance
    """
    
    def __init__(self, username: str, password: str, profile_url: str):
        """
        Initialize the SocialMediaScraper with login credentials and target URL.
        
        Arguments:
        ----------
            username (str)    : Username for social media login
            password (str)    : Password for social media login
            profile_url (str) : URL to scrape
        """
        
        try:
            self.username    = username
            self.password    = password
            self.profile_url = profile_url
            self.driver      = None
            self.wait        = None
            
            scraper_logger.info("LinkedIn Scraper Class Initiated")
            
        except Exception as e:
            scraper_logger.error(f"An Error Occured: {repr(e)}")

    def setup_driver(self, chromedriver_path: str) -> None: 
        """
        Set up and configure the Chrome WebDriver for scraping.
        
        Arguments:
        ----------
            chromedriver_path (str) : Path to the ChromeDriver executable
            
        Returns:
        --------
            None
            
        Raises:
        -------
            WebDriverException      : If ChromeDriver initialization fails
        """
        
        try:
            chrome_options  = Options()
            chrome_options.add_argument("--start-maximized")
            chrome_options.add_argument("--disable-notifications")
        
            service         = Service(chromedriver_path)
            self.driver     = webdriver.Chrome(service = service, 
                                               options = chrome_options)
        
            self.wait       = WebDriverWait(self.driver, 10)
            
            scraper_logger.info("WebDriver Initialized")
            
        except Exception as e:
            scraper_logger.error(f"An Error Occured: {repr(e)}")

    
    def linkedin_scraper(self) -> pd.DataFrame:
        """
        Main function to handle LinkedIn scraping operations.
        
        Returns:
        --------
            pd.DataFrame : DataFrame containing scraped posts with columns for
                         heading, content, hashtags, and emojis
                         
        Raises:
        --------
            Exception    : If scraping or login fails
        """
        try:
            self._linkedin_login()
            posts_data        = self._scrape_linkedin_posts()
            
            if posts_data:
                df            = pd.DataFrame(posts_data)
                df            = df.drop_duplicates()

                output_dir    = LINKEDIN_POST_DATA_PATH
                os.makedirs(output_dir, 
                            exist_ok = True)

                output_file   = os.path.join(output_dir, LINKEDIN_POST_DATA_FILENAME)
                df.to_json(output_file, orient= 'records' , force_ascii= False, indent=4)

                scraper_logger.info("Scraping complete. Data saved to post_data.json")
                scraper_logger.info(f"Total posts scraped: {len(df)}")
                return df
            else:
                scraper_logger.info("No posts found. Please check the selectors and scroll logic.")
                return pd.DataFrame()
            
        except Exception as e:
            scraper_logger.error(f"An Error Occured: {repr(e)}")
                
        finally:
            if self.driver:
                self.driver.quit()


    def _linkedin_login(self) -> None:
        """
        Handle LinkedIn login process.
        
        Returns:
        ---------
            None
            
        Raises:
        --------
            TimeoutException : If login elements are not found
        """
        
        try:
            self.driver.get(LINKEDIN_LOGIN_PAGE_LINK)
            self.wait.until(EC.presence_of_element_located((By.ID, "username")))
        
            email_field    = self.driver.find_element(By.ID, "username")
            password_field = self.driver.find_element(By.ID, "password")
        
            email_field.send_keys(self.username)
            password_field.send_keys(self.password)
            password_field.send_keys(Keys.RETURN)
        
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "feed-shared-update-v2")))
            
            scraper_logger.info("Login Successful")
            
        except Exception as e:
            scraper_logger.error(f"An Error Occured: {repr(e)}")

    def _scroll_down(self) -> None:
        
        """
        Scroll down the page to load more posts.
        
        Returns:
        --------
            None
        """
        last_height     = self.driver.execute_script("return document.body.scrollHeight")
        
        posts_count     = len(self.driver.find_elements(By.CLASS_NAME, 
                                                        "feed-shared-update-v2"
                                                        )
                              )
        
        previous_count  = 0
        no_change_count = 0
        
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
            
            try:
                self.wait.until(lambda driver: len(driver.find_elements(By.CLASS_NAME, 
                                                                        "feed-shared-update-v2"
                                                                        )
                                                   ) > posts_count
                                )
            
            except:
                if posts_count == previous_count:
                    no_change_count += 1
                else:
                    no_change_count  = 0
                    
                if no_change_count  >= 3:
                    break
            
            previous_count           = posts_count
            posts_count              = len(self.driver.find_elements(By.CLASS_NAME, 
                                                                     "feed-shared-update-v2"
                                                                     )
                                           )
            
            new_height               = self.driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height:
                break
            last_height              = new_height

    @staticmethod
    def _extract_hashtags(text: str) -> list:
        
        """
        Extract hashtags from text content.
        
        Arguments:
        ----------
            text (str) : Text to extract hashtags from
            
        Returns:
        --------
            list       : List of unique hashtags found in the text
        """
        
        try:
            hashtags          = re.findall(r'(?:hashtag)?#\w+', text)
            cleaned_hashtags  = list(set
                                     (
                                         [
                                             tag.replace('hashtag#', 
                                                         '#'
                                                         ) 
                                             for tag in hashtags
                                             ]
                                         )
                                     )
            
            scraper_logger.info("Hastags Extracted")
            
            return cleaned_hashtags
        
        except Exception as e:
            scraper_logger.error(f"Error extracting Hastags: {repr(e)}")

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean text by removing hashtags and formatting.
        
        Arguments:
        ----------
            text (str) : Text to clean
            
        Returns:
        --------
            str        : Cleaned text
        """
        try:
            text         = re.sub(r'hashtag#\w+', '', text) 
            text         = re.sub(r'#\w+', '', text) 

            text         = emoji.replace_emoji(text, '')

            text         = re.sub(r'\.{3,}', '.', text)

            cleaned_text = ' '.join(text.split()).strip()

            scraper_logger.info("Text Cleaned")
            
            return cleaned_text

        except Exception as e:
            scraper_logger.error(f"Error cleaning text: {repr(e)}")
            return text 


    @staticmethod
    def _split_at_first_delimiter(text: str) -> tuple:
        """
        Split text into heading and content at first delimiter.
        
        Arguments:
        ----------
            text (str)  : Text to split
            
        Returns:
        ---------
            tuple       : (heading, content) pair
        """
        try:
        
            match            = re.search(r'[.!?]', text)
        
            if match:
                split_index  = match.start() + 1
                heading      = text[:split_index].strip()
                content      = text[split_index:].strip()
                content      = re.sub(r'^[.!?\s]+', '', content)
                scraper_logger.info("Text Splitted into Heading and Content")
                return heading, content
        
            else:
                scraper_logger.warning("No Text Splitted into Heading and Content")
                return text.strip(), ""
            
        except Exception as e:
            scraper_logger.error(f"An Error Occured: {repr(e)}")

    @staticmethod
    def _extract_emojis(text: str) -> tuple:
        """
        Extracts emojis from a string and returns cleaned text and emojis.
        
        Arguments:
        ----------
            text (str) : Text to extract emojis from
            
        Returns:
        --------
            tuple      : (cleaned_text, emojis) pair
        """
        try:
            emojis = [e['emoji'] for e in emoji.emoji_list(text)]
            scraper_logger.info("Emoji Extracted")
            return list(set(emojis))
        
        except Exception as e:
            scraper_logger.error(f"Error extracting emojis: {repr(e)}")
            return []
    
    @staticmethod
    def create_image_folder():
        """
        Function to Create Image Folder
        
        Arguments:
        ----------
            None
            
        Returns:
        --------
            None
        """
        
        folder_path   = LINKEDIN_IMAGE_DATA_PATH
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def is_valid_image(img_element):
        """
        Checks whether an image element contains a valid image source.
        
        Arguments:
        -----------
            img_element (dict): A dictionary representing an HTML image element 
                                (e.g., BeautifulSoup tag or a similar structure).

        Returns:
        --------
            bool              : True if the image source is considered valid, False otherwise.
        """
        
        try:
            
            src                = img_element.get('src', '').lower()
            invalid_indicators = ['logo', 
                                  'profile', 
                                  'company', 
                                  'brand', 
                                  'avatar', 
                                  '8fz8rainn3wh49ad6ef9gotj1'
                                  ]
            
            scraper_logger.info("Image is Valid")
        
            return not any(indicator in src for indicator in invalid_indicators)
        
        except Exception as e:
            scraper_logger.error(f"Error Occured capturing image: {repr(e)}")
    
    def download_image(self, image_url, post_id):
        """
        Downloads an image from the given URL and saves it locally with a filename 
        that includes the post ID.

        Arguments:
        ----------
            image_url (str): The URL of the image to be downloaded.
            post_id (str): The unique identifier for the post associated with the image.

        Returns:
        --------
            str: The local file path of the downloaded image if successful.
            None: If the download fails or an error occurs.

        Logs:
        -----
            - Info: Logs the successful download of an image.
            - Warning: Logs a failure if the image cannot be downloaded.
            - Error: Logs any exceptions encountered during the download process.
        """     
        try:
            response       = requests.get(image_url, stream=True)
            
            if response.status_code == 200:
                image_name = f"{post_id}_{image_url.split('/')[-1].split('?')[0]}.png"
                image_path = os.path.join(LINKEDIN_IMAGE_DATA_PATH, image_name)
                
                with open(image_path, 'wb') as file:
                
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                
                scraper_logger.info(f"Downloaded: {image_name}")
                
                return image_path
            
            else:
                scraper_logger.warning(f"Failed to download image from {image_url}")
            
                return None
        except Exception as e:
            
            scraper_logger.error(f"Error downloading image: {e}")
            
            return None

    def _scrape_linkedin_posts(self) -> list:
        """
        Scrape LinkedIn posts and extract relevant information while preventing duplicates.
    
        Returns:
        ---------
            list  : List of dictionaries containing unique post data
        """
        
        try:
            
            self.driver.get(self.profile_url)
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "feed-shared-update-v2")))
            self._scroll_down()

            page_source               = self.driver.page_source
            soup                      = BeautifulSoup(page_source, 'html.parser')
            posts                     = soup.find_all('div', {'class': ['feed-shared-update-v2', 'update-components-text']})
    
            seen_contents             = set()
            scraped_data              = list()

            self.create_image_folder()

            for post_id, post in enumerate(posts, start = 1):
                try:
                    content_element   = post.find('span', {'class': 'break-words'})
                    if not content_element:
                        continue
            
                    full_text          = content_element.get_text(strip=True)
            
                    content_identifier = full_text[:100].strip() 
            
                    if content_identifier in seen_contents:
                        continue
            
                    seen_contents.add(content_identifier)
            
                    hashtags           = self._extract_hashtags(full_text)
                    cleaned_text       = self._clean_text(full_text)
                    emojis             = self._extract_emojis(full_text)
                    heading, content   = self._split_at_first_delimiter(cleaned_text)

                    if not heading.strip():
                        continue

                    image_elements     = post.find_all('img', {'class': 'ivm-view-attr__img--centered'})
                    valid_images       = [img for img in image_elements if self.is_valid_image(img)]
                    image_urls         = list(set([img['src'] for img in valid_images if img.get('src')]))
                    image_paths        = [self.download_image(url, post_id) for url in image_urls if url]

                    post_data          = {"Post_Heading": heading,
                                          "Post_Content": content if content else "No content",
                                          "Hashtags": ', '.join(hashtags) if hashtags else "No hashtags",
                                          "Emojis": ', '.join(emojis) if emojis else "No emojis",
                                          "Image_URLs": ', '.join(image_urls),
                                          "Image_Paths": ', '.join(filter(None, image_paths))
                                          }

                    full_post_content   = f"{heading} {content}".lower()
                    is_duplicate        = False
            
                    for existing_post in scraped_data:
                        existing_content = f"{existing_post['Post_Heading']} {existing_post['Post_Content']}".lower()
                        
                        if (full_post_content in existing_content) or (existing_content in full_post_content):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        scraped_data.append(post_data)

                except Exception as e:
                    scraper_logger.error(f"Error extracting post: {e}")
                    continue

            scraper_logger.info(f"Total unique posts found: {len(scraped_data)}")
            return scraped_data
        
        except Exception as e:
            scraper_logger.error(f"Error Occured in Scraping: {str(e)}")