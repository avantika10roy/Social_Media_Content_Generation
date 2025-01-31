## ----- DONE BY PRIYAM PAL AND SUBHAS MUKHERJEE -----

# DEPENDENCIES

import re
import os
import time
import logging
import pandas as pd
from bs4 import BeautifulSoup

from config import LINKEDIN_POST_DATA_PATH
from config import LINKEDIN_LOGIN_PAGE_LINK
from config import LINKEDIN_POST_DATA_FILENAME

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
        
        self.username    = username
        self.password    = password
        self.profile_url = profile_url
        self.driver      = None
        self.wait        = None

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
        
        chrome_options  = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-notifications")
        
        service         = Service(chromedriver_path)
        self.driver     = webdriver.Chrome(service = service, 
                                           options = chrome_options)
        
        self.wait       = WebDriverWait(self.driver, 10)

    
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
                df.to_csv(output_file, 
                          index    = False, 
                          encoding = 'utf-8')

                print("Scraping complete. Data saved to post_data.csv")
                print(f"Total posts scraped: {len(df)}")
                return df
            else:
                print("No posts found. Please check the selectors and scroll logic.")
                return pd.DataFrame()
                
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
        
        self.driver.get(LINKEDIN_LOGIN_PAGE_LINK)
        self.wait.until(EC.presence_of_element_located((By.ID, "username")))
        
        email_field    = self.driver.find_element(By.ID, "username")
        password_field = self.driver.find_element(By.ID, "password")
        
        email_field.send_keys(self.username)
        password_field.send_keys(self.password)
        password_field.send_keys(Keys.RETURN)
        
        self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, 
                                                        "feed-shared-update-v2"
                                                        )
                                                       )
                        )

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
        
        hashtags          = re.findall(r'(?:hashtag)?#\w+', text)
        cleaned_hashtags  = list(set
                                 (
                                     [
                                         tag.replace('hashtag#', '#') 
                                         for tag in hashtags
                                         ]
                                     )
                                 )
        
        return cleaned_hashtags

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
        
        text          = re.sub(r'hashtag#\w+', '', text)
        text          = re.sub(r'#\w+', '', text)
        text          = re.sub(r'\.{3,}', '.', text)
        
        return ' '.join(text.split()).strip()


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
        
        match            = re.search(r'[.!?]', text)
        
        if match:
            split_index  = match.start() + 1
            heading      = text[:split_index].strip()
            content      = text[split_index:].strip()
            content      = re.sub(r'^[.!?\s]+', '', content)
            return heading, content
        
        else:
            return text.strip(), ""

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
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F700-\U0001F77F"  # Alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric shapes
            "\U0001F800-\U0001F8FF"  # Supplemental arrows
            "\U0001F900-\U0001F9FF"  # Supplemental symbols & pictographs
            "\U0001FA00-\U0001FA6F"  # Chess symbols
            "\U0001FA70-\U0001FAFF"  # Miscellaneous symbols
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # Enclosed characters
            "]+", 
            flags = re.UNICODE
        )
        
        emojis        = emoji_pattern.findall(text)  
        cleaned_text  = emoji_pattern.sub('', text)  
        
        return cleaned_text.strip(), ''.join(emojis)

    def _scrape_linkedin_posts(self) -> list:
        """
        Scrape LinkedIn posts and extract relevant information.
        
        Returns:
        ---------
            list  : List of dictionaries containing post data
        """
        
        self.driver.get(self.profile_url)
        self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, 
                                                        "feed-shared-update-v2"
                                                        )
                                                       )
                        )
        
        self._scroll_down()
        
        page_source                  = self.driver.page_source
        soup                         = BeautifulSoup(page_source, 'html.parser')
        posts                        = soup.find_all('div', {'class': ['feed-shared-update-v2', 'update-components-text']})
        scraped_data                 = []

        for post in posts:
            try:
                content_element      = post.find('span', {'class': 'break-words'})
                if not content_element:
                    continue

                full_text            = content_element.get_text(strip=True)
                
                cleaned_text         = self._clean_text(full_text)
                
                hashtags             = self._extract_hashtags(full_text)
                
                cleaned_text, emojis = self._extract_emojis(cleaned_text)
                
                heading, content     = self._split_at_first_delimiter(cleaned_text)
                
                if heading:
                    scraped_data.append({
                        "Post Caption/Heading": heading,
                        "Post Content": content if content else "No content",
                        "Hashtags": ', '.join(hashtags) if hashtags else "No hashtags",
                        "Emojis": emojis if emojis else "No emojis"
                    })

            except Exception as e:
                print(f"Error extracting post: {e}")
                continue

        return scraped_data
