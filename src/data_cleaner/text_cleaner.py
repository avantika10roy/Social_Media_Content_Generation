# ---------- DONE BY SOUMIK ----------

# DEPENDENCIES
import os
import re
import emoji
import shutil
import logging
import pandas as pd
from src.utils.logger import LoggerSetup

# Create Logger instance for this file
cleaner_logger = LoggerSetup(logger_name="data_cleaner.py", log_filename_prefix="DataCleaner").get_logger()

class DataCleaner:
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
        cleaner_logger.info("DataCleaner instance created")
        
    def extract_emojis(self, text: pd.Series) -> pd.Series:
        """
        Extracts distinct emojis from each text entry in a pandas Series.

        Args:
            text (pd.Series): The input text data.
        
        Returns:
            pd.Series: A Series containing lists of distinct emojis for each text entry.
        """
        try:
            cleaner_logger.info("Extracting emojis from text.")
            return text.apply(emoji.distinct_emoji_list)
        except Exception as e:
            cleaner_logger.error(f"Error extracting emojis: {e}")
            return text

    def extract_hashtags(self, text: pd.Series) -> pd.Series:
        """
        Extracts hashtags from each text entry in a pandas Series.

        Args:
            text (pd.Series): The input text data.
        
        Returns:
            pd.Series: A Series containing lists of extracted hashtags for each text entry.
        """
        try:
            cleaner_logger.info("Extracting hashtags from text.")
            return text.apply(lambda x: self.HASHTAG_PATTERN.findall(x))
        except Exception as e:
            cleaner_logger.error(f"Error extracting hashtags: {e}")
            return text
        
    def clean_text(self, text: pd.Series) -> pd.Series:
        """
        Cleans text by removing emojis, hashtags, HTML tags, extra spaces, and standardizing case.

        Args:
            text (pd.Series): The input text data.
        
        Returns:
            pd.Series: A cleaned text Series.
        """
        try:
            cleaner_logger.info("Cleaning text data.")
            text = text.apply(lambda x: emoji.replace_emoji(x, replace=''))  # Remove emojis
            text = text.str.replace(self.HASHTAG_PATTERN, '', regex=True).str.strip()
            text = text.str.replace(self.HTML_TAG_PATTERN, '', regex=True)
            text = text.str.replace(self.NEWLINE_PATTERN, '', regex=True)
            text = text.str.replace(self.EXTRA_DOTS_PATTERN, '', regex=True)
            text = text.str.replace(r'\bhashtag\b', '', regex=True)
            text = text.str.replace(self.EXTRA_SPACES_PATTERN, ' ', regex=True)
            cleaner_logger.info("Text cleaning complete.")
            return text
        except Exception as e:
            cleaner_logger.error(f"Error cleaning text: {e}")
            return text
        
    def split_at_first_delimiter(self, gen_data: pd.DataFrame) -> pd.DataFrame:
        """
        Splits text in the 'post_contents' column into heading and content at the first delimiter.

        Args:
            gen_data (pd.DataFrame): The input DataFrame containing a 'post_contents' column.
        
        Returns:
            pd.DataFrame: A DataFrame with 'post_heading' and 'post_content' columns.
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
            gen_data[['post_heading', 'post_content']] = gen_data['post_contents'].apply(split_text).apply(pd.Series)
            return gen_data
        except Exception as e:
            cleaner_logger.error(f"Error splitting text in DataFrame: {e}")
            return gen_data
        
    def instagram_clean_data(self, insta_data: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        Cleans Instagram data by extracting emojis, hashtags, and cleaning text.

        Args:
            insta_data (pd.DataFrame): The input Instagram data DataFrame.
            platform (str): The platform name.
        
        Returns:
            pd.DataFrame: The cleaned Instagram data.
        """
        try:
            insta_data['emoji'] = self.extract_emojis(insta_data['post_content'])
            insta_data['post_content'] = self.clean_text(insta_data['post_content'])
            insta_data['post_heading'] = self.clean_text(insta_data['post_heading'])
            insta_data = self.split_at_first_delimiter(insta_data)
            insta_data.drop(columns=['image_urls', 'post_id'], errors='ignore', inplace=True)
            insta_data['platform'] = 'instagram'
            cleaner_logger.info(f"{platform} data cleaned successfully.")
            return insta_data
        except Exception as e:
            cleaner_logger.error(f"Error in {platform} cleaner: {e}")
            return pd.DataFrame()

    def linkdln_clean_data(self, linkdln_data: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        Cleans LinkedIn data by extracting emojis, hashtags, and cleaning text.

        Args:
            linkdln_data (pd.DataFrame): The input LinkedIn data DataFrame.
            platform (str): The platform name.
        
        Returns:
            pd.DataFrame: The cleaned LinkedIn data.
        """
        try:
            linkdln_data['emoji'] = self.extract_emojis(linkdln_data['post_contents'])
            linkdln_data['hashtags'] = self.extract_hashtags(linkdln_data['post_contents'])
            linkdln_data['post_contents'] = self.clean_text(linkdln_data['post_contents'])
            linkdln_data = self.split_at_first_delimiter(linkdln_data)
            linkdln_data.drop(columns=['image_URLs', 'post_contents'], errors='ignore', inplace=True)
            linkdln_data['platform'] = 'linkedin'
            cleaner_logger.info(f"{platform} data cleaned successfully.")
            return linkdln_data
        except Exception as e:
            cleaner_logger.error(f"Error in {platform} cleaner: {e}")
            return pd.DataFrame()
