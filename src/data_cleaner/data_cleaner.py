import emoji
import re
import pandas as pd
import logging
from src.utils.logger import LoggerSetup

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
        
    def extract_emojis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts distinct emojis from each text entry in a pandas DataFrame.
        """
        try:
            cleaner_logger.info("Extracting emojis from text.")
            data['emoji'] = data['post_contents'].apply(emoji.distinct_emoji_list)
            return data
        except Exception as e:
            cleaner_logger.error(f"Error extracting emojis: {e}")
            return data

    def extract_hashtags(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts hashtags from each text entry in a pandas DataFrame.
        """
        try:
            cleaner_logger.info("Extracting hashtags from text.")
            data['hashtags'] = data['post_contents'].apply(lambda x: self.HASHTAG_PATTERN.findall(x))
            return data
        except Exception as e:
            cleaner_logger.error(f"Error extracting hashtags: {e}")
            return data
        
    def clean_text(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans text by performing various operations.
        """
        try:
            cleaner_logger.info("Cleaning text data.")
            data['post_contents'] = data['post_contents'].apply(lambda x: emoji.replace_emoji(x,replace=''))  # Remove emojis
            data['post_contents'] = data['post_contents'].str.replace(self.HASHTAG_PATTERN, '', regex=True).str.strip()  # Remove hashtags
            data['post_contents'] = data['post_contents'].str.lower()  # Lowercasing
            data['post_contents'] = data['post_contents'].str.replace(self.HTML_TAG_PATTERN, '', regex=True)  # Remove HTML tags
            data['post_contents'] = data['post_contents'].str.replace(self.NEWLINE_PATTERN, '', regex=True)  # Remove '\n'
            data['post_contents'] = data['post_contents'].str.replace(self.EXTRA_DOTS_PATTERN, '', regex=True)  # Remove extra dots
            data['post_contents'] = data['post_contents'].str.replace(self.EXTRA_SPACES_PATTERN, ' ', regex=True)  # Remove extra spaces
            #data.drop('image_URLs',axis=1)
            
            cleaner_logger.info("Text cleaning complete.")
            return data
        
        except Exception as e:
            cleaner_logger.error(f"Error cleaning text: {e}")
            return data
        
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
            self.extract_emojis(insta_data)
            self.extract_hashtags(insta_data)
            cleaned_insta_data = self.clean_text(insta_data)
            
            # Assuming 'image_URLS' is a column in the DataFrame
            #if 'image_URLS' in cleaned_insta_data.columns:
             #   cleaned_insta_data = cleaned_insta_data.drop('image_URLS', axis=1)
            
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
            self.extract_emojis(linkdln_data)
            self.extract_hashtags(linkdln_data)
            self.split_at_first_delimiter(linkdln_data)
            cleaned_text = self.clean_text(linkdln_data)
            
            # Assuming 'image_URLS' is a column in the DataFrame
            
            cleaned_text = cleaned_text.drop(columns=['image_URLs','post_contents'], axis=1)
            
            cleaner_logger.info(f"{platform} data cleaned")
        
            return cleaned_text
        
        except Exception as e:
            cleaner_logger.error(f"Error in {platform} cleaner: {e}")
            return linkdln_data
    