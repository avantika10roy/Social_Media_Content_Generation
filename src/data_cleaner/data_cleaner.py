'''import emoji
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
        Cleans Instagram data by extracting emojis, hashtags, and cleaning text.
        """
        try:
            if 'post_content' not in insta_data:
                raise KeyError("Missing 'post_content' column in Instagram data.")

            insta_data['emoji'] = self.extract_emojis(insta_data['post_content'])
            #insta_data['hashtags'] = self.extract_hashtags(insta_data['post_content'])
            insta_data['post_content'] = self.clean_text(insta_data['post_content'])
            insta_data = self.split_at_first_delimiter(insta_data)

            if 'image_urls' in insta_data and 'post_id' in insta_data:
                insta_data = insta_data.drop(columns=['image_urls', 'post_id'], errors='ignore')

            insta_data['platform'] = 'instagram'
            cleaner_logger.info(f"{platform} data cleaned successfully.")
            return insta_data

        except Exception as e:
            cleaner_logger.error(f"Error in {platform} cleaner: {e}")
            return pd.DataFrame()  

        
    def linkdln_clean_data(self, linkdln_data: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        Cleans LinkedIn data by extracting emojis, hashtags, and cleaning text.
        """
        try:
            if 'post_contents' not in linkdln_data:
                raise KeyError("Missing 'post_contents' column in LinkedIn data.")

            linkdln_data['emoji'] = self.extract_emojis(linkdln_data['post_contents'])
            linkdln_data['hashtags'] = self.extract_hashtags(linkdln_data['post_contents'])
            linkdln_data['post_contents'] = self.clean_text(linkdln_data['post_contents'])
            linkdln_data = self.split_at_first_delimiter(linkdln_data)

            if 'image_URLs' in linkdln_data and 'post_contents' in linkdln_data:
                linkdln_data = linkdln_data.drop(columns=['image_URLs', 'post_contents'], errors='ignore')

            linkdln_data['platform'] = 'linkedin'
            cleaner_logger.info(f"{platform} data cleaned successfully.")
            return linkdln_data

        except Exception as e:
            cleaner_logger.error(f"Error in {platform} cleaner: {e}")
            return pd.DataFrame()  

    def facebook_cleaner(self, facebook_data: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        Cleans Facebook data by extracting emojis, hashtags, and cleaning text.
        """
        try:
            if 'post_contents' not in facebook_data:
                raise KeyError("Missing 'post_contents' column in Facebook data.")
            
            facebook_data['emoji']=self.extract_emojis(facebook_data['post_contents'])
            facebook_data['hashtags']=self.extract_hashtags(facebook_data['post_contents'])
            facebook_data['post_contents']=self.clean_text(facebook_data['post_contents'])
            facebook_data = self.split_at_first_delimiter(facebook_data)

            if 'image_URLs' in facebook_data and 'post_contents' in facebook_data:
                facebook_data = facebook_data.drop(columns=['image_URLs', 'post_contents'], errors='ignore')
            
            facebook_data['platform']='facebook'
            cleaner_logger.info(f"{platform} data cleaned successfully.")
            return facebook_data

        except Exception as e:
            cleaner_logger.error(f"Error in {platform} cleaner: {e}")
            return pd.DataFrame()''' 


import emoji
import re
import pandas as pd
import logging
import os
import shutil
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
            removed_emoji = text.apply(lambda x: emoji.replace_emoji(x, replace=''))  # Remove emojis
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
            gen_data[['post_heading', 'post_content']] = gen_data['post_content'].apply(split_text).apply(pd.Series)
            return gen_data
        except Exception as e:
            cleaner_logger.error(f"Error splitting text in DataFrame: {e}")
            return gen_data
        
    def instagram_clean_data(self, insta_data: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        Cleans Instagram data by extracting emojis, hashtags, and cleaning text.
        """
        try:
            if 'post_content' not in insta_data:
                raise KeyError("Missing 'post_content' column in Instagram data.")

            insta_data['emoji'] = self.extract_emojis(insta_data['post_content'])
            insta_data['post_content'] = self.clean_text(insta_data['post_content'])
            insta_data['post_heading'] = self.clean_text(insta_data['post_heading'])
            insta_data = self.split_at_first_delimiter(insta_data)

            if 'image_urls' in insta_data:
                insta_data = insta_data.drop(columns=['image_urls','post_id'], errors='ignore')

            insta_data['platform'] = 'instagram'
            cleaner_logger.info(f"{platform} data cleaned successfully.")
            return insta_data

        except Exception as e:
            cleaner_logger.error(f"Error in {platform} cleaner: {e}")
            return pd.DataFrame()
        
    '''def handle_images(self, image_paths: pd.Series):
        """
        Ensures proper handling of image paths, removing duplicates and filtering unwanted content.
        """
        # Placeholder for image processing logic (e.g., duplicate removal, face filtering)
        return image_paths

    def linkdln_clean_data(self, linkdln_data: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        Cleans LinkedIn data by extracting emojis, hashtags, and cleaning text.
        """
        try:
            if 'post_contents' not in linkdln_data:
                raise KeyError("Missing 'post_contents' column in LinkedIn data.")

            linkdln_data['emoji'] = self.extract_emojis(linkdln_data['post_contents'])
            linkdln_data['hashtags'] = self.extract_hashtags(linkdln_data['post_contents'])
            linkdln_data['post_contents'] = self.clean_text(linkdln_data['post_contents'])
            linkdln_data = self.split_at_first_delimiter(linkdln_data)
            linkdln_data['platform'] = 'linkedin'
            cleaner_logger.info(f"{platform} data cleaned successfully.")
            return linkdln_data

        except Exception as e:
            cleaner_logger.error(f"Error in {platform} cleaner: {e}")
            return pd.DataFrame()'''
