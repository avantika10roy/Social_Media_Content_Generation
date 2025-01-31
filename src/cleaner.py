# DONE BY JIT
import emoji
import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataCleaner():
    """
    A class for preprocessing text data, including extracting emojis, extracting hashtags,
    and cleaning text by removing emojis, hashtags, HTML tags, extra spaces, and standardizing case.
    """
    
    def __init__(self):
        """
        Initializes the PreProcessor class.
        """
        logger.info("PreProcessor instance created.")
        pass
        
    def extract_emojis(self, text: pd.Series) -> pd.Series:
        """
        Extracts distinct emojis from each text entry in a pandas Series.
        
        Parameters:
        text (pd.Series): A pandas Series containing text data.
        
        Returns:
        pd.Series: A Series containing lists of extracted emojis per text entry.
        """
        logger.info("Extracting emojis from text.")
        return text.apply(emoji.distinct_emoji_list)

    def extract_hashtags(self, text: pd.Series) -> pd.Series:
        """
        Extracts hashtags from each text entry in a pandas Series.
        
        Parameters:
        text (pd.Series): A pandas Series containing text data.
        
        Returns:
        pd.Series: A Series containing lists of extracted hashtags per text entry.
        """
        logger.info("Extracting hashtags from text.")
        return text.apply(lambda x: re.findall(r"#\w+", x))
        
    def clean_text(self, text: pd.Series) -> pd.Series:
        """
        Cleans text by performing the following operations:
        - Removes emojis
        - Removes hashtags
        - Converts text to lowercase
        - Removes HTML tags
        - Removes newline characters
        - Removes extra dots
        - Removes extra spaces
        
        Parameters:
        text (pd.Series): A pandas Series containing text data.
        
        Returns:
        pd.Series: A cleaned pandas Series with processed text.
        """
        logger.info("Cleaning text data.")
        
        # Remove emojis
        logger.debug("Removing emojis.")
        emoji_text = text.apply(lambda x: emoji.replace_emoji(x, replace=''))

        # Remove hashtags
        logger.debug("Removing hashtags.")
        hastag_removed_text = emoji_text.apply(lambda x: re.sub(r"#\w+", "", x).strip())

        # Lowercasing
        logger.debug("Converting text to lowercase.")
        lowered_text = hastag_removed_text.str.lower()

        # Remove HTML tags
        logger.debug("Removing HTML tags.")
        html_tags_removed_text = lowered_text.apply(lambda x: re.sub('<[^>]*>', '', x))

        # Remove '\n'
        logger.debug("Removing newline characters.")
        removed_n_text = html_tags_removed_text.apply(lambda x: re.sub('\n', '', x))

        # Remove extra dots
        logger.debug("Removing extra dots.")
        removed_extra_dots_text = removed_n_text.apply(lambda x: re.sub(r'\.{2,}', '', x))

        # Remove extra spaces
        logger.debug("Removing extra spaces.")
        cleaned_text = removed_extra_dots_text.apply(lambda x: re.sub(r'\s+', ' ', x))
        
        logger.info("Text cleaning complete.")
        return cleaned_text