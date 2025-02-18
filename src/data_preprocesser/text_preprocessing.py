import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.utils.logger import LoggerSetup 

# LOGGING SETUP
preprocessor_logger = LoggerSetup(logger_name="text_preprocessor.py", log_filename_prefix="text_preprocessor").get_logger()
preprocessor_logger.info("Logger successfully initialized.")

class TextPreprocessing:
    """
    This class performs text preprocessing, including cleaning, tokenization, and normalization.
    """
    
    def __init__(self):
        """
        Initialize the TextPreprocessing class with required NLTK resources.
        """
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)

            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))

            preprocessor_logger.info("NLTK resources successfully downloaded and initialized.")
        except LookupError as e:
            preprocessor_logger.error(f"Failed to download NLTK resources: {e}", exc_info=True)
            raise

    def text_preprocess(self, text: str) -> list:
        """
        Clean and normalize input text by removing HTML tags, special characters, and stopwords.
        
        Arguments:
        ----------
        text {str} : Input text to be cleaned.

        Returns:
        --------
        {list} : List of cleaned and lemmatized tokens.
        """
        if not text or not isinstance(text, str):
            preprocessor_logger.warning("Received an empty or invalid string for text cleaning.")
            #raise ValueError("Input text should not be an empty string.")
            pass

        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]*>', '', text)
            # Remove special characters
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Tokenization
            tokens = word_tokenize(text)
            # Lemmatization and stopword removal
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.lower() not in self.stop_words]

            preprocessor_logger.info("Text successfully cleaned and tokenized.")
            return tokens
        except Exception as e:
            preprocessor_logger.error(f"Text cleaning error: {e}", exc_info=True)
            raise Exception(f"Text cleaning error: {e}")
    
    
    def preprocess(self, curated_data: pd.DataFrame, text_column: str = 'post_content') -> pd.DataFrame:
        """
        Cleans a DataFrame by extracting cleaned tokens into a new column `tokenized_text`.
        
        Arguments:
        ----------
        curated_data {pd.DataFrame} : Input DataFrame with a text column.
        text_column {str} : The name of the column to clean (default: 'post_content').

        Returns:
        --------
        {pd.DataFrame} : Processed DataFrame with a new column `tokenized_text`.
        """
        try:
            if text_column not in curated_data.columns:
                raise KeyError(f"Missing '{text_column}' column in DataFrame.")

            preprocessor_logger.info(f"Preprocessing '{text_column}' column...")

            # Create a new column for tokenized text
            curated_data["tokenized_text"] = curated_data[text_column].apply(self.text_preprocess)

            preprocessor_logger.info("Text preprocessing completed successfully.")
            return curated_data
        
        except Exception as e:
            preprocessor_logger.error(f"Error in text preprocessing: {e}")
            return pd.DataFrame()