# Done by Amrit Bag

# Dependencies
import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from logger import LoggerSetup

# LOGGING SETUP
preprocessor_logger = LoggerSetup(logger_name = "text_preprocessor.py", log_filename_prefix = "text_preprocessor").get_logger()
preprocessor_logger.info("Logger successfully initialized.")  # Test log entry

class TextPreprocessing:
    """
    This class performs text preprocessing through cleaning, tokenization, and normalization.

    Attributes:
    -----------
    lemmatizer: WordNetLemmatizer instance for lemmatization.
    stop_words: Set of English stop words.
    """

    def __init__(self):
        """
        Initialize the TextPreprocessor with required NLTK resources.

        Raises:
        -------
        LookupError : If required NLTK resources cannot be downloaded.
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

    def clean_text(self, text: str) -> list:
        """
        Clean and normalize input text by removing HTML tags, special characters,
        and applying text normalization techniques.

        Arguments:
        ----------
        text {str} : Input text to be cleaned.

        Raises:
        -------
        ValueError        : If input text is None or empty.
        Exception         : If any error occurs during text cleaning.

        Returns:
        --------
        {list} : List of cleaned and lemmatized tokens.
        """
        if not text or not isinstance(text, str):
            preprocessor_logger.warning("Received an empty or invalid string for text cleaning.")
            raise ValueError("Input text should not be an empty string.")

        try:
            preprocessor_logger.info("Cleaning text: %s", text[:50])  # Log first 50 characters of input text
            # Remove HTML tags
            text = re.sub('<[^>]*>', '', text)
            # Remove special characters
            text = re.sub('[^a-zA-Z\s]', '', text)
            text = text.lower()
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]

            preprocessor_logger.info("Text successfully cleaned and tokenized.")
            return tokens

        except Exception as e:
            preprocessor_logger.error(f"Text cleaning error: {e}", exc_info=True)
            raise Exception(f"Text cleaning error: {e}")

    def save_to_csv(self, text_list, filename="cleaned_text.csv"):
        """
        Save cleaned tokens to a CSV file.

        Arguments:
        ----------
        text_list {list} : List of text samples to be tokenized and saved.
        filename {str}   : Name of the CSV file to save the cleaned tokens.
        """
        try:
            preprocessor_logger.info(f"Saving cleaned tokens to {filename}...")
            cleaned_texts = [self.clean_text(text) for text in text_list]
            df = pd.DataFrame({"Tokenized_Text": cleaned_texts})
            df.to_csv(filename, index=False)

            preprocessor_logger.info(f"Cleaned tokens successfully saved to {filename}.")
            print(f"Cleaned tokens saved to {filename}")

        except Exception as e:
            preprocessor_logger.error(f"Error while saving to CSV: {e}", exc_info=True)
            raise Exception(f"Error while saving to CSV: {e}")
        
        


