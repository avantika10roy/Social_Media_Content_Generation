# Done by Amrit Bag

#Dependencies
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextPreprocessing:
    """
    This class performs text preprocessing through cleaning, tokenization, and normalization

    Attributes:
    -----------
    lemmatizer: WordNetLemmatizer instance for lemmatization
    stop_words: Set of English stop words
    """
    def __init__(self):
        """
        Initialize the TextPreprocessor with required NLTK resources

        Raises:
        -------
        LookupError : If required NLTK resources cannot be downloaded
        """
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)

            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))

        except LookupError as e:
            raise

    def clean_text(self, text: str) -> list:
        """
        Clean and normalize input text by removing HTML tags, special characters,
        and applying text normalization techniques

        Arguments:
        ----------
        text {str} : Input text to be cleaned

        Raises:
        -------
        ValueError        : If input text is None or empty
        Exception         : If any error occurs during text cleaning

        Returns:
        --------
        {list} : List of cleaned and lemmatized tokens
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text should not be an empty string.")

        try:
            # Remove HTML tags
            text = re.sub('<[^>]*>', '', text)  
            # Remove special characters
            text = re.sub('[^a-zA-Z\s]', '', text)  
            text = text.lower()
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            return tokens 

        except Exception as e:
            raise Exception(f"Text cleaning error: {e}")

    def save_to_csv(self, text_list, filename="cleaned_text.csv"):
        """
        Save cleaned tokens to a CSV file

        Arguments:
        ----------
        text_list {list} : List of text samples to be tokenized and saved
        filename {str}   : Name of the CSV file to save the cleaned tokens
        """
        cleaned_texts = [self.clean_text(text) for text in text_list]
        df = pd.DataFrame({"Tokenized_Text": cleaned_texts})
        df.to_csv(filename, index=False)
        print(f"Cleaned tokens saved to {filename}")
