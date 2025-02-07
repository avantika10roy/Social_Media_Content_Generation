from config.config import Config  # Import the Config class to get the paths
from src.data_preprocesser.image_preprocessing import ImagePreprocessor  # Import the ImagePreprocessor class

def main():
    # Initialize the ImagePreprocessor class with the paths from Config
    preprocessor = ImagePreprocessor(
        Config.LINKEDIN_RAW_IMAGE_DATA_PATH,  # Access the raw image data path from Config
        Config.LINKEDIN_PREPROCESSED_IMAGE_DATA_PATH  # Access the cleaned image data path from Config
    )

    # Preprocess and save images
    preprocessor.preprocess_images()

if __name__ == '__main__':
    main()
