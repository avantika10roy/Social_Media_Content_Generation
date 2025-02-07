from config.config import Config
from src.data_preprocesser.image_preprocessing import ImagePreprocessor

def main():
    # Initialize the ImagePreprocessor class
    preprocessor = ImagePreprocessor(Config.LINKEDIN_RAW_IMAGE_DATA_PATH, Config.LINKEDIN_CLEANED_IMAGE_DATA_PATH)

    # Preprocess and save images
    preprocessor.preprocess_images()

if __name__ == '__main__':
    main()