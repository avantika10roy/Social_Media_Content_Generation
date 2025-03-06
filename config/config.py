## ----- DONE BY PRIYAM PAL -----

import os
from pathlib import Path

class Config:
    
    """
    Configuration class for storing credentials, file paths, and URLs.
    It also provides a method to ensure required directories exist.
    """
    
    # CHROME DRIVER PATH
    CHROME_DRIVER_PATH                    = "/opt/homebrew/bin/chromedriver"
    
    CLIENT_SECRET_CREDENTIALS             = "./credentials/client_secret.json"
    
    # LINKEDIN CONFIGURATIONS
    LINKEDIN_USERNAME                     = "jit@itobuz.com"
    LINKEDIN_PASSWORD                     = "Abcd@1234"
    LINKEDIN_PROFILE_URL                  = "https://www.linkedin.com/company/itobuz-technologies-pvt-ltd/posts/?feedView=all"
    LINKEDIN_LOGIN_PAGE_LINK              = "https://www.linkedin.com/login"
    LINKEDIN_RAW_POST_DATA_PATH           = './data/raw_data/linkedin_raw_data.json'
    LINKEDIN_RAW_IMAGE_DATA_PATH          = './data/raw_data/linkedin_raw_images'
    LINKEDIN_RAW_IMAGE_DATA_LINK          = 'https://drive.google.com/drive/folders/1-PwFzDBfRFHWtadLEXubp-Kl2h8UquqG?usp=sharing'
    LINKEDIN_CLEANED_POST_DATA_PATH       = './data/cleaned_data/linkedin_cleaned_data.json'
    LINKEDIN_CLEANED_IMAGE_DATA_PATH      = './data/cleaned_data/linkedin_cleaned_images'
    LINKEDIN_PREPROCESSED_IMAGE_DATA_PATH = './data/preprocessed_data/linkedin_preprocessed_images'

    
    # INSTAGRAM CONFIGURATIONS
    INSTAGRAM_USERNAME                    = "itobuztechnologies"
    INSTAGRAM_RAW_POST_DATA_PATH          = "./data/raw_data/instagram_raw_data.json"
    INSTAGRAM_RAW_IMAGE_DATA_PATH         = "./data/raw_data/instagram_raw_images"
    INSTAGRAM_RAW_IMAGE_DATA_LINK         = "https://drive.google.com/drive/folders/1yt3LRxXKjp_U8U5sB5FFEAqLcFVhsElo?usp=sharing"
    INSTAGRAM_CLEANED_POST_DATA_PATH      = './data/cleaned_data/instagram_cleaned_data.json'
    INSTAGRAM_CLEANED_IMAGE_DATA_PATH     = './data/cleaned_data/instagram_cleaned_images'
    
    # FACEBOOK CONFIGURATIONS
    FACEBOOK_API                          = 'apify_api_yZXdrMa7P6SXvEKcdOpLwen6xmtQzX3Dr7t1'
    FACEBOOK_USERNAME                     = "/itobuz"
    FACEBOOK_RAW_POST_DATA_PATH           = "./data/raw_data/facebook_raw_data.json"
    FACEBOOK_RAW_IMAGE_DATA_PATH          = "./data/raw_data/facebook_raw_images"
    FACEBOOK_RAW_IMAGE_DATA_LINK          = "https://drive.google.com/drive/folders/19PqGEF74OJWERR1B9k4Oc6-hXzuhJZ4Y?usp=sharing"
    FACEBOOK_CLEANED_POST_DATA_PATH       = './data/cleaned_data/facebook_cleaned_data.json' 
    FACEBOOK_CLEANED_IMAGE_DATA_PATH      = './data/cleaned_data/facebook_cleaned_images'
    
    # CURATED DATA PATHS
    CURATED_POST_DATA_PATH                = './data/curated_data/curated_data.json'
    CURATED_IMAGE_DATA_PATH               = "./data/curated_data/curated_images"
    MIXED_CURATED_DATA_PATH               = "./data/mixed_curated/mixed_curated.json" 
    
    # PREPROCESSED DATA PATHS
    PREPROCESSED_POST_DATA_PATH           = './data/preprocessed_data/preprocessed_data.json'
    PREPROCESSED_IMAGE_DATA_PATH          = "./data/preprocessed_data/preprocessed_images"
    
    # AUGMENTED DATA PATHS
    AUGMENTED_NEW_DATA_PATH               = './data/augmented_data/new_data.json'
    AUGMENTED_IMAGES_PATH                 = './data/augmented_data/augmented_images'
    AUGMENTED_DATA_PATH                   = "./data/augmented_data/augmented_data.json"
    AUGMENTED_LOGO_PATH                   = "./data/augmented_data/logo_data.json"

    # LOGO IDENTIFICATION OUTPUT
    AUGMENTED_LOGO_RESULT                 = "./data/logo_identification_result/output_with_logo_info_and_uploads.json"
    LOGO_PATH                             = "data/logo.jpg"
    
    LOGO_INFO_OUTPUT_PATH                 = './data/logo_identification_result/output_with_logo_info_and_uploads.json'
    
    RANDOM_SEED                           = 42
    
    # BLIP IMAGE TO TEXT PATHS
    BLIP_OUTPUT_PATH                      = './data/Blip_with_context'
    BLIP_IMAGE_CONTEXT_DATA               = './data/Blip_with_context/blip_image_context.json'

    # Logo information 
    LOGO_INFO_OUTPUT_PATH                 = './data/logo_identification_result/output_with_logo_info_and_uploads.json'
    
    # FAST-API
    TEXT_GENERATION_API                   = "http://192.168.68.148:8000/generate_text"
    IMAGE_GENERATION_API                  = "https://mature-usually-impala.ngrok-free.app/generate"
    
    # RESPONSES PATH
    LLM_RESPONSE_JSON_FILE_PATH           = ".data/llm_response.json"
    IMAGE_RESPONSE_JSON_FILE_PATH         = ".data/image_response.json"

    @staticmethod
    def setup_directories():
        """
        Ensures that all required directories exist.
        If a directory does not exist, it creates it.
        """
        
        directories = [
            Config.LINKEDIN_POST_DATA_PATH,
            Config.LINKEDIN_IMAGE_DATA_PATH,
            os.path.dirname(Config.INSTAGRAM_IMAGE_DATA_PATH),
            os.path.dirname(Config.INSTAGRAM_POST_DATA_PATH)
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            else:
                print(f"Directory already exists: {directory}")
