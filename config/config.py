## ----- DONE BY PRIYAM PAL -----

import os

class Config:
    
    """
    Configuration class for storing credentials, file paths, and URLs.
    It also provides a method to ensure required directories exist.
    """
    
    # CHROME DRIVER PATH
    CHROME_DRIVER_PATH           = "/opt/homebrew/bin/chromedriver"
    
    # LINKEDIN CONFIGURATIONS
    LINKEDIN_USERNAME            = "jit@itobuz.com"
    LINKEDIN_PASSWORD            = "Abcd@1234"
    LINKEDIN_POST_DATA_FILENAME  = 'linkedin_post_data.json'
    LINKEDIN_POST_DATA_PATH      = './data/raw'
    LINKEDIN_IMAGE_DATA_PATH     = './data/raw/linkedin_images'
    LINKEDIN_LOGIN_PAGE_LINK     = "https://www.linkedin.com/login"
    LINKEDIN_PROFILE_URL         = "https://www.linkedin.com/company/itobuz-technologies-pvt-ltd/posts/?feedView=all"
    
    # INSTAGRAM CONFIGURATIONS
    INSTAGRAM_USERNAME           = "itobuztechnologies"
    INSTAGRAM_IMAGE_DATA_PATH    = "./data/raw_data/instagram_raw_images"
    INSTAGRAM_POST_DATA_PATH     = "./data/raw_data/instagram_raw_data.json"

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
