## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

import os
import sys

from config.config import Config

from src.utils.data_saver import DataSaver
from src.utils.logger import LoggerSetup
from src.scraper.linkedin_scraper import LinkedinScraper
from src.scraper.instagram_scraper import InstagramDataScraper


# LOGGING SETUP
main_logger              = LoggerSetup(logger_name = "scraper_run.py", log_filename_prefix = "scraper_main").get_logger()

def main():
    """
    Main function to run the All Platform scraper.
    Handles the setup, execution, and error handling of the scraping process.
    
    """
    
    # ----- LINKEDIN SCRAPER -----
    
    try:
        main_logger.info("Initializing LinkedIn scraper...")
        linkedin_scraper = LinkedinScraper(username     = Config.LINKEDIN_USERNAME, 
                                           password     = Config.LINKEDIN_PASSWORD, 
                                           profile_url  = Config.LINKEDIN_PROFILE_URL
                                           )

        main_logger.info("Setting up Chrome driver...")
        linkedin_scraper.setup_driver(Config.CHROME_DRIVER_PATH)

        main_logger.info("Starting LinkedIn scraping process...")
        df               = linkedin_scraper.linkedin_scraper()
        
        DataSaver.data_saver(df, Config.LINKEDIN_POST_DATA_PATH)
        main_logger.info(f"LinkedIn Raw Data saved to {Config.LINKEDIN_POST_DATA_PATH}")

        if not df.empty:
            main_logger.info(f"Total posts scraped: {len(df)}")

        else:
            main_logger.warning("No data was scraped.")

    except Exception as e:
        main_logger.error(f"Error Occured in Scraping Instagram Data: {str(e)}", exc_info = True)
        sys.exit(1)
        
        
    # ----- INSTAGRAM SCRAPER -----    
    try:
        main_logger.info("Initailizing the Instagram Scraper")
        instagram_scraper = InstagramDataScraper()
        
        main_logger.info("Starting Instagram Scraping Process...")
        instagram_scraper.instagram_scraper()
        
    except Exception as e:
        main_logger.error(f"Error Occured in Scraping Instagram Data: {repr(e)}", exc_info = True)
        sys.exit(1)

if __name__ == "__main__":
    main_logger.info("Starting script execution...")
    main()
    main_logger.info("Script execution completed.")