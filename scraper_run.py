## ----- DONE BY PRIYAM PAL AND SUBHAS MUKHERJEE -----

# DEPENDENCIES
import sys
import logging
from pathlib import Path
from datetime import datetime
from config import PROFILE_URL
from src.logger import LoggerSetup
from config import LINKEDIN_USERNAME 
from config import LINKEDIN_PASSWORD
from config import CHROME_DRIVER_PATH
from src.web_scraper import SocialMediaScraper


# LOGGING SETUP
main_logger = LoggerSetup(logger_name = "scraper_run.py", log_filename_prefix = "scraper_main").get_logger()

def main():
    """
    Main function to run the LinkedIn scraper.
    Handles the setup, execution, and error handling of the scraping process.
    
    """
    
    try:
        main_logger.info("Initializing LinkedIn scraper...")
        scraper         = SocialMediaScraper(username     = LINKEDIN_USERNAME, 
                                             password     = LINKEDIN_PASSWORD, 
                                             profile_url  = PROFILE_URL)

        main_logger.info("Setting up Chrome driver...")
        scraper.setup_driver(CHROME_DRIVER_PATH)

        main_logger.info("Starting LinkedIn scraping process...")
        df               = scraper.linkedin_scraper()

        if not df.empty:
            timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')

            main_logger.info(f"Total posts scraped: {len(df)}")

        else:
            main_logger.warning("No data was scraped.")

    except Exception as e:
        main_logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main_logger.info("Starting script execution...")
    main()
    main_logger.info("Script execution completed.")