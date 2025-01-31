## ----- DONE BY PRIYAM PAL AND SUBHAS MUKHERJEE -----

# DEPENDENCIES

import sys
import logging
from pathlib import Path
from datetime import datetime

from config import PROFILE_URL
from config import LINKEDIN_USERNAME 
from config import LINKEDIN_PASSWORD
from config import CHROME_DRIVER_PATH
from src.web_scraper import SocialMediaScraper

# LOGGING SETUP
log_dir   = Path('logs')
log_dir.mkdir(exist_ok=True)

log_file  = log_dir / f'linkedin_post_scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

FORMAT    = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"

logging.basicConfig(level     = logging.INFO,
                    format    = FORMAT,
                    handlers  = [logging.FileHandler(log_file),  
                                 logging.StreamHandler(sys.stdout) 
                                 ]
                    )

logger    = logging.getLogger(__name__)

def main():
    """
    Main function to run the LinkedIn scraper.
    Handles the setup, execution, and error handling of the scraping process.
    
    """
    
    try:
        logger.info("Initializing LinkedIn scraper...")
        scraper         = SocialMediaScraper(username     = LINKEDIN_USERNAME, 
                                             password     = LINKEDIN_PASSWORD, 
                                             profile_url  = PROFILE_URL)

        logger.info("Setting up Chrome driver...")
        scraper.setup_driver(CHROME_DRIVER_PATH)

        logger.info("Starting LinkedIn scraping process...")
        df               = scraper.linkedin_scraper()

        if not df.empty:
            timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')

            logger.info(f"Total posts scraped: {len(df)}")

        else:
            logger.warning("No data was scraped.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting script execution...")
    main()
    logger.info("Script execution completed.")
