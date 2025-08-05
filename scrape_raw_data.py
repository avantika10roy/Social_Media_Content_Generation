# DEPENDENCIES
import os
import sys
from datetime import datetime

# Add the parent directory to the system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import scraper modules
from config.config import Config
from src.utils.logger import LoggerSetup
from src.utils.data_saver import DataSaver
from src.scraper.linkedin_scraper import LinkedinScraper
from src.scraper.instagram_scraper import InstagramDataScraper
from src.scraper.facebook_scraper import FacebookScraper, FacebookDataProcessor

# Initialize logger
logger = LoggerSetup(logger_name="scraper_run", log_filename_prefix="scraper_main").get_logger()

def run_linkedin_scraper():
    """Run the LinkedIn scraper"""
    config = Config()
    logger.info("Starting LinkedIn scraper...")
    
    scraper = LinkedinScraper(
        username=config.LINKEDIN_USERNAME,
        password=config.LINKEDIN_PASSWORD,
        profile_url=config.LINKEDIN_PROFILE_URL
    )
    scraper.setup_driver(config.CHROME_DRIVER_PATH)
    df = scraper.linkedin_scraper()
    
    if not df.empty:
        DataSaver.data_saver(df, config.LINKEDIN_RAW_POST_DATA_PATH)
        logger.info(f"LinkedIn Raw Data saved to {config.LINKEDIN_RAW_POST_DATA_PATH}")
        logger.info(f"Total posts scraped: {len(df)}")
    else:
        logger.warning("No data was scraped from LinkedIn.")

def run_instagram_scraper():
    """Run the Instagram scraper"""
    logger.info("Starting Instagram scraper...")
    scraper = InstagramDataScraper()
    scraper.scrape()
    logger.info("Instagram scraping completed")

def run_facebook_scraper():
    """Run the Facebook scraper"""
    config = Config()
    logger.info("Starting Facebook scraper...")
    
    scraper = FacebookScraper(api_token=config.FACEBOOK_API, page_url=config.FACEBOOK_PAGE_URL, dataset_file= "facebook_data.json")
    scraper.scrape_data()
    
    processor = FacebookDataProcessor(
        dataset_file="facebook_data.json",
        csv_file="facebook_posts.csv",
        output_json=config.FACEBOOK_RAW_POST_DATA_PATH
    )
    processor.process_data()
    processor.download_images()
    processor.format_json()
    
    logger.info(f"Facebook Raw Data saved to {config.FACEBOOK_RAW_POST_DATA_PATH}")

def main():
    """Main function to run all scrapers"""
    logger.info("Running all scrapers...")
    #run_linkedin_scraper()
    #run_instagram_scraper()
    run_facebook_scraper()

if __name__ == "__main__":
    main()
