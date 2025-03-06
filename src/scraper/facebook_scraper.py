# ---------Done by Anjisnu Roy and Arnab Chatterjee-----------
#-----------------Refactored by Avantika Roy------------------

# DEPENDENCIES
import os
import sys
import csv
import json
import requests
import pandas as pd
from apify_client import ApifyClient

# Add the parent directory to the system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.config import Config
from src.utils.logger import LoggerSetup

facebook_logger = LoggerSetup(logger_name = "facebook_web_scraper.py", log_filename_prefix = "facebook").get_logger()

cf = Config()

class FacebookScraper:
    """ Class to scrape company posts from Facebook. """
    def __init__(self, api_token: str, page_url: str, results_limit: int = 500, dataset_file: str ="facebook_data.json"):
        """
            Constructor for the class FacebookScraper:
            
            Arguments:
            ----------
                - api_token     : Token to access Apify API.
                - page_url      : URL for the Facebook Page.
                - results_limit : Limits how many posts are fetched and manage API rate limits.
                - dataset_file  : JSON file where the result is stored in dictionary format.
        """
        self.client        = ApifyClient(api_token)  # API key
        self.page_url      = page_url                # Facebook page URL
        self.results_limit = results_limit           # Max posts to fetch
        self.dataset_file  = dataset_file            # File path to store scraped data in JSON format

        facebook_logger.info(f"Initialized FacebookDataScraper for page: {self.page_url}")
#--------------------------------------------------------------------------------------

    def scrape_data(self):
        """
            Function to scrape facebook posts using Apify Facebook Posts Scraper API.

            Returns:
            --------
                - None

            Raises:
            -------
                - FacebookScrapingError : If an error occurs during the scraping process.

        """
        try:
            # Input parameters for API scraper
            run_input  = {"startUrls"     : [{"url": self.page_url}], # URL of the Facebook page to scrape
                            "resultsLimit" : self.results_limit}      # Limits the number of posts to fetch
            
            # Call API and start scraping process
            run        = self.client.actor("apify/facebook-posts-scraper").call(run_input=run_input)
            
            # Retrieve dataset ID from API Response
            dataset_id = run.get("defaultDatasetId")

            # Check for valid dataset ID
            if not dataset_id:
                facebook_logger.info("No dataset ID found in API response.")
                return                    # Exit function if no dataset ID found
            
            # Fetch scraped data from dataset
            items      = list(self.client.dataset(dataset_id).iterate_items())

            # Extract file saving logic from a different method
            # Save scraped data to JSOn file using save_json method
            self.save_json(items)

        # Raise exception
        except Exception as FacebookScrapingError:
            facebook_logger.exception(f"{FacebookScrapingError}: Error while scraping facebook data.")

#--------------------------------------------------------------------------------------

    def save_json(self, items:str) -> json:
        """
            Function to save scraped data to a JSON file.

            Arguments:
            ----------
                - items (list) : A list of scraped Facebook post data.

            Returns:
            --------
                - (json)       : Saves scraped data from facebook into JSON file.

            Raises:
                - SavingError  : If an error occurs while saving the data.
        """
        try:
            # Open JSON file with utf-8 encoding
            with open(self.dataset_file, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii = False, indent = 4) # Save data in readable format
                
            facebook_logger.info(f"Data saved to {self.dataset_file}")

        except Exception as SavingError:    # Catch exceptions if occurred
            facebook_logger.exception(f"{SavingError} Failed to save data to: {self.dataset_file}")

#--------------------------------------------------------------------------------------

class FacebookDataProcessor:
    def __init__(self, dataset_file: str, csv_file: str, output_json: str, image_dir: str ="./facebook_raw_images"):
        """
            Constructor for class FacebookDataProcessor:

            Arguments:
            ----------
                - dataset_file : The JSON file created using FacebookScraper.
                - csv_file     : JSON file converted and stored in csv format.
                - output_json  : Formatted data is stored in JSON file.
                - image_dir    : Path to directory to store images (directory created if not already present).
        """
        self.dataset_file = dataset_file             # input dataset
        self.csv_file     = csv_file                 # converted csv from json
        self.output_json  = output_json              # formatted output json 
        self.image_dir    = image_dir                # path to image directory

        # Ensure directory exists; create if it does not
        os.makedirs(self.image_dir, exist_ok=True)

#--------------------------------------------------------------------------------------

    def load_json(self) -> json:
        """
            Function to load JSON file.

            Returns:
            --------
                - (json)       : Loads the raw JSON file.

            Raises:
            -------
                - LoadingError : If error occurs while loading the JSON file.
        """
        try:
            # Load/Open input dataset
            with open(self.dataset_file, 'r') as file:
                data = json.load(file)
            facebook_logger.info("Successfully loaded json.")
        
        except Exception as LoadingError: # Catch exception if occurred
            facebook_logger.exception(f"Failed to load json: {LoadingError}")

#--------------------------------------------------------------------------------------

    def write_to_csv(self, data: json) -> csv:
        """
            Function to write csv file.

            Arguments:
            ----------
                - data       : JSON file with facebook raw data.

            Returns:
            --------
                - (csv)      : CSV file where converted JSON is stored.

            Raises:
            -------
                - CSVError   : If error occurs while converting to CSV. 
        """
        try:
            field_names = ['post_url', 'image_url', 'caption']   # Names of the fields in the formatted json dataset

            # Open CSV file in write mode with utf-8 encoding
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer  = csv.DictWriter(csvfile, fieldnames=field_names)   # Initialize CSV Writer
                writer.writeheader()      # Write the header row to the CSV file
            
            # Iterate through each post in scraped data
            for post in data:
                post_url   = post.get('url', 'NA')            # Post URL, 'NA' if missing
                caption    = post.get('text', 'No Caption')   # Caption of the post, 'No Caption' if missing

                media_list = post.get('media', [])            # Extract media list from post

                # If no media is found, keep the 'image_url' cell empty
                if not media_list:
                    writer.writerow({'post_url':post_url, 'image_url': '', 'caption': caption})
                
                else:
                    # If media is present, iterate through each media item and estract image URL
                    for media in media_list:
                        image_url = media.get('image', {}).get('uri', '')
                        writer.writerow({'post_url':post_url, 'image_url': image_url, 'caption': caption})

        except Exception as CSVError:           # Raise exception if occurred
            facebook_logger.exception(f"Unable to convert to CSV: {CSVError}")

#--------------------------------------------------------------------------------------

    def process_data(self):
        """
            Function that combines load_json() function and write_to_csv() function.

            Returns:
            --------
                - csv             : Converted CSV file.

            Raises:
            -------
                - ProcessingError : If error occurs while executing the function.
        """
        try:
            data = self.load_json()    # Load JSON
            self.write_to_csv(data)    # Convert JSON to CSV
            facebook_logger.info("CSV file created successfully!")

        except Exception as ProcessingError:   # Raise Exception if occurred
            facebook_logger.exception(f"Error processing data: {ProcessingError}")

#--------------------------------------------------------------------------------------
    
    def downloader(self, image_url: str, index: int) -> str:
        """
            Downloader function to download an image from the url and save it locally.

            Arguments:
            ----------
                - image_url        : URL of the image to download.
                - index            : Index used to generate file name.

            Returns:
            --------
                - (str) image_name : Name of the image saved after download, otherwise None.

            Raises:
            -------
                - RequestError     : If the request fails or encounters an error.
        """
        try:
            # Send a GET request to fetch the image, with streaming enabled and a 10-second timeout
            response = requests.get(image_url, stream=True, timeout=10)

            # Check if the request was successful
            if response.status_code == 200:
                image_name = f"image_{index}.jpg"      # Create a unique filename for each image downloaded
                image_path = os.path.join(self.image_dir, image_name)

                # Open the file in binary write mode and save the image in chunks
                with open(image_path, "wb") as file:
                    for chunk in response.iter_contents(1024):  # Read in 1KB chunks
                        file.write(chunk)
                return image_name    # Return the saved image filename
            
            else:
                facebook_logger.error(f"Failed to download image {image_url}: HTTP {response.status_code}")

        except Exception as RequestError:    # Catch exceptions if occurred
            facebook_logger.exception(f"Failed to perform requested task: {RequestError}")

#--------------------------------------------------------------------------------------

    def download_images(self):
        """
            Reads a CSV file, downloads images from the URLs in the 'image_url' column, 
            and updates the CSV with the corresponding image filenames.

            Returns:
            --------
                - (csv)         : Updated CSV file.

            Raises:
            -------
                - DownloadError : If an error occurs during the download process.
        """
        try:
            # Load CSV file
            df = pd.read_csv(self.csv_file)
            
            # Check if the "image_url" column exists
            if "image_url" not in df.columns:
                facebook_logger.info("CSV does not contain column named 'image_url'.")
                return

            # Drop the "post_url" column if it exists, ignoring errors if it's not found
            df = df.drop(columns=['post_url'], errors='ignore', inplace=True)
            
            # Download images and store their filenames in a new "image_name" column
            df['image_name'] = [self.downloader(url, index) if pd.notna(url) and url.strip() else None
                                for index, url in enumerate(df['image_name'])]
            
            # Save the updated DataFrame back to the CSV file
            df.to_csv(self.csv_file, index=False)
            facebook_logger.info(f"Images downloaded and CSV updated: {self.csv_file}")

        except Exception as DownloadError: # Catch exception if occurred
            facebook_logger.exception(f"Error downloading images: {DownloadError}")

#--------------------------------------------------------------------------------------
    
    def format_json(self) -> json:
        """
            Formats the CSV file into a structured JSON file for social media content generation task.

            Returns:
            --------
                - (json)          : Formatted JSON file.

            Raises:
            -------
                - FormattingError : If an error occurs during the formatting or file-writing process.

        """
        try:
            # Load the CSV file
            df = pd.read_csv(self.csv_file)

            # Define required columns for processing
            required_columns = {'caption', 'image_url', 'image_name'}

            # Check if all required columns are present in the DataFrame
            if not required_columns.issubset(df.columns):
                facebook_logger.error(f"Missing required columns in CSV: {required_columns - set(df.columns)}")
                return        # Exit function if required columns are missing

            # Group by 'caption', aggregating image URLs and image file paths
            grouped_data = df.groupby("caption", dropna=True).agg(
                                        image_URLs=("image_url", lambda x: ", ".join(x.dropna())),
                                        image_paths=("image_name", lambda x: ", ".join(
                                                                [os.path.join(self.image_dir, name) for name in x.dropna()]
                                                                ))
                                    ).reset_index()
            
            # Rename 'caption' column to 'post_contents' for clarity
            grouped_data.rename(columns={"caption": "post_contents"}, inplace=True)

            # Convert DataFrame to a list of dictionaries
            json_data = grouped_data.to_dict(orient="records")

            # Save processed data as a JSON file
            with open(self.output_json, "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, indent=4, ensure_ascii=False)
            facebook_logger.info(f"JSON saved as {self.output_json}")

        except Exception as FormattingError:   # Raise Exception if occurred
            facebook_logger.info(f"Error formatting JSON: {FormattingError}")

