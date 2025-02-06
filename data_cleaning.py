## IT WILL CLEAN THE RAW DATA AND SAVE IT TO data/cleaned_data/ and withing their corresponding sub-folders

from src.data_cleaner.data_cleaner import DataCleaner, ImageDataCleaning
from src.utils.data_saver import DataSaver
import os
import pandas as pd

class DataCleaning():

    def __init__(self, img_dir, clean_dir, inv_image_dir):
        """Initializes the DataCleaning class."""
        self.data_cleaner = DataCleaner()
        self.saver = DataSaver()
        self.image_cleaner = ImageDataCleaning(img_dir,clean_dir,inv_image_dir)
        self.cleaned_data_dir = "data/cleaned_data/"

    
    def load_raw_data(self,file_path:str) -> pd.DataFrame:
        """
        Load raw data from a CSV file (or any other format).
        
        Parameters:
        -----------
        file_path (str): The path to the raw data file.
        
        Returns:
        --------
        pd.DataFrame: The loaded raw data as a DataFrame.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Raw data not found in {file_path}")
            
            raw_data = pd.read_json(file_path)
            return raw_data
        except Exception as e:
            print(f"error in loading file: {e}")
            return pd.DataFrame()

    def save_clean_data(self,platform:str, cleaned_data: pd.DataFrame, file_name:str):
        """
        Save cleaned data to the appropriate folder.
        
        Parameters:
        -----------
        platform (str): The name of the platform used (Ex - Instagram, Linkdln and Facebook)
        cleaned_data (pd.DataFrame): The cleaned data.
        file_name (str): The name of the file to save (e.g., 'cleaned_data.csv').
        """

        try:
            #platform_dir = os.path.join(self.cleaned_data_dir, platform)
            cleaned_data_dir = self.cleaned_data_dir
            os.makedirs(cleaned_data_dir,exist_ok=True)

            if not file_name.endswith('.json'):
                file_name += '.json'


            self.saver.raw_data_saver(cleaned_data, os.path.join(cleaned_data_dir,file_name))
            print(f"Cleaned data saved to {cleaned_data_dir}/{file_name}")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")


    def clean_save_data(self, file_path: str, platform: str, file_name: str):
        """
        Clean the raw data and save the cleaned data.
        
        Parameters:
        -----------
        file_path (str): The path to the raw data file.
        platform (str): The platform name (e.g., 'instagram', 'linkedin').
        file_name (str): The name of the file to save the cleaned data as.
        """
        try:
            # Step 1: Load the raw data
            raw_data = self.load_raw_data(file_path)
            
            if raw_data.empty:
                print("No data to clean.")
                return
            
            # Step 2: Clean the data based on the platform
            if platform.lower() == "instagram":
                cleaned_data = self.data_cleaner.instagram_clean_data(raw_data, platform)
            elif platform.lower() == "linkedin":
                cleaned_data = self.data_cleaner.linkdln_clean_data(raw_data, platform)
            else:
                print(f"Unsupported platform: {platform}")
                return
            
            # Step 3: Save the cleaned data

            self.save_clean_data(platform, cleaned_data, file_name)

            cleaned_data = self.image_cleaner.is_image_valid(cleaned_data)
            
            #self.image_cleaner.remove_duplicates(cleaned_data)

        
        except Exception as e:
            print(f"Error in cleaning and saving data: {e}")


# Example Usage
if __name__ == "__main__":
   
    #data_cleaner = DataCleaning()
    
    img_directory = "/Users/it012305/Desktop/linkedin_images"  
    cleaned_img_directory = "/Users/it012305/Desktop/linkedin_images/cleaned_images/"  
    invalid_img_directory = "/Users/it012305/Desktop/linkedin_images/invalid_images/" 

    raw_file_path = "data/raw_data/linkedin_raw_data.json"  
    platform = "linkedin" 
    file_name = "linkedin_cleaned_data.json"

    data_cleaner = DataCleaning(img_directory, cleaned_img_directory, invalid_img_directory)

    raw_file_path = "data/raw_data/instagram_raw_data.json"  
    platform = "instagram" 
    file_name = "instagram_cleaned_data.json"
    
    # Clean and save the data
    data_cleaner.clean_save_data(raw_file_path, platform, file_name)


'''from src.data_cleaner.data_cleaner import DataCleaner, ImageDataCleaning
from src.utils.data_saver import DataSaver
import os
import pandas as pd

class DataCleaning():
    def __init__(self, img_dir, clean_dir, inv_image_dir):
        """Initializes the DataCleaning class."""
        self.data_cleaner = DataCleaner()
        self.saver = DataSaver()
        self.image_cleaner = ImageDataCleaning(img_dir, clean_dir, inv_image_dir)
        self.cleaned_data_dir = "data/cleaned_data/"
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw data from a JSON file.
        
        Parameters:
        -----------
        file_path (str): The path to the raw data file.
        
        Returns:
        --------
        pd.DataFrame: The loaded raw data as a DataFrame.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Raw data not found in {file_path}")
            
            raw_data = pd.read_json(file_path)
            required_columns = ['image_urls', 'post_content']
            missing_columns = [col for col in required_columns if col not in raw_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in raw data: {missing_columns}")
        
            return raw_data
        except Exception as e:
            print(f"Error in loading file: {e}")
            return pd.DataFrame()

    def save_clean_data(self, platform: str, cleaned_data: pd.DataFrame, file_name: str):
        """
        Save cleaned data to the appropriate folder.
        
        Parameters:
        -----------
        platform (str): The name of the platform used (Ex - Instagram, LinkedIn, Facebook).
        cleaned_data (pd.DataFrame): The cleaned data.
        file_name (str): The name of the file to save (e.g., 'cleaned_data.json').
        """
        try:
            cleaned_data_dir = self.cleaned_data_dir
            os.makedirs(cleaned_data_dir, exist_ok=True)
            
            # Ensure the file has a .json extension
            if not file_name.endswith('.json'):
                file_name += '.json'

            # Save the cleaned data using the DataSaver utility
            self.saver.raw_data_saver(cleaned_data, os.path.join(cleaned_data_dir, file_name))
            print(f"Cleaned data saved to {cleaned_data_dir}/{file_name}")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")

    def clean_save_data(self, file_path: str, platform: str, file_name: str):
        """
        Clean the raw data, process images, and save the cleaned data.
        
        Parameters:
        -----------
        file_path (str): The path to the raw data file.
        platform (str): The platform name (e.g., 'instagram', 'linkedin').
        file_name (str): The name of the file to save the cleaned data as.
        """
        try:
            # Step 1: Load the raw data
            raw_data = self.load_raw_data(file_path)
            
            if raw_data.empty:
                print("No data to clean.")
                return
            
            # Step 2: Clean the data based on the platform
            if platform.lower() == "instagram":
                cleaned_data = self.data_cleaner.instagram_clean_data(raw_data, platform)
            elif platform.lower() == "linkedin":
                cleaned_data = self.data_cleaner.linkdln_clean_data(raw_data, platform)
            else:
                print(f"Unsupported platform: {platform}")
                return
            
            # Step 3: Filter out invalid images based on keywords
            cleaned_data = self.image_cleaner.is_image_valid(cleaned_data)
            
            # Step 4: Remove duplicate images
            self.image_cleaner.remove_duplicates()
            
            # Step 5: Move valid images to the clean directory
            self.image_cleaner._move_valid_images(cleaned_data)
            
            # Step 6: Save the cleaned data (metadata)
            self.save_clean_data(platform, cleaned_data, file_name)
        
        except Exception as e:
            print(f"Error in cleaning and saving data: {e}")

# Example Usage
if __name__ == "__main__":
    # Define directories for images
    img_directory = "/Users/it012305/Desktop/linkedin_images"
    cleaned_img_directory = "/Users/it012305/Desktop/linkedin_images/cleaned_images/"
    invalid_img_directory = "/Users/it012305/Desktop/linkedin_images/invalid_images/"
    
    # Define raw data file paths
    linkedin_raw_file_path = "data/raw_data/linkedin_raw_data.json"
    instagram_raw_file_path = "data/raw_data/instagram_raw_data.json"
    
    # Initialize the DataCleaning class
    data_cleaner = DataCleaning(img_directory, cleaned_img_directory, invalid_img_directory)
    
    # Clean and save LinkedIn data
    data_cleaner.clean_save_data(linkedin_raw_file_path, "linkedin", "linkedin_cleaned_data.json")
    
    # Clean and save Instagram data
    data_cleaner.clean_save_data(instagram_raw_file_path, "instagram", "instagram_cleaned_data.json")'''