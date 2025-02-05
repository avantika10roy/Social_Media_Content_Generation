## IT WILL CLEAN THE RAW DATA AND SAVE IT TO data/cleaned_data/ and withing their corresponding sub-folders

from src.data_cleaner.cleaner import DataCleaner
import os
import pandas as pd

class DataCleaning():

    def __init__(self):
        """Initializes the DataCleaning class."""
        self.data_cleaner = DataCleaner()
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
            platform_dir = os.path.join(self.cleaned_data_dir, platform)
            os.makedirs(platform_dir,exist_ok=True)

            if not file_name.endswith('.json'):
                file_name += '.json'

            cleaned_data.to_json(os.path.join(platform_dir,file_name),orient='records',indent=4,force_ascii=False)
            print(f"Cleaned data saved to {platform_dir}/{file_name}")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")


    def clean_and_save_data(self, file_path: str, platform: str, file_name: str):
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
        
        except Exception as e:
            print(f"Error in cleaning and saving data: {e}")

# Example Usage
if __name__ == "__main__":
   
    data_cleaner = DataCleaning()
    
   
    raw_file_path = "data/raw_data/linkedin_raw_data.json"  
    platform = "linkedin" 
    file_name = "linkedin_cleaned_data.json"  
    
    # Clean and save the data
    data_cleaner.clean_and_save_data(raw_file_path, platform, file_name)



