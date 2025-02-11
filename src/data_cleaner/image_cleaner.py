import os
import shutil
import pandas as pd

class ImageCleaner:
    def __init__(self, keywords=None):
        """
        Initialize the ImageFilterCopier class.
        
        Parameters:
        keywords (list): List of keywords to filter out posts. Default keywords provided.
        """
        self.keywords = keywords or ['celebrated', 'celebration', 'throwback', 'lunch', 'unforgettable', 'dinner', 'anniversary', 'outing', 'enjoyed']
    
    def copy_images(self, data, destination_folder):
        """
        Copy images from their paths to the specified destination folder and update image paths in the DataFrame.
        
        Parameters:
        data (DataFrame): The DataFrame containing an "image_paths" column with image paths.
        destination_folder (str): The destination folder where images will be copied.
        
        Returns:
        DataFrame: Updated DataFrame with new image paths.
        """
        os.makedirs(destination_folder, exist_ok=True)
        
        updated_image_paths = []
        
        for _, row in data.iterrows():
            image_paths = row["image_paths"]
            new_paths = []
            
            if isinstance(image_paths, str):
                image_paths = image_paths.split(", ")
            elif not isinstance(image_paths, list):
                updated_image_paths.append([])
                continue
            
            for image_path in image_paths:
                image_path = image_path.strip()
                abs_image_path = os.path.abspath(image_path)
                
                if os.path.exists(abs_image_path) and os.path.isfile(abs_image_path):
                    filename = os.path.basename(abs_image_path)
                    dest_path = os.path.join(destination_folder, filename)
                    
                    try:
                        shutil.copy2(abs_image_path, dest_path)
                        new_paths.append(dest_path)
                        print(f"Copied: {abs_image_path} -> {dest_path}")
                    except Exception as e:
                        print(f"Error copying {abs_image_path}: {e}")
                else:
                    print(f"Warning: File not found - {abs_image_path}")
            
            updated_image_paths.append(new_paths)
        
        data["image_paths"] = updated_image_paths
        return data
    
    def filter_and_copy_images(self, data, platform):
        """
        Filter images and copy only the cleaned images to the respective platform folder.
        
        Parameters:
        data (DataFrame): Data containing posts and image paths.
        platform (str): The platform name (linkedin, facebook, instagram).
        
        Returns:
        DataFrame: The cleaned dataset with updated image paths.
        """
        pattern = '|'.join(self.keywords)
        
        invalid_posts = data[(data['post_heading'].str.contains(pattern, case=False, na=False)) 
                             | (data['post_content'].str.contains(pattern, case=False, na=False))]
        
        # Drop invalid posts
        cleaned_data = data.drop(invalid_posts.index)
        
        # Define destination folder
        destination_folder = f"./data/cleaned_data/{platform}_cleaned_images/"
        cleaned_data = self.copy_images(cleaned_data, destination_folder)
        
        return cleaned_data
    