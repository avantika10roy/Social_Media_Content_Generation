import os
import gdown
from PIL import Image

class ImageDataCleaner:
    """
    A class to handle cleaning of image data from a given Google Drive folder.
    The images will be cleaned by removing corrupted images and ensuring all images are of the same dimensions.
    """

    def __init__(self, folder_url, cleaned_data_path):
        """
        Initializes the ImageDataCleaner class.

        Args:
            folder_url (str): Google Drive folder URL or direct file link for the images.
            cleaned_data_path (str): Path where cleaned image data will be saved.
        """
        self.folder_url = folder_url
        self.cleaned_data_path = cleaned_data_path

        # Ensure the cleaned data directory exists
        if not os.path.exists(self.cleaned_data_path):
            os.makedirs(self.cleaned_data_path)

    def download_images_from_drive(self):
        """
        Downloads images from the Google Drive folder to the local environment.
        """
        print(f"Downloading images from {self.folder_url}...")

        # Download the folder from Google Drive using gdown
        gdown.download_folder(self.folder_url, quiet=False)

    def clean_images(self):
        """
        Cleans the images by removing corrupted files and ensuring all images are valid PNGs.
        Resizes images to a standard size (optional).
        """
        self.download_images_from_drive()

        # Assuming all downloaded images are in the current directory
        image_files = [f for f in os.listdir() if f.endswith('.png')]

        for img_file in image_files:
            img_path = os.path.join(img_file)
            try:
                with Image.open(img_path) as img:
                    img.verify()

                    # Reopen image after verification, as verify() closes it
                    with Image.open(img_path) as img:
                        img = img.resize((256, 256))  # Optional: Resize to 256x256

                        # Save the cleaned image to the destination folder
                        cleaned_img_path = os.path.join(self.cleaned_data_path, img_file)
                        img.save(cleaned_img_path)

            except (IOError, SyntaxError) as e:
                print(f"Skipping corrupted image: {img_file}. Error: {e}")
            except Exception as e:
                print(f"Unexpected error while processing {img_file}: {e}")

if __name__ == "__main__":
    folder_url = 'https://drive.google.com/drive/folders/1-PwFzDBfRFHWtadLEXubp-Kl2h8UquqG?usp=sharing'  
    cleaned_data_path = 'cleaned_data'   # Folder where cleaned images will be saved

    image_cleaner = ImageDataCleaner(folder_url, cleaned_data_path)

    image_cleaner.clean_images()
