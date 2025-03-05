# ________ DONE BY JIT________

# DEPENDENCIES
import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from ..utils.logger import LoggerSetup
from ..utils.color_themes import get_color_themes
from sklearn.cluster import MiniBatchKMeans

preprocessor_log = LoggerSetup(logger_name='preprocessor_log', log_filename_prefix='PreProcessor').get_logger()
class PreProcessor:
    '''
    PreProcesses the data.
    Arguments:
    ----------
    img_size : Size of the images 
    '''
    def __init__(self, img_size: tuple=(1024, 1024)):
        preprocessor_log.info("Initialized PreProcessor.")
        self.img_size   = img_size
        self.transform  = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(p=0.1),
            A.Resize(*self.img_size)
        ])
    
    def convert_to_separate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' 
        Converts the data to separate rows based on the image paths.
        Arguments:
        ---------
        df          : Input Data.

        Returns:
        -------
        separated_df : Row Separated Data.
        '''
        preprocessor_log.info("Converting to separate rows.")
        separated_data = []
        try:
            for _, post in df.iterrows():
                post_heading    = post.get("post_heading", "")
                post_content    = post.get("post_content", "")
                hashtags        = post.get("hashtags", [])
                emojis          = post.get("emoji", [])
                platform_name   = post.get("platform", "")
                image_paths     = post.get("image_paths", [])

                if not isinstance(image_paths, list):
                    preprocessor_log.warning(f"Skipping post with invalid image_paths format: {image_paths}")
                    continue
                
                for image_path in image_paths:
                    if os.path.exists(image_path):
                        separated_data.append({
                            "post_heading"  : post_heading,
                            "post_content"  : post_content,
                            "hashtags"      : hashtags,
                            "image_path"    : image_path,
                            "emojis"        : emojis,
                            "platform_name" : platform_name
                        })
                    else:
                        preprocessor_log.warning(f"Image file not found: {image_path}")

        except Exception as e:
            preprocessor_log.error(f"Error during row conversion: {repr(e)}")

        preprocessor_log.info("Successfully converted to separate rows")
        separated_df = pd.DataFrame(separated_data)
        return separated_df
    
    
    def image_augmentation(self, df: pd.DataFrame, save_dir: str, num_augmented_copies: int=3) -> pd.DataFrame:
        '''
        Performs Image Augmentation.
        Arguments:
        ----------
        df                      : Input Data.
        save_dir                : Directory to save augmented images.
        num_augmented_copies    : Number of augmented copies per images (Default = 3).   

        Returns:
        --------
        augmented_data          : Data with updated image paths.
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        augmented_data = []

        preprocessor_log.info("Augmenting images....")
        try:
            for _, item in df.iterrows():
                image_path = item["image_path"]

                if not isinstance(image_path, str) or not os.path.exists(image_path):
                    preprocessor_log.warning(f"Skipping augmentation: Invalid or missing image path {image_path}")
                    continue

                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        preprocessor_log.error(f"Skipping: Unable to read image {image_path}")
                        continue
                    image                   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_resized           = cv2.resize(image, self.img_size)

                    resized_file_name       = f"{os.path.splitext(os.path.basename(image_path))[0]}_resized.jpg"
                    resized_image_path      = os.path.join(save_dir, resized_file_name)
                    cv2.imwrite(resized_image_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                    
                    augmented_data.append({
                        "post_heading"      :   item["post_heading"],
                        "post_content"      :   item["post_content"],
                        "hashtags"          :   item["hashtags"],
                        "emojis"            :   item["emojis"],
                        "platform_name"     :   item["platform_name"],
                        "image_path"        :   resized_image_path
                    })
                    
                    for i in range(num_augmented_copies):
                        try:
                            augmented = self.transform(image=image_resized)["image"]
                            
                            file_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i}.jpg"
                            save_path = os.path.join(save_dir, file_name)
                            
                            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                            
                            augmented_data.append({
                                "post_heading"  : item["post_heading"],
                                "post_content"  : item["post_content"],
                                "hashtags"      : item["hashtags"],
                                "emojis"        : item["emojis"],
                                "platform_name" : item["platform_name"],
                                "image_path"    : save_path
                            })
                        except Exception as aug_err:
                            preprocessor_log.error(f"Augmentation failed for {image_path} (copy {i}): {repr(aug_err)}")
                    
                except Exception as img_err:
                    preprocessor_log.error(f"Error processing image {image_path}: {repr(img_err)}")
                
        except Exception as e:
            preprocessor_log.error(f"Unexpected error in image augmentation: {repr(e)}")

        preprocessor_log.info(f"Augmentation completed.")
        return pd.DataFrame(augmented_data)
    

    def extract_color(self, image_path : str) -> str:
        '''Extracts dominant colors using clustering.'''
        try:
            image           = cv2.imread(image_path)
            image           = cv2.resize(image, (100, 100))
            image           = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixels          = image.reshape(-1, 3)

            kmeans          = MiniBatchKMeans(n_clusters=3, random_state=42)
            kmeans.fit(pixels)
            dominant_color  = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))].astype(int)

            color_dict      = get_color_themes()
            color_array     = np.array(list(color_dict.values()))
            distances       = np.linalg.norm(color_array - dominant_color, axis=1)
            closest_colors  = np.array(list(color_dict.keys()))[np.argsort(distances)[:2]]

            return f"{closest_colors[0]}-{closest_colors[1]}"
        
        except Exception as e:
            preprocessor_log.error(f"Error extracting color from {image_path}: {repr(e)}")
            return "unknown"

    def get_dominant_colors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract dominant colors for each image in the DataFrame and add a 'color_theme' column.
        """
        preprocessor_log.info("Extracting dominant colors.")
        df["color_theme"] = df["image_path"].apply(self.extract_color)
        preprocessor_log.info("Successfully extracted dominant colors.")
        return df
    
    def determine_layout(self, image_path: str) -> str:
        """
            Function to check layout of images

            Arguments:
            ----------
            - image : path to input images
        """
        try:
            image = cv2.imread(image_path)
            gray        = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, thresh   = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Determine Contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate whether layout is above or below threshold
            text_area   = sum(cv2.contourArea(cnt) for cnt in contours)
            img_area    = image.shape[0] * image.shape[1]

            return "text-heavy" if text_area / img_area > 0.5 else "minimalist"
        except Exception as e:
            preprocessor_log.error(f"Error determining layout for {image_path}: {repr(e)}")
            return "unknown"
    
    def get_layout(self, df: pd.DataFrame) -> pd.DataFrame: # DONE BY AVANTIKA
       ''' Extracts layout of the image.'''
       preprocessor_log.info("Determining Layout.")
       df['image_layout'] =  df['image_path'].apply(self.determine_layout)
       preprocessor_log.info('Image layout determined.')
       return df
    

    def run_preprocessor(self, df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        '''
        Runs the data preprocessor.
        Arguments:
        ---------
        df : Input Data.
        save_dir : Directory to save augmented images.

        Returns:
        --------
        df : Preprocessed data.
        '''
        try:
            df = self.convert_to_separate_rows(df)
            df = self.image_augmentation(df, save_dir)
            df = self.get_dominant_colors(df)
            df = self.get_layout(df)
            preprocessor_log.info("Preprocessing completed successfully.")
            return df
        except Exception as e:
            preprocessor_log.error(f"Error in run_all: {repr(e)}")
            return df

