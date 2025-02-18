# DEPENDENCIES
import os
import cv2
import pandas as pd
import albumentations as A
from ..utils.logger import LoggerSetup

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
    
    def convert_to_separate_rows(self, df: pd.DataFrame):
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
    
    
    def image_augmentation(self, df: pd.DataFrame, save_dir: str, num_augmented_copies: int=3):
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


