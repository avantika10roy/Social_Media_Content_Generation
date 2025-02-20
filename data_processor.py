# ------------------------Done by Amrit Bag---------------------------#

# Dependencies
import os
import time
import pandas as pd
from dotenv import load_dotenv
from config.config import Config
from src.utils.logger import LoggerSetup          
from src.utils.data_saver import DataSaver
#from src.identify_logo import logo_identification
from src.data_cleaner.data_cleaner import DataCleaner
from src.data_cleaner.image_cleaner import ImageCleaner
from src.data_curator.data_curation import DataCuration
from src.utils.download_from_drive import DownloadData
#from src.feature_engineering import blip_feature_extraction
from src.data_preprocesser.image_augmentor import PreProcessor
from src.data_preprocesser.text_preprocessing import TextPreprocessing
from src.data_preprocesser.image_preprocessing import ImagePreprocessor
from src.data_preprocesser.llm_finetune_data_preprocessor import merge_post_contents
"""
# To download Data From Drive run this lines
downloader = DownloadData(client_secrets_path = Config.CLIENT_SECRET_CREDENTIALS)
downloader.download_google_drive_folder(Config.INSTAGRAM_RAW_IMAGE_DATA_LINK, Config.INSTAGRAM_RAW_IMAGE_DATA_PATH)
downloader.download_google_drive_folder(Config.FACEBOOK_RAW_IMAGE_DATA_LINK, Config.FACEBOOK_RAW_IMAGE_DATA_PATH)
downloader.download_google_drive_folder(Config.LINKEDIN_RAW_IMAGE_DATA_LINK, Config.LINKEDIN_RAW_IMAGE_DATA_PATH)
"""

# Set up logger
logger = LoggerSetup(logger_name="run.py", log_filename_prefix="RunScript").get_logger()

def main():
    try:
        #   Data Cleaning
        # -----------------
        logger.info("Starting data cleaning process...")
        cleaner                = DataCleaner()
        imageclean             = ImageCleaner()

        # Setting The raw data Path
        raw_facebook_data      = Config.FACEBOOK_RAW_POST_DATA_PATH
        raw_instagram_data     = Config.INSTAGRAM_RAW_POST_DATA_PATH
        raw_linkdin_data       = Config.LINKEDIN_RAW_POST_DATA_PATH

        # Raw json data converted into dataframe
        raw_facebook_data_df   = pd.read_json(path_or_buf = raw_facebook_data)  
        raw_instagram_data_df  = pd.read_json(path_or_buf = raw_instagram_data)
        raw_linkdin_data_df    = pd.read_json(path_or_buf = raw_linkdin_data)
       
        # Setting Cleaned Data path
        cleaned_facebook_data  = Config.FACEBOOK_CLEANED_POST_DATA_PATH
        cleaned_instagram_data = Config.INSTAGRAM_CLEANED_POST_DATA_PATH
        cleaned_linkedin_data  = Config.LINKEDIN_CLEANED_POST_DATA_PATH
        
        # Setting Updated Curated Data Path
        mixed_curated_data_path       = Config.MIXED_CURATED_DATA_PATH

        # Data Cleaning
        
        # Facebook Cleaning
        
        # Text Cleaning
        cleaned_facebook       = (cleaner.facebook_cleaner(facebook_data = raw_facebook_data_df ,
                                                            platform     ='facebook'))
        
        DataSaver.data_saver(data             = cleaned_facebook,
                             file_path        = cleaned_facebook_data)
        
        # Image cleaning
        
        fb_data                 = DataSaver.data_reader(cleaned_facebook_data)
        cleaned_fb_image        = imageclean.filter_and_copy_images(data     = fb_data,
                                                                    platform = 'facebook')
        DataSaver.data_saver(data             = cleaned_fb_image,
                             file_path        = cleaned_facebook_data)

        # Instagram Cleaning
        
        # Text Cleaning
        cleaned_instagram       = (cleaner.instagram_clean_data(insta_data = raw_instagram_data_df,
                                                               platform    ='instagram'))
        DataSaver.data_saver(data      = cleaned_instagram,
                             file_path = cleaned_instagram_data)
        
        # Image cleaning

        insta_data              = DataSaver.data_reader(cleaned_instagram_data)
        cleaned_instagram_image = imageclean.filter_and_copy_images(data     = insta_data,
                                                                    platform = 'instagram')
        DataSaver.data_saver(cleaned_instagram_image,cleaned_instagram_data)
        
        # Linkedin Cleaning
        
        # Text Cleaning
        
        cleaned_linkedin       = (cleaner.linkdln_clean_data(linkdln_data  = raw_linkdin_data_df,
                                                             platform      = 'linkedin'))
        DataSaver.data_saver(data        = cleaned_linkedin,
                             file_path   = cleaned_linkedin_data)
        
        # Image Cleaning
        
        linkedin_data          = DataSaver.data_reader(cleaned_linkedin_data)
        cleaned_linkedin_image = imageclean.filter_and_copy_images(data      = linkedin_data,
                                                                   platform  = 'linkedin')
        DataSaver.data_saver(cleaned_linkedin_image,cleaned_linkedin_data)

        logger.info("Data cleaning process completed successfully.")
        
        
        # Save cleaned data
       

        #         Data Curation
        #     --------------------

        # Setting Curated Data Path
        curated_data_path        = Config.CURATED_POST_DATA_PATH

        # Text Data curation
        curation = DataCuration(linkedin_cleaned_data_path   = cleaned_linkedin_data,
                                 instagram_cleaned_data_path = cleaned_instagram_data, 
                                 facebook_cleaned_data_path  = cleaned_facebook_data)
        logger.info("Starting data curation process...")

        curated_data             = curation.text_curation()

        # Curated Data saving

        DataSaver.data_saver(curated_data, curated_data_path)

        logger.info("Data curation process completed successfully.")
        
        
        # Generate LLM Finetuning Data
        
        merge_post_contents(
            raw_linkdin_data, 
            raw_facebook_data, 
            raw_instagram_data, 
            curated_data_path,
            mixed_curated_data_path
        ) 
        

        # Image Data Curation
        curation.image_curation(json_path                    = Config.CURATED_POST_DATA_PATH, 
                                curated_images_dir           = Config.CURATED_IMAGE_DATA_PATH
                                )

        #     Data Preprocessing
        #  ------------------------

        data_preprocessor   = PreProcessor()

        if os.path.exists(Config.BLIP_IMAGE_CONTEXT_DATA):
            blip_data       = pd.read_json(Config.BLIP_IMAGE_CONTEXT_DATA)
            separated_data  = data_preprocessor.convert_to_separate_rows(curated_data)
            augmented_data  = data_preprocessor.image_augmentation(df=separated_data,save_dir=Config.AUGMENTED_IMAGES_PATH)
            theme_data      = data_preprocessor.get_dominant_colors(df=blip_data)
            layout_data     = data_preprocessor.get_layout(df=theme_data)
            layout_data.to_json(Config.PREPROCESSED_POST_DATA_PATH,orient="records", indent=4, force_ascii=False)
        '''
        else:
            # Preprocessing the curated data
            preprocesed_data = pp.run_preprocessor(curated_data,save_dir=Config.AUGMENTED_IMAGES_PATH)

            # BLIP Context Extraction
            image_counter = 0
            start_time = time.time()
            for item in preprocesed_data:
                try:
                    image_path = item["image_path"]  
                    caption = blip_feature_extraction.generate_caption(image_path)  
                    
                
                    item["context"] = caption

                    image_counter += 1
                    
                    if image_counter % 10 == 0:
                        current_time = time.time()
                        time_diff = current_time - start_time
                        print(f"{image_counter} done, Running for {time_diff:.2f} sec")
                        
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    item["context"] = "Caption not available (error)"

            # Logo Identification
            folder_path = ""
            load_dotenv()

            MY_KEY = os.getenv("API_KEY")

            for post in preprocesed_data:
                logo_info_list = []
                image_path = post["image_path"]  # Now image_path is a string, not a list
                logo_info = logo_identification.generate_logo_info(image_path)
                img_path = os.path.join(folder_path, image_path)
                response = logo_identification.upload_image(img_path, MY_KEY)
                print(f"Response for {image_path}: {response}")
                logo_info_list.append(response)
                
                post["logo_info"] = logo_info_list


            for post in preprocesed_data:
                for logo_info in post.get("logo_info", []):
                    predictions = logo_info.get("predictions", [])
                    
                    if predictions:
                        # Extract logo details from the first prediction
                        first_prediction = predictions[0]
                        # Keep only the required fields
                        logo_info_cleaned = {
                            "logo_presence": "yes",
                            "logo_x": first_prediction.get("x", 0),
                            "logo_y": first_prediction.get("y", 0),
                            "logo_width": first_prediction.get("width", 0),
                            "logo_height": first_prediction.get("height", 0)
                        }
                    else:
                        # No logo detected
                        logo_info_cleaned = {
                            "logo_presence": "no",
                            "logo_x": 0,
                            "logo_y": 0,
                            "logo_width": 0,
                            "logo_height": 0
                        }
                
                # Replace the original logo_info with the cleaned version
                logo_info.clear()
                logo_info.update(logo_info_cleaned)
            
            preprocesed_data.to_json(Config.PREPROCESSED_POST_DATA_PATH,orient="records", indent=4, force_ascii=False)

            '''

    except Exception as e:
        logger.error(f"Error in data cleaning process: {e}")

if __name__ == "__main__":
    main()