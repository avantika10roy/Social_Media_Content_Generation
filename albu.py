import albumentations as A 
import os 
import shutil
import cv2
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
import numpy as np


class PreProcessor:
    def __init__(self,input_dir, output_dir, aug_per_img=3):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augmentations_per_image = aug_per_img
        os.makedirs(self.output_dir,exist_ok=True)

        self.transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(p=0.5,limit=10),
        #A.RandomCrop(height=512,width=512,p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.1),
        A.resize(1024,1024),
        #ToTensorV2()
    ])
    
    def convert_to_separate_rows(self, data, save_dir='new_data.json'):
        separated_data = []
        for post in data:
            post_heading = post.get("post_heading", "")
            post_content = post.get("post_content", "")
            hashtags = post.get("hashtags", [])
            emojis = post.get("emoji", [])
            platform_name = post.get("platform", "")
            image_paths = post.get("image_paths", [])
            
            for image_path in image_paths:
                if os.path.exists(image_path):  
                    separated_data.append({
                        "post_heading": post_heading,
                        "post_content": post_content,
                        "hashtags": hashtags,
                        "image_path": image_path,
                        "emoji": emojis,
                        "platform": platform_name
                    })
                else:
                    print(f"Image file not found: {image_path}")
        
        with open(save_dir, "w") as file:
            json.dump(separated_data, file, indent=4, ensure_ascii=False)
            print(f"Data successfully converted and saved to {save_dir}")

# Load the JSON data into a DataFrame
#df = pd.read_json('./data/curated_data/curated_data.json')

# Convert DataFrame to list of dictionaries
#data = df.to_dict(orient='records')

# Instantiate PreProcessor and process the data
#pp = PreProcessor()
#pp.convert_to_separate_rows(data)

    

    def augment_and_save(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        for img_name in tqdm(image_files, desc="Processing Images"):
            img_path = os.path.join(self.input_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i in range(self.augmentations_per_image):
                augmented = self.transform(image=image)["image"]
                
                # Convert tensor to NumPy array
                #augmented_np = augmented.permute(1, 2, 0).cpu().numpy()
                augmented_np = np.clip(augmented.astype(np.uint8), 0, 255)
                
                # Save transformed image
                new_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                new_img_path = os.path.join(self.output_dir, new_img_name)
                cv2.imwrite(new_img_path, cv2.cvtColor(augmented_np, cv2.COLOR_RGB2BGR))

        print(f"Augmentation complete! Augmented images saved in {self.output_dir}")

    

# Load the JSON data
df = pd.read_json('./data/curated_data/curated_data.json')
data = df.to_dict(orient='records')

# Instantiate and process
#pp = PreProcessor()
#pp.convert_to_separate_rows(data)

input_folder = "data/curated_data/curated_images"
output_folder = "curated_new_cleaned_images"
augmentor = PreProcessor(input_folder, output_folder)
augmentor.augment_and_save()







