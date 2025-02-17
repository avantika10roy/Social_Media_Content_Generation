import albumentations as A
import cv2
import json
import pandas as pd
import os

class PreProcessor:
    def __init__(self, img_size=(1024, 1024)):
        self.img_size = img_size
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(p=0.1),
            A.Resize(*self.img_size)
        ])
    
    def convert_to_separate_rows(self, df, save_path='new_data.json'):
        separated_data = []
        for _, post in df.iterrows():
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
                        "emojis": emojis,
                        "platform_name": platform_name
                    })
                else:
                    print(f"Image file not found: {image_path}")
        
        separated_df = pd.DataFrame(separated_data)
        separated_df.to_json(save_path, orient='records', indent=4, force_ascii=False)
        print(f"Data successfully converted and saved to {save_path}")
        
        return separated_df
    
    
    def image_augmentation(self, df, save_dir='augmented_images', num_augmented_copies=3):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        augmented_data = []
        
        for _, item in df.iterrows():
            image_path = item["image_path"]
            
            if os.path.exists(image_path):
                # Read image using OpenCV
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize image
                image_resized = cv2.resize(image, self.img_size)

                resized_file_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_resized.jpg"
                resized_image_path = os.path.join(save_dir, resized_file_name)
                cv2.imwrite(resized_image_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                
                augmented_data.append({
                    "post_heading": item["post_heading"],
                    "post_content": item["post_content"],
                    "hashtags": item["hashtags"],
                    "emojis": item["emojis"],
                    "platform_name": item["platform_name"],
                    "image_path": resized_image_path
                })
                
                for i in range(num_augmented_copies):
                    augmented = self.transform(image=image_resized)["image"]
                    
                    file_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i}.jpg"
                    save_path = os.path.join(save_dir, file_name)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                    
                    augmented_data.append({
                        "post_heading": item["post_heading"],
                        "post_content": item["post_content"],
                        "hashtags": item["hashtags"],
                        "emojis": item["emojis"],
                        "platform_name": item["platform_name"],
                        "image_path": save_path
                    })
            
            else:
                print(f"Skipping augmentation: Image file not found {image_path}")
        
        return pd.DataFrame(augmented_data)



df = pd.read_json('./data/curated_data/curated_data.json')

pp = PreProcessor()

separated_df = pp.convert_to_separate_rows(df)

augmented_df = pp.image_augmentation(separated_df)
augmented_df.to_json("augmented_data.json", orient="records", indent=4, force_ascii=False)

