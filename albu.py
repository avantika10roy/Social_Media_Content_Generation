import albumentations as A 
import os 
import cv2
from tqdm import tqdm
import json
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
        A.HorizontalFlip(p=0.1),
        A.GaussNoise(p=0.1),
        A.Resize(1024,1024),
        #ToTensorV2()
    ])

    def augment_and_save(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        augmented_mapping ={}

        for img_name in tqdm(image_files, desc="Processing Images"):
            img_path = os.path.join(self.input_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            augmented_images = []
            for i in range(self.augmentations_per_image):
                augmented = self.transform(image=image)["image"]
                
                # Convert tensor to NumPy array
                #augmented_np = augmented.permute(1, 2, 0).cpu().numpy()
                augmented_np = np.clip(augmented.astype(np.uint8), 0, 255)
                
                # Save transformed image
                new_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                new_img_path = os.path.join(self.output_dir, new_img_name)
                cv2.imwrite(new_img_path, cv2.cvtColor(augmented_np, cv2.COLOR_RGB2BGR))

                augmented_images.append(new_img_path)
            
            augmented_mapping[img_path] = augmented_images

        print(f"Augmentation complete! Augmented images saved in {self.output_dir}")
        return augmented_mapping


    def check_logo(self, image_path, logo_path, threshold=0.5):
        """Check if a specific logo is present in the given image using contour detection."""
        # Load the image and logo
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        logo = cv2.imread(logo_path, cv2.IMREAD_COLOR)
        
        if image is None or logo is None:
            print(f"Error: Image or logo not found ({image_path} or {logo_path})!")
            return False, None

        # Preprocess the logo (resize and convert to grayscale)
        logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        _, logo_thresh = cv2.threshold(logo_gray, 127, 255, cv2.THRESH_BINARY)
        logo_contours, _ = cv2.findContours(logo_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Preprocess the image (convert to grayscale)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image_thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
        image_contours, _ = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Match contours between the logo and the image
        match_found = False
        position = None
        for contour in image_contours:
            # Approximate the contour to reduce complexity
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Compare the contour with the logo's contour
            for logo_contour in logo_contours:
                match = cv2.matchShapes(logo_contour, approx, cv2.CONTOURS_MATCH_I1, 0.0)
                if match < threshold:  # Lower values indicate better matches
                    match_found = True

                    # Determine the position of the matched contour
                    x, y, w, h = cv2.boundingRect(contour)
                    img_h, img_w = image.shape[:2]
                    if x < img_w / 3 and y < img_h / 3:
                        position = "top-left"
                    elif x > 2 * img_w / 3 and y < img_h / 3:
                        position = "top-right"
                    elif x < img_w / 3 and y > 2 * img_h / 3:
                        position = "bottom-left"
                    elif x > 2 * img_w / 3 and y > 2 * img_h / 3:
                        position = "bottom-right"
                    else:
                        position = "center"

                    # Draw the bounding box for visualization (optional)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    break

        return match_found, position
        

    def convert_to_separate_rows(self, data, logo_path, augmented_mapping, save_dir="new_data.json"):
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
                    logo_present,logo_position = self.check_logo(image_path, logo_path)

                    separated_data.append({
                        "post_heading": post_heading,
                        "post_content": post_content,
                        "hashtags": hashtags,
                        "image_path": image_path,
                        "emoji": emojis,
                        "platform": platform_name,
                        "logo_present": logo_present,
                        "logo_position":logo_position  
                    })
                    if image_path in augmented_mapping:
                        for aug_img_path in augmented_mapping[image_path]:
                            separated_data.append({
                                "post_heading": post_heading,
                                "post_content": post_content,
                                "hashtags": hashtags,
                                "image_path": aug_img_path,
                                "emoji": emojis,
                                "platform": platform_name,
                                "logo_present": logo_present,
                                "logo_position": logo_position
                            })
                else:
                    print(f"Image file not found: {image_path}")

        with open(save_dir, "w") as file:
            json.dump(separated_data, file, indent=4, ensure_ascii=False)
            print(f"Data successfully converted and saved to {save_dir}")


# # Example usage
# df = pd.read_json('./data/curated_data/curated_data.json')
# data = df.to_dict(orient='records')
# pp = PreProcessor()
# pp.convert_to_separate_rows(data)

input_folder = "data/curated_data/curated_images"
output_folder = "curated_new_cleaned_images"
# augmentor = PreProcessor(input_folder, output_folder)
# augmentor.augment_and_save()

preprocessor = PreProcessor(input_folder, output_folder)

# Load JSON data
with open("data/curated_data/curated_data.json", "r") as f:
    data = json.load(f)

augmented_mapping = preprocessor.augment_and_save()
# Process data with logo detection
preprocessor.convert_to_separate_rows(data, "data/raw_data/logo_image/itobuz_logo.jpg", augmented_mapping, "output.json")

# Check all augmented images
# detected_images = preprocessor.check_logo("data/curated_data/curated_images")
# print(detected_images)

