import albumentations as A 
import os 
import cv2
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

class PreProcessor:
    def __init__(self, input_dir, output_dir, aug_per_img=3):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augmentations_per_image = aug_per_img
        os.makedirs(self.output_dir, exist_ok=True)

        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(p=0.5, limit=10),
            A.HorizontalFlip(p=0.1),
            A.GaussNoise(p=0.1),
            A.Resize(1024, 1024),
        ])

    def _validate_path(self, path):
        """Validate and convert path to string."""
        if path is None:
            return None
        try:
            # Convert to Path object and resolve to absolute path
            path_obj = Path(path).resolve()
            return str(path_obj)
        except Exception as e:
            print(f"Invalid path: {path}, Error: {e}")
            return None

    def check_logo(self, image_path, logo_path, threshold=0.6):
        """
        Check if a specific logo is present in the given image using template matching.
        Returns whether logo was found and its position.
        """
        # Validate paths
        image_path = self._validate_path(image_path)
        logo_path = self._validate_path(logo_path)
        
        if not image_path or not logo_path:
            print(f"Invalid path provided - Image: {image_path}, Logo: {logo_path}")
            return False, None

        if not os.path.exists(image_path) or not os.path.exists(logo_path):
            print(f"File not found - Image: {image_path}, Logo: {logo_path}")
            return False, None

        try:
            # Load the image and logo
            image = cv2.imread(image_path)
            logo = cv2.imread(logo_path)
            
            if image is None or logo is None:
                print(f"Failed to load - Image: {image_path}, Logo: {logo_path}")
                return False, None

            # Convert images to grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

            # Create a list of scales to try
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
            best_match_val = 0
            best_position = None
            logo_found = False

            # Try different scales of the logo
            for scale in scales:
                # Resize logo
                if scale != 1.0:
                    width = int(logo_gray.shape[1] * scale)
                    height = int(logo_gray.shape[0] * scale)
                    resized_logo = cv2.resize(logo_gray, (width, height))
                else:
                    resized_logo = logo_gray

                # Perform template matching
                result = cv2.matchTemplate(image_gray, resized_logo, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # Update best match if current match is better
                if max_val > best_match_val:
                    best_match_val = max_val
                    
                    # Get position coordinates
                    h, w = resized_logo.shape
                    x, y = max_loc
                    img_h, img_w = image.shape[:2]
                    
                    # Determine position in image
                    if x < img_w / 3 and y < img_h / 3:
                        best_position = "top-left"
                    elif x > 2 * img_w / 3 and y < img_h / 3:
                        best_position = "top-right"
                    elif x < img_w / 3 and y > 2 * img_h / 3:
                        best_position = "bottom-left"
                    elif x > 2 * img_w / 3 and y > 2 * img_h / 3:
                        best_position = "bottom-right"
                    else:
                        best_position = "center"

            # Check if the best match exceeds our threshold
            logo_found = best_match_val >= threshold

            return logo_found, best_position

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return False, None

    def convert_to_separate_rows(self, data, logo_path, augmented_mapping, save_dir="new_data.json"):
        separated_data = []
        
        for post in tqdm(data, desc="Processing posts"):
            post_heading = post.get("post_heading", "")
            post_content = post.get("post_content", "")
            hashtags = post.get("hashtags", [])
            emojis = post.get("emoji", [])
            platform_name = post.get("platform", "")
            image_paths = post.get("image_paths", [])

            # Ensure image_paths is a list
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            elif not isinstance(image_paths, list):
                print(f"Invalid image_paths format for post: {post_heading}")
                continue

            for image_path in image_paths:
                # Validate and process image path
                valid_image_path = self._validate_path(image_path)
                if not valid_image_path:
                    print(f"Invalid image path: {image_path}")
                    continue

                if os.path.exists(valid_image_path):  
                    logo_present, logo_position = self.check_logo(valid_image_path, logo_path)

                    separated_data.append({
                        "post_heading": post_heading,
                        "post_content": post_content,
                        "hashtags": hashtags,
                        "image_path": valid_image_path,
                        "emoji": emojis,
                        "platform": platform_name,
                        "logo_present": logo_present,
                        "logo_position": logo_position  
                    })

                    # Process augmented images
                    if valid_image_path in augmented_mapping:
                        for aug_img_path in augmented_mapping[valid_image_path]:
                            # Check logo in augmented image
                            aug_logo_present, aug_logo_position = self.check_logo(aug_img_path, logo_path)
                            
                            separated_data.append({
                                "post_heading": post_heading,
                                "post_content": post_content,
                                "hashtags": hashtags,
                                "image_path": aug_img_path,
                                "emoji": emojis,
                                "platform": platform_name,
                                "logo_present": aug_logo_present,
                                "logo_position": aug_logo_position
                            })
                else:
                    print(f"Image file not found: {valid_image_path}")

        # Save the results
        try:
            with open(save_dir, "w", encoding='utf-8') as file:
                json.dump(separated_data, file, indent=4, ensure_ascii=False)
            print(f"Data successfully converted and saved to {save_dir}")
        except Exception as e:
            print(f"Error saving data to {save_dir}: {str(e)}")

    def augment_and_save(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        augmented_mapping = {}

        for img_name in tqdm(image_files, desc="Processing Images"):
            try:
                img_path = os.path.join(self.input_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                augmented_images = []
                for i in range(self.augmentations_per_image):
                    augmented = self.transform(image=image)["image"]
                    augmented_np = np.clip(augmented.astype(np.uint8), 0, 255)
                    
                    new_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                    new_img_path = os.path.join(self.output_dir, new_img_name)
                    cv2.imwrite(new_img_path, cv2.cvtColor(augmented_np, cv2.COLOR_RGB2BGR))

                    augmented_images.append(new_img_path)
                
                augmented_mapping[img_path] = augmented_images

            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                continue

        print(f"Augmentation complete! Augmented images saved in {self.output_dir}")
        return augmented_mapping
    

input_folder = "data/curated_data/curated_images"
output_folder = "curated_new_cleaned_images"
preprocessor = PreProcessor(input_folder, output_folder)

# Load JSON data
with open("data/curated_data/curated_data.json", "r", encoding='utf-8') as f:
    data = json.load(f)

# Make sure your data structure has 'image_paths' as a list
# If it's not already in the correct format, you might need to modify it:
for item in data:
    if 'image_paths' in item and isinstance(item['image_paths'], str):
        item['image_paths'] = [item['image_paths']]

# Run augmentation and processing
augmented_mapping = preprocessor.augment_and_save()
preprocessor.convert_to_separate_rows(
    data, 
    "data/raw_data/logo_image/itobuz_logo.jpg", 
    augmented_mapping, 
    "output2.json"
)