import albumentations as A
import os
import cv2
from tqdm import tqdm
import json
import numpy as np


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

    def check_logo(image_path, logo_path, threshold=10):  
        """Check if a specific logo is present in an image using ORB + RANSAC and determine its position."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)

        if image is None or logo is None:
            print(f"Error: Image or logo not found ({image_path} or {logo_path})!")
            return False, None

        orb = cv2.ORB_create()
        
        # Detect keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(logo, None)
        kp2, des2 = orb.detectAndCompute(image, None)

        if des1 is None or des2 is None:
            return False, None  

        # Use BFMatcher with cross-check
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < threshold:
            return False, None  

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find Homography to filter out bad matches
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        if M is None or sum(matchesMask) < threshold:
            return False, None  

        # Get bounding box of detected logo
        h, w = logo.shape[:2]
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Compute center of detected region
        center_x = np.mean(dst[:, 0, 0])
        center_y = np.mean(dst[:, 0, 1])

        # Determine logo position
        img_h, img_w = image.shape[:2]

        if center_x < img_w / 3 and center_y < img_h / 3:
            position = "top-left"
        elif center_x > 2 * img_w / 3 and center_y < img_h / 3:
            position = "top-right"
        elif center_x < img_w / 3 and center_y > 2 * img_h / 3:
            position = "bottom-left"
        elif center_x > 2 * img_w / 3 and center_y > img_h / 3:
            position = "bottom-right"
        else:
            position = "center"

        return True, position

    def augment_and_save(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        augmented_mapping = {}

        for img_name in tqdm(image_files, desc="Processing Images"):
            img_path = os.path.join(self.input_dir, img_name)
            image = cv2.imread(img_path)
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

        print(f"Augmentation complete! Augmented images saved in {self.output_dir}")
        return augmented_mapping

    def convert_to_separate_rows(self, data, logo_path, augmented_mapping, save_dir="output.json"):
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
                    logo_present, logo_position = self.check_logo(image_path, logo_path)

                    separated_data.append({
                        "post_heading": post_heading,
                        "post_content": post_content,
                        "hashtags": hashtags,
                        "image_path": image_path,
                        "emoji": emojis,
                        "platform": platform_name,
                        "logo_present": logo_present,
                        "logo_position": logo_position
                    })

                    if image_path in augmented_mapping:
                        for aug_img_path in augmented_mapping[image_path]:
                            logo_present, logo_position = self.check_logo(aug_img_path, logo_path)
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


# Example Usage
input_folder = "data/curated_data/curated_images"
output_folder = "curated_new_cleaned_images"
preprocessor = PreProcessor(input_folder, output_folder)

# Load JSON data
with open("data/curated_data/curated_data.json", "r") as f:
    data = json.load(f)

augmented_mapping = preprocessor.augment_and_save()

# Process data with logo detection
preprocessor.convert_to_separate_rows(data, "data/raw_data/logo_image/itobuz_logo.jpg", augmented_mapping, "output2.json")
