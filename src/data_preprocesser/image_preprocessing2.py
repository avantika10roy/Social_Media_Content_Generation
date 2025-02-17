# ------------Done by Avantika Roy--------------

# DEPENDENCIES
import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from collections import Counter
from PIL import Image, ImageEnhance

# Add the root project directory to the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from src.utils.color_themes import get_color_themes

class ImagePreprocessor:
    """
        Class to Preprocess Images and derive the following:
        - Image Path       : Path of the image stored after preprocessing
        - Platform         : SNS from where the images are derived
        - Variant          : Type of preprocessing performed on the image (e.g., Noisy, Enhanced, Rotated, Original)
        - Color Theme      : The combination of dominating colors in the image
        - Layout           : Whether image has heavy text or is minimalistic
        - Text Description : Description of the image derived from BLIP
    """
    def __init__(self, image_dir, output_dir, json_output, blip_context_path, image_size=(1024, 1024)):
        """
            Constructor for ImagePreprocessor class:
            
            Arguments:
            ----------
            - image_dir         : folder path to input images
            - output_dir        : folder path to output images to store
            - json_output       : json file containing the directory of preprocessing features
            - blip_context_path : path to blip_image_context.json
            - image_size        : size input images are resized into
        """

        # Define parameters for the class constructor
        self.image_dir         = image_dir
        self.output_dir        = output_dir
        self.json_output       = json_output
        self.img_size          = image_size
        self.blip_context_path = blip_context_path
        
        # Create directory to store output
        os.makedirs(self.output_dir, exist_ok=True)

        # Image Preprocessing Parameters
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(p=0.1),
            A.Resize(*self.img_size)
        ])

    def load_blip_context(self):
        """
            Function to open blip_image_context.json file

            Arguments:
            ----------
            - self : Represents instance of the class
        """

        # Load blip_image_context.json
        with open(self.blip_context_path, "r") as file:
            return json.load(file)
        
    def preprocess_image(self, image_path):
        """
            Function to preprocess the images

            Arguments:
            ----------
            - image_path : path to input images
        """
        # Read, Convert Color and Resize images
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        
        # Reduced noise intensity
        noise       = np.random.normal(0, 1, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)

        # Color enhancement
        pil_image      = Image.fromarray(noisy_image)
        enhancer       = ImageEnhance.Color(pil_image)
        enhanced_image = np.array(enhancer.enhance(1.2))

        # Random rotation
        angle         = np.random.uniform(-5, 5)
        (h, w)        = enhanced_image.shape[:2]
        M             = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated_image = cv2.warpAffine(enhanced_image, M, (w, h))

        return {
            "original": image,
            "enhanced": enhanced_image,
            "noisy": noisy_image,
            "rotated": rotated_image
        }
    
    # def detect_logo(self, image):
    #     logo = cv2.imread(self.logo_path, 0)
    #     image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     res = cv2.matchTemplate(image_gray, logo, cv2.TM_CCOEFF_NORMED)
    #     _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
    #     threshold = 0.8
    #     if max_val >= threshold:
    #         x, y = max_loc
    #         h, w = logo.shape
    #         position = self.determine_position(x, y, image.shape, w, h)
    #         return True, position
    #     return False, None

    # def determine_position(self, x, y, img_shape, w, h):
    #     height, width = img_shape[:2]
    #     vertical = "top" if y < height // 2 else "bottom"
    #     horizontal = "left" if x < width // 2 else "right"
    #     return f"{vertical}-{horizontal}"
    
    def get_dominant_color(self, image):
        """
            Function to check dominant colors in images

            Arguments:
            ----------
            - image : path to input images
        """

        pixels             = np.float32(image.reshape(-1, 3))
        n_colors           = 5
        criteria           = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        counts             = Counter(labels.flatten())
        dominant_color     = palette[np.argmax(list(counts.values()))].astype(int)

        color_dict = get_color_themes()
        # color_dict = {
        #     "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
        #     "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
        #     "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
        #     "orange": (255, 165, 0), "purple": (128, 0, 128), "pink": (255, 192, 203), "brown": (165, 42, 42)
        # }
        distances = {name: np.linalg.norm(np.array(rgb) - dominant_color) for name, rgb in color_dict.items()}
        closest_colors = sorted(distances, key=distances.get)[:2]
        return f"{closest_colors[0]}-{closest_colors[1]}"
    
    def determine_layout(self, image):
        """
            Function to check layout of images

            Arguments:
            ----------
            - image : path to input images
        """

        # Convert image to greyscale
        gray        = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh   = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Determine Contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate whether layout is above or below threshold
        text_area   = sum(cv2.contourArea(cnt) for cnt in contours)
        img_area    = image.shape[0] * image.shape[1]

        return "text-heavy" if text_area / img_area > 0.5 else "minimalist"
    
    def process_images(self):
        """
            Function to create json file for the features extracted from preprocessing step
        """
        # Dictionary for json file
        results      = []

        # Load BLIP json file for context extraction
        blip_context = self.load_blip_context()
        blip_dict    = {}

        # Match Contexts with corresponding images
        for item in blip_context:
            text_descriptor = item.get("contexts", "No description available")
            for img_path in item.get("image_paths", []):  # Iterate through image paths
                blip_dict[img_path.strip()] = text_descriptor

        # Load BLIP json file for platform extraction
        blip_platforms = self.load_blip_context()

        blip_dict2 = {}
        # Match Platforms with corresponding images
        for option in blip_platforms:
            platform_name = option.get("platform", "No platforms given")
            for img_path in option.get("image_paths", []):
                blip_dict2[img_path.strip()] = platform_name

        # Dictionary for images with different format included
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Preprocessing Images
        for filename in tqdm(image_files, desc="Processing Images", unit="image"):
            image_path = os.path.join(self.image_dir, filename)
            preprocessed_images = self.preprocess_image(image_path)
            
            # Store variants of preprocessed images with different file names
            for variant_name, preprocessed_image in preprocessed_images.items():
                variant_filename = f"{os.path.splitext(filename)[0]}_{variant_name}.jpg"
                preprocessed_path = os.path.join(self.output_dir, variant_filename)

                # Add Platform name to corresponding images
                platform_name = blip_dict.get(image_path, "No platforms available")

                # logo_present, logo_position = self.detect_logo(preprocessed_image)
                color_theme = self.get_dominant_color(preprocessed_image)
                layout = self.determine_layout(preprocessed_image)
                
                cv2.imwrite(preprocessed_path, cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR))
                
                # Add text descriptor to corresponding images
                text_descriptor = blip_dict.get(image_path, "No description available")
                
                # Append all features
                results.append({
                    "image_path": preprocessed_path,
                    "variant": variant_name,
                    # "logo_present": logo_present,
                    # "logo_position": logo_position if logo_present else "none",
                    "color_theme": color_theme,
                    "layout": layout,
                    "text_descriptor": text_descriptor
                })

        with open(self.json_output, "w") as file:
            json.dump(results, file, indent=4)

        with open("image_preprocessing.json", "w") as file:
            json.dump(results, file, indent=4)

# Example Usage
preprocessor = ImagePreprocessor(
    image_dir="./data/curated_data/curated_images",
    output_dir="data/preprocessed_data/preprocessed_images2",
    json_output="data/preprocessed_data/preprocessed_data2.json",
    blip_context_path="data/Blip_with_context/blip_image_context.json"
)
preprocessor.process_images()
