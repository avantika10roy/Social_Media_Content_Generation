import os
import torch
from torchvision import transforms
from PIL import Image
import random
import shutil

class ImagePreprocessor:
    def __init__(self, raw_data_path, cleaned_data_path):
        self.raw_data_path     = raw_data_path
        self.cleaned_data_path = cleaned_data_path
        self.image_size        = (1024, 1024)  # Resize target
        
        if not os.path.exists(self.cleaned_data_path):
            os.makedirs(self.cleaned_data_path)

        # Define the transformations: Resize, Normalize, and Augment
        self.transform = transforms.Compose([
                                        transforms.Resize(self.image_size),  # Resize to 1024x1024
                                        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
                                        transforms.RandomHorizontalFlip(),  # Random horizontal flip
                                        #transforms.RandomRotation(30),  # Random rotation within 30 degrees
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
                                    ])
    
    def process_image(self, image_path):
        """Load, resize, normalize, and augment the image."""
        image               = Image.open(image_path)
        
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            return []

        augmented_images     = []
        # Generate 4 to 5 augmentations
        for _ in range(random.randint(4, 5)):  # Randomly choose 4 or 5 augmentations
            # Apply the transformations
            image_transformed = self.transform(image)

            # Convert tensor back to PIL for saving
            image_transformed = transforms.ToPILImage()(image_transformed)
            augmented_images.append(image_transformed)

        return augmented_images
    
    def save_image(self, image, base_name, index):
        """Save the processed image to the cleaned data path with a unique name."""
        file_name = f"{base_name}_aug_{index}.jpg"
        save_path = os.path.join(self.cleaned_data_path, file_name)
        image.save(save_path)
        print(f"Saved processed image: {save_path}")

    def preprocess_images(self):
        """Preprocess all images in the raw data path."""
        for file_name in os.listdir(self.raw_data_path):
            file_path            = os.path.join(self.raw_data_path, file_name)
            
            if os.path.isfile(file_path):
                augmented_images = self.process_image(file_path)
                
                if augmented_images:
                    base_name, _ = os.path.splitext(file_name)
                    for idx, img in enumerate(augmented_images):
                        self.save_image(img, base_name, idx)

