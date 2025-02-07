import os
import random
import shutil
from PIL import Image
from torchvision import transforms
import cv2  # OpenCV for face detection

class ImagePreprocessor:
    def __init__(self, raw_data_path, cleaned_data_path):
        """
        Preprocesses Images
        -------------------
        Arguments:
                raw_data_path     : path to directory to raw image data
                cleaned_data_path : path to directory to clean image data
        """
        self.raw_data_path = raw_data_path
        self.cleaned_data_path = cleaned_data_path
        self.image_size = (1024, 1024)  # Resize target

        # Create directories for original and augmented images
        self.original_images_path = os.path.join(self.cleaned_data_path, "linkedin_original_images")
        self.augmented_images_path = os.path.join(self.cleaned_data_path, "linkedin_augmented_images")

        os.makedirs(self.original_images_path, exist_ok=True)
        os.makedirs(self.augmented_images_path, exist_ok=True)

        # Define the transformations: Resize, Normalize, and Augment
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),  # Resize to 1024x1024
            transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            # transforms.RandomRotation(30),  # Random rotation within 30 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
        ])
        
        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def contains_face(self, image_path):
        """Check if an image contains any faces."""
        image = cv2.imread(image_path)
        
        # Check if the image was loaded successfully
        if image is None:
            print(f"Warning: Unable to load image {image_path}. Skipping it.")
            return False  # Return False, meaning no faces detected as image can't be processed
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        return len(faces) > 0  # If any faces are detected, return True
    
    def is_gif(self, image_path):
        """Check if an image is a GIF, even if saved with a .png extension."""
        try:
            with Image.open(image_path) as img:
                if img.format == 'GIF':
                    print(f"Removing {image_path} because it is a GIF saved as PNG.")
                    return True
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        return False

    def process_image(self, image_path):
        """Load, resize, normalize, and augment the image."""
        image = Image.open(image_path)
        
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            return []

        augmented_images = []
        # Generate 4 to 5 augmentations
        for _ in range(random.randint(4, 5)):  # Randomly choose 4 or 5 augmentations
            # Apply the transformations
            image_transformed = self.transform(image)

            # Convert tensor back to PIL for saving
            image_transformed = transforms.ToPILImage()(image_transformed)
            augmented_images.append(image_transformed)

        return augmented_images
    
    def save_image(self, image, base_name, index, save_path):
        """Save the processed image to the specified path with a unique name."""
        file_name = f"{base_name}_aug_{index}.png"
        image.save(os.path.join(save_path, file_name))
        print(f"Saved processed image: {os.path.join(save_path, file_name)}")

    def preprocess_images(self):
        """Preprocess all images in the raw data path."""
        for file_name in os.listdir(self.raw_data_path):
            file_path = os.path.join(self.raw_data_path, file_name)
            
            if os.path.isfile(file_path):
                # Skip GIFs saved as PNG
                if self.is_gif(file_path):
                    os.remove(file_path)  # Remove the file if it's a GIF saved as PNG
                    continue

                # Skip images with faces
                if self.contains_face(file_path):
                    print(f"Skipping {file_name} because it contains a face.")
                    continue  # Skip this image if it contains a face

                # Copy the original image to the original images directory
                shutil.copy(file_path, os.path.join(self.original_images_path, file_name))

                # Process the image and generate augmented versions
                augmented_images = self.process_image(file_path)
                
                if augmented_images:
                    base_name, _ = os.path.splitext(file_name)
                    for idx, img in enumerate(augmented_images):
                        self.save_image(img, base_name, idx, self.augmented_images_path)
