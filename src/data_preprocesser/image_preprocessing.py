import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

class ImagePreprocessor:
    """
    A class to preprocess images by performing face detection, white space filtering,
    resizing, normalization, and data augmentation using PIL.
    """

    def __init__(self, data_dir, output_dir, target_size=(1024, 1024)):
        """
        Initialize the ImagePreprocessor.

        Args:
            data_dir (str): Directory containing input images.
            output_dir (str): Directory to save preprocessed and augmented images.
            target_size (tuple): Target size for resizing images (width, height).
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_size = target_size
        os.makedirs(output_dir, exist_ok=True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # List of post numbers to skip
        self.skip_posts = ['7', '23', '55', '61', '63', '65', '69', '83', '91']

    def contains_faces(self, img):
        """
        Check if an image contains faces using Haar Cascade face detection.

        Args:
            img (PIL.Image): Input image.

        Returns:
            bool: True if faces are detected, False otherwise.
        """
        img = img.convert('RGB')
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(faces) > 0

    def has_too_much_white_space(self, img, white_threshold=0.98, white_pixel_threshold=240):
        """
        Check if an image has too much white space.

        Args:
            img (PIL.Image): Input image.
            white_threshold (float): Percentage threshold for white pixels.
            white_pixel_threshold (int): Pixel intensity threshold for white.

        Returns:
            bool: True if the image has too much white space, False otherwise.
        """
        img_array = np.array(img)
        white_mask = np.all(img_array >= white_pixel_threshold, axis=-1)
        white_percentage = np.sum(white_mask) / img_array.size
        return white_percentage > white_threshold

    def resize_and_normalize(self, img):
        """
        Resize and normalize an image.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Resized and normalized image.
        """
        img = img.convert('RGB')
        img_resized = img.resize(self.target_size)
        img_normalized = np.array(img_resized) / 255.0
        return Image.fromarray((img_normalized * 255).astype(np.uint8))

    def augment_image(self, img):
        """
        Apply data augmentation to an image (Hue/Saturation adjustment, Median Blur, Horizontal Flip).

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Augmented image or None if it should be skipped.
        """
        img = img.convert('RGB')
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.5)  # Enhance color
        img = img.filter(ImageFilter.MedianFilter(size=3))  # Apply median blur
        img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Horizontal flip

        # Check if the augmented image has too much white space
        if self.has_too_much_white_space(img):
            print(f"Skipping augmented image with too much white space")
            return None

        return img

    def process_images(self):
        """
        Process all images in the input directory, applying preprocessing and augmentation.
        """
        for img_name in os.listdir(self.data_dir):
            img_path = os.path.join(self.data_dir, img_name)

            try:
                # Skip posts starting with specified numbers
                if any(img_name.startswith(post) for post in self.skip_posts):
                    print(f"Skipping post: {img_name}")
                    continue

                # Open image
                img = Image.open(img_path)

                # Skip if the image is a GIF
                if img.format == 'GIF':
                    print(f"Skipping GIF image: {img_name}")
                    continue

                # Skip images with faces
                if self.contains_faces(img):
                    print(f"Skipping image with faces: {img_name}")
                    continue

                # Skip images with too much white space
                if self.has_too_much_white_space(img):
                    print(f"Skipping image with too much white space: {img_name}")
                    continue

                # Preprocess the image
                img_processed = self.resize_and_normalize(img)

                # Save the processed image
                save_path = os.path.join(self.output_dir, img_name)
                img_processed.save(save_path)

                # Data Augmentation: Generate and save augmented images
                for i in range(5):  # Generate 5 augmented versions per image
                    augmented_img = self.augment_image(img)
                    if augmented_img is None:
                        continue
                    
                    augmented_img_path = os.path.join(self.output_dir, f"aug_{i}_{img_name}")
                    augmented_img.save(augmented_img_path)

            except Exception as e:
                print(f"Error processing image {img_name}: {e}")


# Example usage
if __name__ == "__main__":
    data_dir = "data/linkedin_data/linkedin_images"
    output_dir = "preprocessed_data/linkedin_processed_data"

    preprocessor = ImagePreprocessor(data_dir, output_dir)
    preprocessor.process_images()