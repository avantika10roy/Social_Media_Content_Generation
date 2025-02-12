## ----- DONE BY SUBHAS MUKHERJEE -----
## ----- FORMATTED BY PRIYAM PAL -----


# DEPENDENCIES

import os
import cv2  
import random
from PIL import Image
from torchvision import transforms
from ..utils.logger import LoggerSetup

# LOGGING SETUP
imagePreprocessor_logger = LoggerSetup(logger_name = "image_preprocessing.py", log_filename_prefix = "image_preprocessor").get_logger()

class ImagePreprocessor:
    """
    Preprocesses Images.
    
    Arguments:
    ----------
        raw_data_path     : path to directory containing raw image data
        cleaned_data_path : path to directory where clean image data will be saved
    """
    
    def __init__(self, raw_data_path, cleaned_data_path):
        
        self.raw_data_path         = raw_data_path
        self.cleaned_data_path     = cleaned_data_path
        self.image_size            = (1024, 1024)  
        
        self.original_images_path  = os.path.join(self.cleaned_data_path, "curated_original_images")
        self.augmented_images_path = os.path.join(self.cleaned_data_path, "curated_augment_images")

        for path in [self.original_images_path, self.augmented_images_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        self.transform             = transforms.Compose([transforms.Resize(self.image_size),  
                                                         transforms.ToTensor(),  
                                                         transforms.RandomHorizontalFlip(), 
                                                         transforms.ColorJitter(brightness  = 0.2, 
                                                                                contrast    = 0.2, 
                                                                                saturation  = 0.2, 
                                                                                hue         = 0.2
                                                                                ), 
                                                         ]
                                                        )
        

        self.face_cascade          = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        imagePreprocessor_logger.info("Image Preprocessing Class Initialized")

    def contains_face(self, image_path):
        """Check if an image contains any faces."""
        
        image                      = cv2.imread(image_path)
        
        if image is None:
            imagePreprocessor_logger.warning(f"Unable to load image {image_path}. Skipping it.")
            return False  
        
        gray                       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces                      = self.face_cascade.detectMultiScale(gray, 
                                                                        scaleFactor  = 1.1, 
                                                                        minNeighbors = 5, 
                                                                        minSize      = (30, 30)
                                                                        )
        return len(faces) > 0  

    def is_gif(self, image_path):
        """Check if an image is a GIF."""
        
        try:
            with Image.open(image_path) as img:
                if img.format == 'GIF':
                    imagePreprocessor_logger.info(f"Removing {image_path} because it is a GIF saved as PNG.")
                    
                    return True
        
        except Exception as e:
            imagePreprocessor_logger.error(f"Error processing {image_path}: {e}")
        
            return False

    
    def process_image(self, image_path):
        """Load, resize, normalize, and augment the image."""
        
        try:
            image             = Image.open(image_path).convert('RGB')
        
        except Exception as e:
            print(f"Error: Unable to load image {image_path}: {e}")
            
            return []
        
        augmented_images      = []
        
        for _ in range(random.randint(4, 5)):
            image_transformed = self.transform(image)
            image_transformed = transforms.ToPILImage()(image_transformed)
            augmented_images.append(image_transformed)
        
        return augmented_images

    
    def save_image(self, image, save_path):
        """Save the image to the given path."""
        
        image.save(save_path)
        
        print(f"Saved image: {save_path}")

    
    def preprocess_images(self):
        """Preprocess all images in the raw data path."""
        
        for file_name in os.listdir(self.raw_data_path):
            file_path = os.path.join(self.raw_data_path, file_name)
            
            if os.path.isfile(file_path):
                if self.is_gif(file_path):
                    os.remove(file_path)
                    
                    continue
                
                if self.contains_face(file_path):
                    imagePreprocessor_logger.info(f"Skipping {file_name} because it contains a face.")
                    
                    continue
                
                try:
                    image              = Image.open(file_path).convert('RGB')
                    base_name, _       = os.path.splitext(file_name)
                    
                    original_save_path = os.path.join(self.original_images_path, f"{base_name}.jpg")
                    self.save_image(image, original_save_path)
                    
                    augmented_images   = self.process_image(file_path)
                    
                    for idx, img in enumerate(augmented_images):
                        aug_save_path  = os.path.join(self.augmented_images_path, f"{base_name}_aug_{idx}.jpg")
                        self.save_image(img, aug_save_path)
                
                except Exception as e:
                    imagePreprocessor_logger.error(f"Error processing {file_name}: {e}")
                    
                    continue

# TO RUN THE PROGRAM 
if __name__ == "__main__":
    raw_data_path = "data/curated_data/curated_images"
    cleaned_data_path = "data/preprocessed_data/preprocessed_images"
    preprocessor = ImagePreprocessor(raw_data_path, cleaned_data_path)
    preprocessor.preprocess_images()

# EXAMPLE CODE

# preprocessor = ImagePreprocessor(Config.CURATED_IMAGE_DATA_PATH, Config.PREPROCESSED_IMAGE_DATA_PATH)
# preprocessor.preprocess_images()