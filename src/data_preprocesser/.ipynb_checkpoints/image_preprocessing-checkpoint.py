import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define paths
DATA_DIR = "data/linkedin_data/linkedin_images"
OUTPUT_DIR = "data/linkedin_processed_images"
LATENT_OUTPUT_DIR = "data/linkedin_latent_images"

# Create output directories if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LATENT_OUTPUT_DIR, exist_ok=True)

# Define image transformations
IMG_SIZE = (256, 256)  

transform = A.Compose([
    A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=(0.5,), std=(0.5,)),  
    ToTensorV2()
])

# Load Stable Diffusion VAE Encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

def preprocess_and_save(image_path, output_dir, latent_dir):
    """ Load image, apply transformations, and save augmented images. """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    augmented = transform(image=img)['image']  # Apply transformations
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    latent_path = os.path.join(latent_dir, os.path.basename(image_path).replace(".jpg", ".pt"))

    # Save processed image
    Image.fromarray(augmented.permute(1, 2, 0).numpy().astype(np.uint8)).save(output_path)

    # Convert image to latent space
    with torch.no_grad():
        latent = vae.encode(augmented.unsqueeze(0).to(device) * 2 - 1).latent_dist.sample()
    torch.save(latent, latent_path)  # Save latent representation

# Process all images
for file_name in os.listdir(DATA_DIR):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        preprocess_and_save(os.path.join(DATA_DIR, file_name), OUTPUT_DIR, LATENT_OUTPUT_DIR)

print("Image preprocessing and latent conversion completed!")
