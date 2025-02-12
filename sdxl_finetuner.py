#-----------Done by Avantika Roy----------

# IMPORT DEPENDENCIES
import os
import cv2
import torch
import numpy as np
import pandas as pd
from collections import Counter
from colorthief import ColorThief
from datasets import load_dataset
from sklearn.cluster import KMeans
from transformers import BlipProcessor
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

class SDXLFineTuner:
    def __init__(self, 
                 model_name = "stabilityai/stable-diffusion-xl-base-1.0", 
                 data_path  = df):
        
        # variable definition
        self.model_name = model_name
        self.data_path  = data_path
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset    = self.load_data()
        self.pipe       = self.setup_pipeline()
        self.tokenizer  = 

    def load_data(self, 
                  image_column   = "image", 
                  caption_column = "caption"):
        dataset = load_dataset()

    def setup_pipeline(self):
        pipe = StableDiffusionXLPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float16)
        pipe.vae = vae.to(self.device)
        pipe.to(self.device)
        return pipe
    
    def check_logo_presence(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges) > 10000
    
    def get_dominant_color(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = np.reshape(image, (-1, 3))
        most_common = Counter(map(tuple, pixels)).most_common(1)
        return most_common[0][0] if most_common else None
    
    def fine_tune(self, lora_adapter, output_dir="sdxl_finetuned", num_epochs=5):
        self.pipe = lora_adapter.apply(self.pipe)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.pipe.unet.parameters(), lr=1e-5)
        
        # Training loop
        for epoch in range(num_epochs):
            for batch in self.dataset:
                image_path = batch["image"]
                captions = batch["caption"]
                
                # Logo presence check
                logo_present = self.check_logo_presence(image_path)
                dominant_color = self.get_dominant_color(image_path)
                print(f"Image: {image_path}, Logo Present: {logo_present}, Dominant Color: {dominant_color}")
                
                images = torch.tensor(cv2.imread(image_path)).permute(2, 0, 1).to(self.device)
                input_ids = self.tokenizer(text=captions, return_tensors="pt", padding=True).input_ids.to(self.device)
                
                # Forward pass
                loss = self.pipe.unet(images, input_ids)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f"Epoch {epoch + 1} completed.")
        
        # Save the fine-tuned model
        self.pipe.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")



