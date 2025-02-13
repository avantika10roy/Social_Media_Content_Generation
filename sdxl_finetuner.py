#-----------Done by Avantika Roy----------

# IMPORT DEPENDENCIES
import os
import cv2
import torch
import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset
from transformers import BlipProcessor
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

class SDXLFineTuner:
    """
        This class finetunes the Stable Diffusion XL model.
    """
    def __init__(self, model_name = "stabilityai/stable-diffusion-xl-base-1.0", data_path = ):
        """
            Initialization of Fine Tuner
 
            Arguments:
            ----------
                model_name : Address of the model on HuggingFace

                data_pth   : Dataset used to finetune the model
        """
        # variable definition
        self.model_name = model_name
        self.data_path  = data_path
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset    = self.load_data()
        self.pipe       = self.setup_pipeline()
        self.tokenizer  = BlipProcessor.from_pretrained("Sales/blip_image_captioning_base")

    def load_data(self, image_column = "image", caption_column = "caption"):
        """
            Loading Dataset

            Arguments: 
            ----------
                image_column   : Column containing image name or image path

                caption_column : Column containing caption column
        """
        dataset = load_dataset()

    def setup_pipeline(self):

        pipe     = StableDiffusionXLPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        vae      = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float16)
        pipe.vae = vae.to(self.device)
        pipe.to(self.device)
        return pipe
    
    def fine_tune(self, lora_adapter, output_dir="sdxl_finetuned", num_epochs=5):
        self.pipe = lora_adapter.apply(self.pipe)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.pipe.unet.parameters(), lr=1e-5)
        
        # Training loop
        for epoch in range(num_epochs):
            for batch in self.dataset:
                image_path = batch["image"]
                captions = batch["caption"]
                
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

# if __name__ == "__main__":
#     lora_adapter = LoraAdapter()
#     finetuner = SDXLFinetuner()
#     finetuner.fine_tune(lora_adapter)