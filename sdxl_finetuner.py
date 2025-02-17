# DEPENDENCIES
import os
import json
import torch
import numpy as np
from PIL import Image
from typing import List
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import DiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel

class StableDiffusionWithBLIP:
    """
        Class that uses Stable Diffusion XL with BLIP for tokenization.
    """
    def __init__(self, input_json, output_dir, device = "cuda" if torch.cuda.is_available() else "cpu"):
        """
            Constructor for class StableDiffusionWithBLIP.

            Arguments:
            ----------
                - input_json : json file containing preprocessed data as input for model. 
                - output_dir : output directory where the generated images will be stored.
                - device     : uses cuda if available; if not then uses CPU.
        """
        # Define Parameters
        self.input_json = input_json
        self.output_dir = output_dir
        self.device     = device
        os.mkdir(self.output_dir, exist_ok = True)

        # Define Models
        self.blip_processor    = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model        = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
        self.text2img_pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(self.device)
        self.inpainting        = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to(self.device)

    def load_data(self):

        # Load input JSON data
        with open(self.input_json, "r") as f:
            data = json.load(f)
        self.input_json = data

    def encode_text_w_blip(self, prompt):

        inputs = self.blip_processor(text=prompt, return_tensors="pt").to(self.device)
        outputs = self.blip_model.generate(**inputs)
        return self.blip_processor.decode(outputs[0], skip_special_tokens=True)
    
    def generate_image(self, prompt, mask_image=None, base_image=None):

        if base_image is not None:
            return self.inpainting(prompt=prompt, image=base_image, mask_image=mask_image).images[0]
        else:
            return self.text2img_pipeline(prompt).images[0]
        
    def process_data(self):

        for idx, item in enumerate(self.input_json):
            prompt = item["text"]
            image_path = item.get("image_path")

            print(f"Processing {idx + 1}/{len(self.input_json)}: {prompt}")

            # Tokenize with BLIP
            processed_prompt = self.encode_text_w_blip(prompt)

            base_image = None
            mask_image = None

            if image_path and os.path.exists(image_path):
                base_image = Image.open(image_path).convert("RGB")
                mask_image = Image.new("L", base_image.size, 255)  # Placeholder mask (full white)

            # Generate image
            image = self.generate_image(processed_prompt, mask_image, base_image)

            # Save image
            output_path = os.path.join(self.output_dir, f"generated_{idx}.png")
            image.save(output_path)

            print(f"Saved: {output_path}")

        print("Image generation complete!")

    def run(self):
        """
        Function to execute the full image generation pipeline.
        It loads the input data, processes it, and saves the generated images.
        """
        # Load the input data
        print(f"Loading data from {self.input_json}...")
        self.load_data()

        # Process and generate images for each entry in the data
        print("Starting image generation process...")
        self.process_data()

        print("All images generated and saved!")
#-------------------------------------------------------------------------------------------------------------------------------

class StableDiffusionWithCLIP:
    def __init__(self, json_file, output_dir, device=None):
        self.json_file = json_file
        self.output_dir = output_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load Models
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(self.device)
        self.scheduler = EulerDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        self.text2img_pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(self.device)
        self.inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to(self.device)
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth").to(self.device)
        self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=self.controlnet).to(self.device)

        # Load CLIP components
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

    class CustomDiffusionPipeline(DiffusionPipeline):
        def __init__(self, unet, scheduler):
            super().__init__()
            self.unet = unet
            self.scheduler = scheduler

        def forward(self, latent: torch.FloatTensor, num_inference_steps: int = 50):
            # Apply the denoising process
            for i in range(num_inference_steps):
                latent = self.scheduler.step(self.unet(latent), t=i).prev_sample
            return latent

    def load_data(self):
        # Load input JSON data
        with open(self.json_file, "r") as f:
            data = json.load(f)
        self.input_json = data

    def encode_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return self.text_encoder(**inputs).last_hidden_state

    def generate_image(self, prompt, mask_image=None, base_image=None):
        if base_image is not None:
            return self.inpainting_pipeline(prompt=prompt, image=base_image, mask_image=mask_image).images[0]
        else:
            return self.text2img_pipeline(prompt).images[0]

    def process_json(self):

        for idx, item in enumerate(self.json_file):
            prompt = item["text"]  # Assuming "text" contains the prompt
            image_path = item.get("image_path")  # Optional: Path for inpainting if available
            print(f"Processing {idx + 1}/{len(self.json_file)}: {prompt}")

            base_image = None
            mask_image = None

            if image_path and os.path.exists(image_path):
                base_image = Image.open(image_path).convert("RGB")
                mask_image = Image.new("L", base_image.size, 255)  # Placeholder mask (full white)

            # Generate image
            image = self.generate_image(prompt, mask_image, base_image)

            # Save image
            output_path = os.path.join(self.output_dir, f"generated_{idx}.png")
            image.save(output_path)
            print(f"Saved: {output_path}")
        
        print("Image generation complete!")

    def run(self):
        """
        Function to execute the full image generation pipeline.
        It loads the input data, processes it, and saves the generated images.
        """
        # Load the input data
        print(f"Loading data from {self.input_json}...")
        self.load_data()

        # Process and generate images for each entry in the data
        print("Starting image generation process...")
        self.process_data()

        print("All images generated and saved!")

#-------------------------------------------------------------------------------------------------------------------------------

data   = .......
output = ....... 
# Object for class1
sdxlBLIP = StableDiffusionWithBLIP(data, output, device= "cuda" if torch.cuda.is_available() else "cpu")
# Object for class2
sdxlCLIP = StableDiffusionWithCLIP(data, output, device= "cuda" if torch.cuda.is_available() else "cpu")

sdxlBLIP.run()
sdxlCLIP.run()