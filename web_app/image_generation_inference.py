# -------- DONE BY ARNAB CHATERJEE AND SUBHAS MUKHERJEE -------

# DEPENDENCIES

import torch
import os
import base64
import io
from typing import Optional
from PIL import Image, ImageDraw
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pyngrok import ngrok
from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
import nest_asyncio
import uvicorn

class KandinskyInpainting:
    """
    A class to handle image inpainting using Kandinsky 2.2.
    """
    def __init__(self, model_id="kandinsky-community/kandinsky-2-2-decoder-inpaint"):
        """Initialize the inpainting model and scheduler."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
    
    def generate_image(self, prompt: str, negative_prompt: str, image_size=(1024, 1024), 
                       base_bg_color=(255, 255, 255), inference_steps=30, mask_position='left'):
        """
        Generates an image with inpainting.
        """
        mask_area = (20, 20, image_size[0] * 0.3, image_size[1] - 20) if mask_position == 'left' \
                    else (image_size[0] - image_size[0] * 0.3, 20, image_size[0], image_size[1] - 20)
        
        base_image = Image.new("RGB", image_size, base_bg_color)
        mask = Image.new("L", image_size, 255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(mask_area, fill=0)
        
        seed = torch.seed()
        generator = torch.Generator().manual_seed(seed)
        
        image = self.pipe(prompt=prompt, image=base_image, mask_image=mask, negative_prompt=negative_prompt,
                          prior_guidance_scale=1, generator=generator, output_type="pil",
                          height=image_size[1], width=image_size[0], num_inference_steps=inference_steps).images[0]
        
        return image, seed

class ImageGeneratorAPI:
    """
    A class to handle FastAPI endpoints for image generation.
    """
    def __init__(self):
        """Initialize FastAPI app and Kandinsky model."""
        self.app = FastAPI()
        self.inpainter = KandinskyInpainting()
        nest_asyncio.apply()
        self.setup_routes()
    
    def setup_routes(self):
        """Define API routes."""
        @self.app.get("/")
        def home():
            return {"message": "Use /generate to create images"}
        
        @self.app.get("/generate")
        def generate(prompt: str, negative_prompt: Optional[str] = "blur", mask_position: Optional[str] = "right",
                     height: Optional[int] = 1024, width: Optional[int] = 1024, base_bg_color: Optional[tuple] = (255, 255, 255),
                     inference_steps: Optional[int] = 30):
            """
            API endpoint to generate an image using inpainting.
            """
            try:
                image_size = (width, height)
                image, seed = self.inpainter.generate_image(prompt, negative_prompt, image_size, base_bg_color, inference_steps, mask_position)
                output_path = "output.png"
                image.save(output_path)
                
                if not os.path.exists(output_path):
                    return {"error": "Generated image not found."}
                
                return FileResponse(output_path, media_type="image/jpeg")
            except Exception as e:
                return {"error": str(e)}
    
    def run(self, port=8000):
        """Run the FastAPI server with ngrok."""
        public_url = ngrok.connect(port).public_url
        print(f"🚀 API is live at: {public_url}")
        uvicorn.run(self.app, host="0.0.0.0", port=port)

# Run the API
if __name__ == "__main__":
    api = ImageGeneratorAPI()
    api.run()
