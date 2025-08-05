# Dependencies
import os
import clip
import time
import json
import torch
import lpips
import numpy as np
import pandas as pd
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import models
from torchvision import transforms
from torchvision.models import inception_v3


class ImageGenerationEvaluator:
    """
    A class for evaluating image generation models using CLIP Score, FID Score, and LPIPS Score
    """
    def __init__(self, device: str = None) -> None:
        """
        Initializes the evaluator by loading necessary models.

        Arguments:
        ----------
            device { str } : The device to run computations on ('cuda', 'mps', or 'cpu')
        """
        try:
            if (device == None):
                self.device = device if device else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            else:
                self.device = device

            # Load CLIP Model
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", self.device)

            # Load LPIPS Model
            self.lpips_model                      = lpips.LPIPS(net="alex").to(self.device)

            # Load InceptionV3 for FID
            self.inception_model                  = inception_v3(pretrained      = True, 
                                                                 transform_input = False).to(self.device).eval()

            # Common Image Transform
            self.transform                        = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
        
        except Exception as e:
            print(f"Error initializing models: {repr(e)}")


    def compute_clip_score(self, image_path: str, prompt: str) -> float:
        """
        Computes CLIP Score to measure text-image alignment

        Arguments:
        ----------
            image_path { str } : Path to the generated image
            
            prompt     { str } : Text prompt used for image generation
        
        Returns:
        ---------
                 { float }     :  CLIP similarity score
        """
        try:
            image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            text  = clip.tokenize([prompt]).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                text_features  = self.clip_model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features  /= text_features.norm(dim=-1, keepdim=True)

            similarity      = (image_features @ text_features.T).item()
            
            return similarity
        
        except Exception as e:
            print(f"Error computing CLIP score: {e}")
            
            return 0.0


    def compute_fid(self, real_image: Image.Image, generated_image: Image.Image) -> float:
        """
        Computes FID Score between a single pair of real and generated images

        Arguments:
        ----------
            real_image      { Image.Image } : The original real image

            generated_image { Image.Image } : The generated image

        Returns:
        --------
                      { float }             : FID score
        """
        try:
            def get_activations(image):
                img = self.transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    act = self.inception_model(img).cpu().numpy()
                
                return act


            act_real            = get_activations(real_image)
            act_gen             = get_activations(generated_image)

            mu_real, sigma_real = act_real.mean(axis = 0), np.cov(act_real, rowvar = False)
            mu_gen, sigma_gen   = act_gen.mean(axis = 0), np.cov(act_gen, rowvar = False)

            diff                = mu_real - mu_gen
            cov_mean            = sqrtm(sigma_real @ sigma_gen)

            if (np.iscomplexobj(cov_mean)):
                cov_mean = cov_mean.real

            fid_score = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * cov_mean)
            
            return fid_score
        
        except Exception as e:
            print(f"Error computing FID score: {e}")
            
            return float('inf')


    def compute_lpips(self, real_image_path: str, generated_image_path: str) -> float:
        """
        Computes LPIPS Score to measure perceptual similarity
        
        Arguments:
        ----------
            real_image_path      { str } : Path to the real image
            
            generated_image_path { str } : Path to the generated image

        Returns:
        --------
                         { float }       : LPIPS perceptual similarity score
        """
        try:
            transform   = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

            image_1     = transform(Image.open(real_image_path)).unsqueeze(0).to(self.device)
            image_2     = transform(Image.open(generated_image_path)).unsqueeze(0).to(self.device)

            lpips_score = self.lpips_model(image_1, image_2).item()

            return lpips_score

        except Exception as e:
            print(f"Error computing LPIPS score: {e}")
            
            return float('inf')


    def measure_inference_time(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
        """
        Measures inference time for a given model

        Arguments:
        ----------
            model      { torch.nn.Module } : The PyTorch model
            
            input_tensor { torch.Tensor }  : The input tensor for inference

        Returns:
        --------
                         { float }         : Time taken for inference in seconds
        """
        try:
            model          = model.to(self.device).eval()
            input_tensor   = input_tensor.to(self.device)

            start_time     = time.time()
            
            with torch.no_grad():
                model(input_tensor)
            
            end_time       = time.time()

            inference_time = (end_time - start_time)

            return time_taken

        except Exception as e:
            print(f"Error measuring inference time: {e}")
            return float('inf')


def generate_image(prompt: str, save_path: str):
    """
    Placeholder for your image generation model
    """
    # Replace this with your actual image generation logic
    pass


def evaluate_images(json_file: str, output_csv: str = "evaluation_results.csv") -> None:
    """
    Evaluates images using CLIP Score, FID Score, LPIPS Score, and inference time

    Arguments:
    ----------
    json_file  { str } : Path to the input JSON file containing image paths and prompts

    output_csv { str } : Path to save the evaluation results in CSV format
    """

    evaluator = ImageGenerationEvaluator()

    # Load JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    results = list()
    
    for entry in data:
        prompt                   = entry["prompt"]
        real_image_path          = entry["real_image"]
        generated_image_path     = f"generated_images/{os.path.basename(real_image_path)}"

        # Generate Image
        generate_image(prompt    = prompt, 
                       save_path = generated_image_path)

        # Compute Metrics
        clip_score               = evaluator.compute_clip_score(generated_image_path, prompt)
        fid_score                = evaluator.compute_fid(Image.open(real_image_path), Image.open(generated_image_path))
        lpips_score              = evaluator.compute_lpips(real_image_path, generated_image_path)

        # Measure Inference Time
        inference_time           = evaluator.measure_inference_time(dummy_model, dummy_input)

        # Save Results
        entry["generated_image"] = generated_image_path
        entry["clip_score"]      = clip_score
        entry["fid_score"]       = fid_score
        entry["lpips_score"]     = lpips_score
        entry["inference_time"]  = inference_time

        results.append(entry)


    # Save results as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Append results back to JSON
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation completed! Results saved to {output_csv} and updated in {json_file}.")


# Run Evaluation
evaluate_images("data.json")
