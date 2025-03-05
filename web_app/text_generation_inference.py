# Dependencies
import os
import torch
import logging
from peft import PeftModel
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SocialMediaPostGenerator:
    def __init__(self, model_path: str, device: str = "auto"):
        self.device                = self._set_device(device)
        self.tokenizer, self.model = self.load_model(model_path)


    def _set_device(self, device: str) -> str:
        if (device == "auto"):
            if torch.cuda.is_available():
                return "cuda"
            
            elif torch.backends.mps.is_available():
                return "mps"
            
            else:
                return "cpu"
        
        return device


    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

        logging.info(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        logging.info(f"Loading model from {model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_path,
                                                          torch_dtype                   = torch.float16 if self.device == "cuda" else torch.float32
                                                         )

        if (not hasattr(base_model, "peft_config")):
            logging.info("Applying LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, model_path)
        
        else:
            model = base_model
            logging.info("LoRA adapter already applied.")

        model.to(self.device)
        model.eval()
        return tokenizer, model

    def generate_post(self, occasion: str, brief: str, platform: str, target_audience: str, tone: str, extra_details: str = "", max_length: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        prompt = (f"Generate a high-quality social media post based on the following details:\n"
                  f"- Occasion: {occasion}\n"
                  f"- Brief: {brief}\n"
                  f"- Platform: {platform}\n"
                  f"- Target Audience: {target_audience}\n"
                  f"- Tone: {tone}\n"
                  f"- Extra Details: {extra_details}\n\n"
                  f"### Response:\n")

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length, temperature=temperature, top_p=top_p, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)

        post = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return post.split("### Response:\n")[-1].strip()

# Initialize Model
MODEL_PATH = "models/zephyr_7b_finetuned"
generator  = SocialMediaPostGenerator(MODEL_PATH)
