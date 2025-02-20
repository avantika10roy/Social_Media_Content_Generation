import numpy as np
import random
import time
import torch
import pandas
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig, set_seed

seed = 42
np.random.seed(seed)
random.seed(seed)
set_seed(seed=seed)
torch.manual_seed(seed)
torch.mps.manual_seed(seed)

# The topic should be decriptive enough so that the model does not hallucinate
prompt = """
    Instruction:
    Generate a high-quality, engaging and professional social media post for a company called "Itobuz Private Limited" in a descriptive format.
    Follow the example structure, fulfil the requirements and ensure clarity, creativity, context awareness, and audience engagement.
    
    Context:
    - Platform: Facebook
    - Topic: Republic Day Greetings for year 2025
    - Language: English
    - Word Limit: 250

    Requirements:
    - Craft a compelling opening that grabs attention.  
    - Highlight key details about the business or occasion.  
    - Maintain a consistent and engaging tone throughout.  
    - Usage of emoji sould depend on the platform.
    - Tone should be determined by the platform.
    - Don't put any placeholders.
    - Use persuasive language and storytelling where applicable.  
    - Include a strong call to action (CTA) to encourage engagement.
    - Generate only five hashtags.
    - Generated post should be ready to be posted online."""

base_model = AutoModelForCausalLM.from_pretrained('src/base_models/falcon1b/model')
tokenizer = AutoTokenizer.from_pretrained('src/base_models/falcon1b/tokenizer')

# Load LoRA adapters on top of the base model
model = PeftModel.from_pretrained(base_model, 'results/llm_results/pipeline_finetuning_v9')

# Move to MPS (Mac) or CPU
# device = "mps"  # Use "cpu" if you don’t have Metal enabled
# model.to(device)

# Function for inference
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens = 250, do_sample = True, top_p = 0.95, top_k = 40, temperature = 0.3, no_repeat_ngram_size=3, repetition_penalty=1.2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
# prompt = "Once upon a time"
start = time.time()
result = generate_text(prompt)
end = time.time()
print(result)
print(f"\nTime Taken:{end-start}")