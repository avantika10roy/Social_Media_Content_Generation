# -------- Done By Manu Bhaskar -----------

# -------- Dependencies ----------
import re
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

class LLMInference:
    """
    A class that makes inference based on the given model name
    Currently only supports Mistral-7B-Instruct and Falcon-1B-Instruct
    """
    def __init__(self, model_path, tokenizer_path, lora_path):
        self.base_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)

    def generate(self, prompt):

        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors='pt')

        with torch.no_grad():
            output = self.model.generate(**inputs, 
                                         min_new_tokens       = 100,
                                         max_new_tokens       = 250, 
                                         do_sample            = True, 
                                         top_p                = 0.95, 
                                         top_k                = 40, 
                                         temperature          = 0.3, 
                                         no_repeat_ngram_size = 3, 
                                         repetition_penalty   = 1.2,
                                         pad_token_id         = self.tokenizer.eos_token_id,
                                         eos_token_id         = self.tokenizer.eos_token_id)

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        match          = re.search(r"--\s*(.*)", generated_text, re.DOTALL)

        if match:
            clean_text = match.group(1).strip()
        else:
            clean_text = generated_text
        
        return clean_text

