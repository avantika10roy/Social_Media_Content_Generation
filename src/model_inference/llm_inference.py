# -------- Done By Manu Bhaskar -----------

# -------- Dependencies ----------

import pandas
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

class LLMInference:
    """
    A class that makes inference based on the given model name
    Currently only supports Mistral-7B-Instruct and Falcon-1B-Instruct
    """
    MODEL_PATH = 'base_models/falcon1b/model'
    TOKENIZER_PATH = 'base_models/falcon1b/tokenizer'
    LORAPATH = 'results/llm_results/pipeline_finetuning_v9'
    def __init__(self, model_name:str, prompt_info:dict):
        self.base_model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_PATH)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = PeftModel.from_pretrained(self.base_model, self.LORAPATH)
        

    
    def generate(self, prompt):

        inputs = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(**inputs, max_new_tokens = 250, do_sample = True, top_p = 0.95, top_k = 40, temperature = 0.3, no_repeat_ngram_size=3, repetition_penalty=1.2)

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return generated_text

    

if __name__ == '__main__':
    # prompt_info = {'platform':'facebook','topic':'Republic Day'}
    # infer = LLMInference(model_name='falcon', **prompt_info)
    # infer.generate()
    prompt = """Generate a high-quality, engaging social media post for a business in a descriptive format. Follow the example structure and ensure clarity, creativity, and audience engagement.

    Context:
    - Platform: facebook 
    - Topic: Republic Day
    - Language: English
    - Word Limit: 250

    Requirements:
    - Craft a compelling opening that grabs attention.  
    - Highlight key details about the business or occasion.  
    - Maintain a consistent and engaging tone throughout.  
    - Tone should be determined by the platform.
    - Use persuasive language and storytelling where applicable.  
    - Include a strong call to action (CTA) to encourage engagement.  


    Now, generate a social media post using the provided context."""

    base_model = AutoModelForCausalLM.from_pretrained('../base_models/falcon1b/model')
    tokenizer = AutoTokenizer.from_pretrained('./base_models/falcon1b/tokenizer')

    # Load LoRA adapters on top of the base model
    model = PeftModel.from_pretrained(base_model, '../results/llm_results/fine_tuning_results_v1/checkpoint-115')

    # Move to MPS (Mac) or CPU
    # device = "mps"  # Use "cpu" if you don’t have Metal enabled
    # model.to(device)

    # Function for inference
    def generate_text(prompt, max_length=100):
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # Example usage
    # prompt = "Once upon a time"
    result = generate_text(prompt)
    print(result)