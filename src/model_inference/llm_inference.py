# -------- Done By Manu Bhaskar -----------

# -------- Dependencies ----------

import pandas
import transformers
from datasets import Dataset
from src.prompts.prompts import llm_prompt
from peft import LoraConfig, get_peft_model
from src.base_models import FALCON3_1B_INSTRUCT, MISTRAL_7B_INSTRUCT
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

class LLMInference:
    """
    A class that makes inference based on the given model name
    Currently only supports Mistral-7B-Instruct and Falcon-1B-Instruct
    """
    FALCON3_1B_INSTRUCT = FALCON3_1B_INSTRUCT
    MISTRAL_7B_INSTRUCT = MISTRAL_7B_INSTRUCT
    def __init__(self, model_name:str, prompt_info:dict):
        if (model_name.lower() == 'falcon3-1b-instruct'):
            self.model = AutoModelForCausalLM.from_pretrained(FALCON3_1B_INSTRUCT)
            self.tokenizer = AutoTokenizer.from_pretrained(FALCON3_1B_INSTRUCT)
        elif (model_name.lower() == 'mistral-1b-instruct'):
            self.model = AutoModelForCausalLM.from_pretrained(MISTRAL_7B_INSTRUCT)
            self.tokenizer = AutoTokenizer.from_pretrained(MISTRAL_7B_INSTRUCT)
        
        self.prompt = llm_prompt(**prompt_info)
    
        
        pass

    pass