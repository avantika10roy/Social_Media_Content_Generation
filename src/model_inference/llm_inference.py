# -------- Done By Manu Bhaskar -----------

# -------- Dependencies ----------

import pandas
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from src.base_models import FALCON3_1B_INSTRUCT, MISTRAL_7B_INSTRUCT
from transformers import TrainingArguments, Trainer, set_seed # use set_seed during the calling of the class in beginning of the script
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

class LLMInference:
    """
    A class that makes inference based on the given model name
    Currently only supports Mistral-7B-Instruct and Falcon-1B-Instruct
    """
    FALCON3_1B_INSTRUCT = FALCON3_1B_INSTRUCT
    MISTRAL_7B_INSTRUCT = MISTRAL_7B_INSTRUCT
    def __init__(self, model_name:str):
        if (model_name.lower() == 'falcon3-1b-instruct'):
            self.model = AutoModelForCausalLM.from_pretrained(FALCON3_1B_INSTRUCT)
            self.tokenizer = AutoTokenizer.from_pretrained(FALCON3_1B_INSTRUCT)
        elif (model_name.lower() == 'mistral-1b-instruct'):
            self.model = AutoModelForCausalLM.from_pretrained(MISTRAL_7B_INSTRUCT)
            self.tokenizer = AutoTokenizer.from_pretrained(MISTRAL_7B_INSTRUCT)
        
        pass

    pass