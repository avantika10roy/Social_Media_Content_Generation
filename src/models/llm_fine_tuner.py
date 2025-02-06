# ---- Done By Manu Bhaskar ------

# Dependencies

import transformers
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

class LLMFineTuner:
    """
    A class to fine tune a LLM using peft techniques
    and save the model in .gguf format for faster inference 
    """
    def __init__(self, model_path:str ,**kwargs):
        """The Initialization of Fine Tuner
        --------------
        """
    pass