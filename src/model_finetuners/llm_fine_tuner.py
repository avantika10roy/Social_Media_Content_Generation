# ---- Done By Manu Bhaskar ------

# Dependencies

import pandas
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, set_seed # use set_seed during the calling of the class in beginning of the script
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

class LLMFineTuner:
    """
    A class to fine tune a LLM using peft techniques
    and save the model in .gguf format for faster inference 
    """
    def __init__(self, model_path:str , tokenizer_path:str, dataset: Dataset,**kwargs):
        """The Initialization of Fine Tuner
        Arguments:
        --------------
            model_path     : Path to the saved model or address of the model on Huggingface

            tokenizer_path : Path to the saved tokenizer or address of the tokenizer on Huggingface

            dataset        : The dataset provided for finetuning
        """
        self.model         = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer     = AutoTokenizer.from_pretrained(tokenizer_path)
        self.dataset       = dataset
        self.lora_config   = None
        self.training_args = None
        self.trainer       = None

    def define_lora_config(self, **kwargs) -> None:
        """
        Defines the lora config
        If not used will not ause any problems
        """
        try:
            self.lora_config   = LoraConfig(**kwargs)
            self.model         = get_peft_model(self.model, self.lora_config)
            
        except Exception as e:
            print(e)
    def define_training_args(self, **kwargs) -> None:
        """
        Defines the training arguments
        """
        try:
            self.training_args = TrainingArguments(**kwargs)
        except Exception as e:
            print(e)
    def define_trainer(self) -> None:
        """
        Defines the trainer that is used during fine tuning
        """
        try:
            self.trainer = Trainer(
                model          = self.model,
                args           = self.training_args,
                train_dataset  = self.dataset['train'],
                eval_dataset   = self.dataset['test'],
                tokenizer      = self.tokenizer
            )
        except Exception as e:
            print(e)

    def start_fine_tuning(self) -> AutoModelForCausalLM:
        """
        Strats the fine tuning process
        """
        try:
            self.trainer.train()

            return self.model
        except Exception as e:
            print(e)