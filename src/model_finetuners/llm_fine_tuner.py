# ---- Done By Manu Bhaskar ------

# ---- Dependencies -----

import pandas
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from src.utils.logger import LoggerSetup
from transformers import TrainingArguments, Trainer, set_seed # use set_seed during the calling of the class in beginning of the script
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig


class LLMFineTuner:
    """
    A class to fine tune a LLM using peft techniques
    and save the model in .gguf format for faster inference 
    """
    def __init__(self, model_path:str , tokenizer_path:str, dataset:Dataset, finetune_logger:LoggerSetup, **kwargs) -> None:
        """
        The Initialization of Fine Tuner

        Arguments:
        --------------
            model_path     : Path to the directory of saved model or address of the model on Huggingface

            tokenizer_path : Path to the dirrectory of saved tokenizer or address of the tokenizer on Huggingface

            dataset        : The dataset provided for finetuning
        """
        try:

            self.model         = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer     = AutoTokenizer.from_pretrained(tokenizer_path)
            self.dataset       = dataset
            self.lora_config   = None
            self.training_args = None
            self.trainer       = None
            self.logger        = finetune_logger

        except Exception as LLMFineTunerInitializationError:
            self.logger        = finetune_logger
            self.logger.error(repr(LLMFineTunerInitializationError), exc_info=True)
            return repr(LLMFineTunerInitializationError)


    def define_lora_config(self, **kwargs) -> None:
        """
        Defines the lora config
        If not used will not ause any problems
        """
        try:
            self.lora_config   = LoraConfig(**kwargs)
            self.model         = get_peft_model(self.model, self.lora_config)
            
        except Exception as LoraConfigDefinitionError:
            self.logger.error(LoraConfigDefinitionError, exc_info=True)
            return repr(LoraConfigDefinitionError)


    def define_training_args(self, **kwargs) -> None:
        """
        Defines the training arguments
        """
        try:
            self.training_args  = TrainingArguments(**kwargs)
        except Exception as TrainingArgumentDefinitionError:
            self.logger.error(repr(TrainingArgumentDefinitionError), exc_info=True)
            return repr(TrainingArgumentDefinitionError)


    def define_trainer(self) -> None:
        """
        Defines the trainer that is used during fine tuning
        """
        try:
            self.trainer         = Trainer(model            = self.model,
                                           args             = self.training_args,
                                           train_dataset    = self.dataset['train'],
                                           eval_dataset     = self.dataset['test'],
                                           processing_class = self.tokenizer
                                            )
        except Exception as TrainerDefinitionError:
            self.logger.error(repr(TrainerDefinitionError), exc_info=True)
            return repr(TrainerDefinitionError)


    def start_fine_tuning(self) -> AutoModelForCausalLM:
        """
        Strats the fine tuning process
        """
        try:
            self.trainer.train()
            return self.model

        except Exception as FineTuningStartError:
            self.logger.error(repr(FineTuningStartError), exc_info=True)
            return repr(FineTuningStartError)