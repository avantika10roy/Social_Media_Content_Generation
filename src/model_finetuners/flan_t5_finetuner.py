## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

import torch
from src.utils.logger import LoggerSetup

from datasets import Dataset

from peft import LoraConfig
from peft import get_peft_model

from transformers import Trainer
from transformers import set_seed
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForSeq2SeqLM


class FLAN_T5_FineTuner:
    """
    A class to fine-tune a LLM using PEFT techniques
    and save the model in .gguf format for faster inference.
    """

    def __init__(self, model_path: str, tokenizer_path: str, dataset: Dataset, finetune_logger: LoggerSetup, **kwargs) -> None:
        """
        Initialize the Fine Tuner.

        Arguments:
        -----------
            model_path     : Path to the directory of saved model or address of the model on Huggingface.
            tokenizer_path : Path to the directory of saved tokenizer or address of the tokenizer on Huggingface.
            dataset        : The dataset provided for fine-tuning.
        
        """
        
        try:

            self.device         = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            
            model_kwargs        = {"device_map": "auto" 
                                   if self.device == "cuda" 
                                   else None,"torch_dtype": torch.float16 
                                   if self.device == "cuda" 
                                   else torch.float32}
            
            self.model          = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
            
    
            if self.device in ["mps", "cpu"]:
                self.model      = self.model.to(self.device)

            self.tokenizer      = AutoTokenizer.from_pretrained(tokenizer_path)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.dataset        = dataset
            self.lora_config    = None
            self.training_args  = None
            self.trainer        = None
            self.logger         = finetune_logger

        except Exception as LLMFineTunerInitializationError:
            self.logger         = finetune_logger
            self.logger.error(repr(LLMFineTunerInitializationError), exc_info = True)

    def define_lora_config(self, **kwargs) -> None:
        """
        Defines the LoRA config.
        If not used, it will not cause any problems.
        
        """
        
        try:
            self.lora_config    = LoraConfig(**kwargs)
            self.model          = get_peft_model(self.model, self.lora_config)

        except Exception as LoraConfigDefinitionError:
            self.logger.error(repr(LoraConfigDefinitionError), exc_info = True)

    def define_training_args(self, **kwargs) -> None:
        """
        Defines the training arguments.
        """
        try:
            
            if 'device_map' in kwargs:
                del kwargs['device_map']
            
            self.training_args   = TrainingArguments(**kwargs)

        except Exception as TrainingArgumentDefinitionError:
            self.logger.error(repr(TrainingArgumentDefinitionError), exc_info = True)

    def define_trainer(self) -> None:
        """
        Defines the trainer used during fine-tuning.
        """
        try:
            train_dataset        = self.dataset["train"] if "train" in self.dataset else self.dataset
            eval_dataset         = self.dataset["test"] if "test" in self.dataset else None

            self.trainer         = Trainer(model          = self.model,
                                           args           = self.training_args,
                                           train_dataset  = train_dataset,
                                           eval_dataset   = eval_dataset,
                                           tokenizer      = self.tokenizer
                                           )
        
        except Exception as TrainerDefinitionError:
            self.logger.error(repr(TrainerDefinitionError), exc_info = True)
            
    def set_model(self, model_path: str) -> None:
        """
        Set up the model for fine-tuning.
        
        Arguments:
        -----------
            model_path: Path to the directory of saved model or address of the model on Huggingface.
        """
        try:
            model_kwargs          = {"device_map"  : "auto" if self.device == "cuda" else None,
                                     "torch_dtype" : torch.float16 if self.device == "cuda" else torch.float32
                                     }
            
            self.model            = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
            
            if self.device in ["mps", "cpu"]:
                self.model        = self.model.to(self.device)

            if self.logger:
                self.logger.info(f"Model loaded successfully from {model_path}")

        except Exception as ModelSetupError:
            if self.logger:
                self.logger.error(repr(ModelSetupError), exc_info = True)


    def start_fine_tuning(self) -> AutoModelForSeq2SeqLM:
        """
        Starts the fine-tuning process.
        
        """
        
        try:
            self.trainer.train()
            
            return self.model

        except Exception as FineTuningStartError:
            self.logger.error(repr(FineTuningStartError), exc_info = True)
            
            return repr(FineTuningStartError)