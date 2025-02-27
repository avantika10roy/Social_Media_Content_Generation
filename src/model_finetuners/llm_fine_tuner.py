# ---- Done By Manu Bhaskar ------

# ---- Dependencies -----

import torch
from datasets import Dataset
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from src.utils.logger import LoggerSetup
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM


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
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
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

    def use_mps(self):
        self.model.to("mps")
    
    def use_mps_mistral(self):
        for name, param in self.model.named_parameters():
            if ("q_proj" in name) or ("k_proj" in name) or ("v_proj" in name) or ("o_proj" in name):# or ("gate_proj" in name) or ("up_proj" in name) or ("down_proj" in name):
                param.data = param.data.to("mps")

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
        


class LLMTrainer:
    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer, device:str, logger:LoggerSetup, lr:int=5e-5, gradient_accumulation_steps:int=1):
        try:
            self.model              = model
            self.tokenizer          = tokenizer
            self.device             = device
            self.logger             = logger
            self.lr                 = lr
            self.accumulation_steps = gradient_accumulation_steps

        except Exception as InitializationError:
            self.logger             = logger
            self.logger.error(repr(InitializationError), exc_info=True)
            return repr(InitializationError)
        
    def lora_model_setup(self, r:int, lora_alpha: int, target_modules: list, bias: str, lora_dropout:int, task_type:str):
        try:
            self.logger.info("Setting up lora adapters.")
            self.lora_config   = LoraConfig(r=r, 
                                            lora_alpha     = lora_alpha, 
                                            target_modules = target_modules, 
                                            bias           = bias, 
                                            lora_dropout   = lora_dropout,
                                            task_type      = task_type)
            self.model         = get_peft_model(self.model, self.lora_config)
            self.model         = self.model.to(self.device)
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
            self.logger.info("Lora setup completed.")
        except Exception as LoraSetupError:
            self.logger.error(repr(LoraSetupError), exc_info=True)
            return repr(LoraSetupError)

    def train_step(self, batch):
        inputs                 = {key: batch[key].to(self.device) for key in batch}
        outputs                = self.model(**inputs)
        loss                   = outputs.loss / self.accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        return loss.item()
    
    def save_model(self, path:str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)