# ------Doane by Manu Bhaskar --------

# ------ Dependencies --------
import io
import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from config.config import Config
from torch.optim import AdamW
from huggingface_hub import login
from typing import Optional, Callable
from src.utils.set_seed import set_seed
from src.utils.logger import LoggerSetup
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from src.prompts.prompts import llm_finetuning_prep
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------ Custom Dataset class --------

class CustomDatasetForLLMFineTuning(Dataset):
    """Fine-tuning dataset for Large Language Model"""

    def __init__(self, json_file_path:str, tokenizer:AutoTokenizer, transform:Optional[Callable] = None, max_length:int=512):
        """
        Initialization of CustomDataset class
        Arguments:
        -------------
            json_file_path {str}       : Path to the json file containing the data for fine tuning

            transform      {callable}  : Callable function which will be applied on the data 
        """
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len   = max_length
        with open(json_file_path, 'r', encoding="utf-8") as file:
            self.json_data = json.load(file)

    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # self.json_data = np.array(self.json_data)
        sample = self.json_data[idx]
        data       = self.transform(sample)
        input      = self.tokenizer(data, 
                                    padding        = "max_length", 
                                    truncation     = True, 
                                    return_tensors = 'pt', 
                                    max_length     = self.max_len)
        labels = input["input_ids"].clone().detach()
        shifted_lables = torch.cat((labels[:, 1:], torch.tensor([[-100]], dtype=labels.dtype)), dim=-1)
        input["labels"] = shifted_lables
        return {key: val.squeeze(0) for key, val in input.items()}


# ---------- Trainer ---------------

class LLMTuner:
    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer, device:str, logger:LoggerSetup, lr:int=5e-5, gradient_accumulation_steps:int=1):
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.logger = logger
            self.lr = lr
            self.accumulation_steps = gradient_accumulation_steps

        except Exception as InitializationError:
            self.logger = logger
            self.logger.error(repr(InitializationError), exc_info=True)
            return repr(InitializationError)
        
    def lora_model_setup(self, r:int, lora_alpha: int, target_modules: list, bias: str, lora_dropout:int, task_type:str):
        try:
            self.logger.info("Setting up lora adapters.")
            self.lora_config   = LoraConfig(r=r, 
                                            lora_alpha=lora_alpha, 
                                            target_modules=target_modules, 
                                            bias = bias, 
                                            lora_dropout=lora_dropout,
                                            task_type=task_type)
            self.model         = get_peft_model(self.model, self.lora_config)
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
            self.logger.info("Lora setup completed.")
        except Exception as LoraSetupError:
            self.logger.error(repr(LoraSetupError), exc_info=True)
            return repr(LoraSetupError)

    def train_step(self, batch):
        inputs = {key: batch[key].to(self.device) for key in batch}
        outputs = self.model(**inputs)
        loss = outputs.loss / self.accumulation_steps
        loss.backward()
        return loss.item()
    
    def save_model(self, path:str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

# --------- FineTuning ---------
class LLMFineTuning:
    def __init__(self, model_path:str, tokenizer_path:str, device:str, logger:LoggerSetup, epochs:int, results_dir:str, model_name:str=None):
        try:
            if (model_path is None) and (model_name is None):
                raise
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path
            self.model_name = model_name
            self.device = device
            self.logger = logger
            self.epochs = epochs
            self.results_dir = results_dir

            if (not os.path.isdir(model_path)) or (not os.path.isdir(tokenizer_path)):
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"

                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(tokenizer_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"

        except Exception as InitializationError:
            self.logger = logger
            self.logger.error(repr(InitializationError), exc_info=True)
            return repr(InitializationError)
    
    def setup(self, batch_size:int, data_file_path:str, lr:int=5e-5, setup_lora:bool=True, gradient_accumulation_steps:int=1):
        try:
            self.accumulation_steps = gradient_accumulation_steps
            self.dataset = CustomDatasetForLLMFineTuning(json_file_path=data_file_path,
                                                         tokenizer=self.tokenizer,
                                                         transform=llm_finetuning_prep,
                                                         max_length=256)

            self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=False)

            self.tuner = LLMTuner(model=self.model,
                                  tokenizer=self.tokenizer,
                                  device=self.device, 
                                  logger=self.logger,
                                  lr=lr,
                                  gradient_accumulation_steps=self.accumulation_steps)
            
            if setup_lora:
                self.tuner.lora_model_setup(r=4,
                                            lora_alpha=8,
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                            bias="none",
                                            lora_dropout=0.0,
                                            task_type="CAUSAL_LM")
        
        except Exception as SetupError:
            self.logger.error(repr(SetupError), exc_info=True)
            return repr(SetupError)

    def train(self):
        step_counter = 0
        self.tuner.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            self.tuner.optimizer.zero_grad()

            buffer = io.StringIO()
            
            for batch in tqdm(self.dataloader, desc="Training Progress", file=buffer):
                loss = self.tuner.train_step(batch=batch)
                total_loss+=loss
                step_counter += 1

                if step_counter % self.accumulation_steps == 0:
                    self.tuner.optimizer.step()
                    self.tuner.optimizer.zero_grad()

            self.logger.info(buffer.getvalue())
            avg_loss = total_loss/len(self.dataloader)
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Total Loss : {total_loss}, Average Loss: {avg_loss:.4f}")
            
            self.tuner.save_model(self.results_dir)
        


# --------- Main ------------
if __name__ == "__main__":
    finetune_logger = LoggerSetup(logger_name="llm_fine_tuner_pipeline.py", log_filename_prefix="llm_fine_tuner_pipeline").get_logger()
    finetune_logger.info("Logger Successfully Initialized")
    fine_tuning = LLMFineTuning(model_path='src/base_models/falcon1b/model',
                                tokenizer_path='src/base_models/falcon1b/tokenizer',
                                device="cpu",
                                logger=finetune_logger,
                                epochs=5,
                                results_dir="results/llm_results/pipeline_finetuning/"
                                )
    fine_tuning.setup(batch_size=2,
                      data_file_path=Config.MIXED_CURATED_DATA_PATH)
    fine_tuning.train()
    