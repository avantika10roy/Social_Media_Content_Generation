# ------Doane by Manu Bhaskar --------

# ------ Dependencies --------
import os
import json
import torch
import numpy as np
import pandas as pd
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
    def __init__(self, model_path:str, tokenizer_path:str, device:torch.device, logger:LoggerSetup, lr:int=5e-5, model_name:str = None):
        try:
            if (model_path is None) and (model_name is None):
                raise
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path
            self.model_name = model_name
            self.device = device
            self.logger = logger
            self.optimizer = AdamW(self.model.parameters(), lr=lr)

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
        
    def lora_model_setup(self, r:int, lora_alpha: int, target_modules: list, bias: str, lora_dropout:int, task_type:str):
        self.lora_config   = LoraConfig(r=r, 
                                        lora_alpha=lora_alpha, 
                                        target_modules=target_modules, 
                                        bias = bias, 
                                        lora_dropout=lora_dropout,
                                        task_type=task_type)
        self.model         = get_peft_model(self.model, self.lora_config)
    
    def train_step(self, batch):
        self.model.train()
        inputs = {key: batch[key].to(self.device) for key in batch}
        outputs = self.model(**inputs)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()


# --------- FineTuning ---------
class LLMFineTuning:
    pass


# --------- Main ------------
if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("src/base_models/falcon1b/tokenizer")
    # tokenizer.pad_token = tokenizer.eos_token 
    # tokenizer.padding_side = "left"
    # dataset = CustomDatasetForLLMFineTuning(json_file_path=Config.MIXED_CURATED_DATA_PATH, tokenizer=tokenizer, transform=llm_finetuning_prep, max_length=256)
    # dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False)
    # for batch in dataloader:
    #     print(batch['input_ids'][0]==batch['labels'][0])
    #     print(batch)
    #     break
    
    # with open(Config.MIXED_CURATED_DATA_PATH) as file:
    #     data = json.load(file)

    # l = [llm_finetuning_prep(item) for item in data]
    # n = [len(i.split(" ")) for i in l]
    
    pass