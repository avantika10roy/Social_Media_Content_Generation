# ------- Done By manu Bhaskar ---------


# ----- Dependencies -------
import json
import torch
from typing import Optional, Callable
from torch.utils.data import Dataset
from transformers import AutoTokenizer

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
        
        sample = self.json_data[idx]
        data       = self.transform(sample)
        input      = self.tokenizer(data, 
                                    padding        = "max_length", 
                                    truncation     = True, 
                                    return_tensors = 'pt', 
                                    max_length     = self.max_len)
        labels = input["input_ids"].clone().detach()
        shifted_lables = torch.cat((labels[:, 1:], torch.tensor([[11]], dtype=labels.dtype)), dim=-1)
        input["labels"] = shifted_lables
        return {key: val.squeeze(0) for key, val in input.items()}