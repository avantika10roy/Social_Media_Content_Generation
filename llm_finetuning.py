# ---------- Done By Manu Bhaskar & Arnab Chatterjee -------------

# ---------- Dependencies ------------
import os 
import json
import torch
from datasets import Dataset
from config.config import Config
from transformers import set_seed
from src.utils.logger import LoggerSetup
from src.utils.set_seed import set_global_seed
from sklearn.model_selection import train_test_split
from src.model_finetuners.llm_fine_tuner import LLMFineTuner
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompts.prompts import llm_finetuning_prep

finetune_logger = LoggerSetup(logger_name="llm_fine_tuner.py", log_filename_prefix="llm_fine_tuner").get_logger()
finetune_logger.info("Logger Successfully Initialized")



def llm_fine_tune_main(logger:LoggerSetup) -> None:
    """
    Main function to run the fine tuner of llm.
    
    """
    set_global_seed(logger=logger, seed=42)
    try:
        model_path         = 'src/base_models/falcon1b/model'
        tokenizer_path     = 'src/base_models/falcon1b/tokenizer'

        if (not os.path.isdir(model_path)) or (not os.path.isdir(tokenizer_path)):
            model = AutoModelForCausalLM.from_pretrained('tiiuae/Falcon3-1B-Instruct', device_map='cpu')
            tokenizer = AutoTokenizer.from_pretrained('tiiuae/Falcon3-1B-Instruct')

            model.save_pretrained(model_path)
            tokenizer.save_pretrained(tokenizer_path)
        # No need to write this part since model will be initialized in llm_fine_tuner.py if saved in path 
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu')
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        with open(Config.MIXED_CURATED_DATA_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Data Preparation
        for item in data:
            item.pop('image_paths',None)

        data_list = [llm_finetuning_prep(item) for item in data]


        data = Dataset.from_dict({'texts':data_list})
        
        def tokenize_function(examples):
            inputs = tokenizer(examples["texts"], padding="max_length", truncation=True, return_tensors='pt', max_length=256)
            inputs["labels"] = inputs["input_ids"].clone().detach() # Add labels for causal LM training
            return inputs
        
        tokenized_dataset = data.map(tokenize_function, batched=True)
        split_dataset = tokenized_dataset.train_test_split(test_size=0.2)


        fine_tuner = LLMFineTuner(model_path=model_path,
                                  tokenizer_path=tokenizer_path,
                                  dataset=split_dataset,
                                  finetune_logger=finetune_logger)
        
        training_args = {
            'output_dir' : 'results/llm_results/fine_tuning_results_v1',
            'learning_rate': 2e-7,
            'warmup_steps' : 100,
            'per_device_train_batch_size' : 1,
            'per_device_eval_batch_size' :1,
            'gradient_accumulation_steps':8,
            'num_train_epochs':5,
            'weight_decay':0.01,
            'bf16':False,
            'fp16':False,
            'eval_strategy':"epoch",
            'use_cpu':True
        }

        lora_args = {
            'r' : 4,
            'lora_alpha' : 8,
            'target_modules' : ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            'bias' :"none",
            'lora_dropout':0.05,
            'task_type' : "CAUSAL_LM"
        }

        fine_tuner.define_lora_config(**lora_args)
        fine_tuner.define_training_args(**training_args)
        fine_tuner.use_mps()
        fine_tuner.define_trainer()
        model = fine_tuner.start_fine_tuning()

    except Exception as MainError:
        finetune_logger.error(repr(MainError), exc_info=True)
    pass

if __name__ == '__main__':
    finetune_logger.info("Test Run")
    llm_fine_tune_main(logger=finetune_logger)
    finetune_logger.info("Test Run Successfully")
