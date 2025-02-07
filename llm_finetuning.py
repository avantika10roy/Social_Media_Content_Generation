# ---------- Done By Manu Bhaskar -------------

# ---------- Dependencies ------------
import os
import json
from datasets import Dataset, DatasetDict
from config.config import Config
from transformers import set_seed
from src.utils.logger import LoggerSetup
from sklearn.model_selection import train_test_split
from src.model_finetuners.llm_fine_tuner import LLMFineTuner
from transformers import AutoTokenizer, AutoModelForCausalLM

finetune_logger = LoggerSetup(logger_name="llm_fine_tuner.py", log_filename_prefix="llm_fine_tuner").get_logger()
finetune_logger.info("Logger Successfully Initialized")



def llm_ftmain():
    """
    Main function to run the fine tuner of llm.
    
    """
    set_seed(42)
    try:
        model_path         = 'src/base_models/falcon1b/model'
        tokenizer_path     = 'src/base_models/falcon1b/tokenizer'

        if (not os.path.isdir(model_path)) or (not os.path.isdir(tokenizer_path)):
            model = AutoModelForCausalLM.from_pretrained('tiiuae/Falcon3-1B-Instruct')
            tokenizer = AutoTokenizer.from_pretrained('tiiuae/Falcon3-1B-Instruct')

            model.save_pretrained(model_path)
            tokenizer.save_pretrained(tokenizer_path)

        with open("data/cleaned_data/linkedin_cleaned_data.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        for item in data:
            item.pop('image_paths',None)

        data_list = []
        for item in data:
            text = ""
            for key, value in item.items():
                text = text + "".join(value)
            data_list.append(text)
            
        
        train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
        train_dataset = Dataset.from_list(train_data)
        test_dataset  = Dataset.from_list(test_data)

        dataset = DatasetDict({
            'train':train_dataset,
            'test':test_dataset
        })

        fine_tuner = LLMFineTuner(model_path=model_path,
                                  tokenizer_path=tokenizer_path,
                                  dataset=dataset,
                                  finetune_logger=finetune_logger)
        
        training_args = {
            'output_dir' : 'results/llm_results',
            'learning_rate': 2e-5,
            'warmup_steps' : 500,
            'per_device_train_batch_size' : 1,
            'per_device_eval_batch_size' :1,
            'num_train_epochs':3,
            'weight_decay':0.01,
            'evaluation_strategy':"epoch",
            'no_cuda':True
        }

        fine_tuner.define_training_args(**training_args)
        fine_tuner.define_trainer()
        model = fine_tuner.start_fine_tuning()

    except Exception as e:
        finetune_logger.error(e, exc_info=True)
        raise
    pass

if __name__ == '__main__':
    finetune_logger.info("Test Run")
    llm_ftmain()
    finetune_logger.info("Test Run Successfully")