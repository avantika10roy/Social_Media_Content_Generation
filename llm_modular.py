# ------Doane by Manu Bhaskar --------

# ------ Dependencies --------
import os
import time
from tqdm import tqdm
from config.config import Config
from huggingface_hub import login
from src.utils.logger import LoggerSetup
from src.utils.set_seed import set_global_seed
from torch.utils.data import DataLoader
from src.prompts.prompts import llm_finetuning_prep
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model_finetuners.llm_fine_tuner import LLMTrainer
from src.custom_dataset.llm_dataset import CustomDatasetForLLMFineTuning


# --------- FineTuning ---------
class LLMFineTuning:
    def __init__(self, model_path:str, tokenizer_path:str, device:str, logger:LoggerSetup, epochs:int, results_dir:str, model_name:str=None):
        try:
            if (model_path is None) and (model_name is None):
                raise
            self.model_path     = model_path
            self.tokenizer_path = tokenizer_path
            self.model_name     = model_name
            self.device         = device
            self.logger         = logger
            self.epochs         = epochs
            self.results_dir    = results_dir

            if (not os.path.isdir(model_path)) or (not os.path.isdir(tokenizer_path)):
                self.model                  = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="cpu")
                self.tokenizer              = AutoTokenizer.from_pretrained(self.model_name)

                self.tokenizer.pad_token    = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"

                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(tokenizer_path)
            else:
                self.model                  = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cpu")
                self.tokenizer              = AutoTokenizer.from_pretrained(self.tokenizer_path)

                self.tokenizer.pad_token    = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"

        except Exception as InitializationError:
            self.logger = logger
            self.logger.error(repr(InitializationError), exc_info=True)
            return repr(InitializationError)
    
    def setup(self, batch_size:int, data_file_path:str, lr:int=5e-5, setup_lora:bool=True, gradient_accumulation_steps:int=1):
        try:
            self.accumulation_steps        = gradient_accumulation_steps
            self.dataset                   = CustomDatasetForLLMFineTuning(json_file_path = data_file_path,
                                                                           tokenizer      = self.tokenizer,
                                                                           transform      = llm_finetuning_prep,
                                                                           max_length     = 256)

            self.train_loader              = DataLoader(dataset    = self.dataset, 
                                                        batch_size = batch_size, 
                                                        shuffle    = False)

            self.tuner                     = LLMTrainer(model                       = self.model,
                                                        tokenizer                   = self.tokenizer,
                                                        device                      = self.device, 
                                                        logger                      = self.logger,
                                                        lr                          = lr,
                                                        gradient_accumulation_steps = self.accumulation_steps)
            
            if setup_lora:
                self.tuner.lora_model_setup(r              = 8,
                                            lora_alpha     = 16,
                                            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                            bias           = "none",
                                            lora_dropout   = 0.1,
                                            task_type      = "CAUSAL_LM")
        
        except Exception as SetupError:
            self.logger.error(repr(SetupError), exc_info=True)
            return repr(SetupError)

    def train(self):
        step_counter   = 0
        self.tuner.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            self.tuner.optimizer.zero_grad()

            start_time = time.time()
            for batch in tqdm(self.train_loader, desc="Training Progress"):
                loss = self.tuner.train_step(batch=batch)
                total_loss+=loss
                step_counter += 1

                if step_counter % self.accumulation_steps == 0:
                    self.tuner.optimizer.step()
                    self.tuner.optimizer.zero_grad()
            end_time = time.time()

            avg_loss = total_loss/len(self.train_loader)
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Total Loss : {total_loss}, Average Loss: {avg_loss:.4f}, Time Taken: {end_time-start_time}")
            
            self.tuner.save_model(self.results_dir)

        


# --------- Main ------------
if __name__ == "__main__":

    # login to huggingface ID - Manu Bhaskar
    access_token = "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"
    login(access_token)

    # Fine Tuning of Model
    finetune_logger = LoggerSetup(logger_name="llm_fine_tuner_pipeline.py", log_filename_prefix="llm_fine_tuner_pipeline").get_logger()
    finetune_logger.info("Logger Successfully Initialized")
    set_global_seed(logger=finetune_logger, seed=42)
    fine_tuning = LLMFineTuning(model_path='src/base_models/falcon1b/model',
                                tokenizer_path='src/base_models/falcon1b/tokenizer',
                                device="cpu",
                                logger=finetune_logger,
                                epochs=2,
                                results_dir="results/llm_results/pipeline_finetuning_v10/"
                                )
    fine_tuning.setup(batch_size=2,
                      data_file_path=Config.MIXED_CURATED_DATA_PATH)
    fine_tuning.train()
    