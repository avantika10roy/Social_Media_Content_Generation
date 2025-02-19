## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES
import os 
import json
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

from config.config import Config
from src.utils.logger import LoggerSetup
from src.utils.set_seed import set_global_seed
from src.prompts.prompts import llm_finetuning_prep

from peft import LoraConfig
from peft import get_peft_model
from peft import prepare_model_for_kbit_training  

from transformers import Trainer
from transformers import set_seed
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForSeq2SeqLM


# LOGGER SETUP
FLAN_T5_LOGGER = LoggerSetup(logger_name="FLAN_T5.py", log_filename_prefix="flan_t5").get_logger()

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
            finetune_logger: Logger instance for tracking operations.
        """
        try:
            self.device          = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            
            model_kwargs         = {
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
            }
            
            self.model           = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
            
            if self.device in ["mps", "cpu"]:
                self.model       = self.model.to(self.device)

            self.tokenizer       = AutoTokenizer.from_pretrained(tokenizer_path)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.dataset         = dataset
            self.lora_config     = None
            self.training_args   = None
            self.trainer         = None
            self.logger          = finetune_logger

        except Exception as LLMFineTunerInitializationError:
            if self.logger:
                self.logger.error(repr(LLMFineTunerInitializationError), exc_info = True)
            
            raise

    def set_model(self, model) -> None:
        """
        Set the model for fine-tuning.
        
        Arguments:
        -----------
            model: The model to be used for fine-tuning
        """
        try:
            self.model            = model
            
            if self.logger:
                self.logger.info("Model set successfully")
        
        except Exception as ModelSetupError:
            if self.logger:
                self.logger.error(repr(ModelSetupError), exc_info = True)
            
            raise

    def define_lora_config(self, **kwargs) -> None:
        """
        Defines the LoRA config.
        If not used, it will not cause any problems.
        """
        try:
            self.lora_config      = LoraConfig(**kwargs)
            self.model            = get_peft_model(self.model, self.lora_config)
            
            if self.logger:
                self.logger.info("LoRA config defined successfully")

        except Exception as LoraConfigDefinitionError:
            if self.logger:
                self.logger.error(repr(LoraConfigDefinitionError), exc_info=True)
            
            raise

    def define_training_args(self, **kwargs) -> None:
        """
        Defines the training arguments.
        """
        try:
            if 'device_map' in kwargs:
                del kwargs['device_map']
            
            self.training_args    = TrainingArguments(**kwargs)
            if self.logger:
                self.logger.info("Training arguments defined successfully")

        except Exception as TrainingArgumentDefinitionError:
            if self.logger:
                self.logger.error(repr(TrainingArgumentDefinitionError), exc_info=True)
            
            raise

    def define_trainer(self) -> None:
        """
        Defines the trainer used during fine-tuning.
        """
        try:
            if not all([self.model, self.tokenizer, self.training_args, self.dataset]):
                raise ValueError("Model, tokenizer, training arguments, and dataset must be set before defining trainer")

            train_dataset         = self.dataset["train"] if "train" in self.dataset else self.dataset
            eval_dataset          = self.dataset["test"] if "test" in self.dataset else None

            self.trainer          = Trainer(model=self.model,args=self.training_args,train_dataset=train_dataset,eval_dataset=eval_dataset,tokenizer=self.tokenizer)
            
            if self.logger:
                self.logger.info("Trainer defined successfully")
        
        except Exception as TrainerDefinitionError:
            if self.logger:
                self.logger.error(repr(TrainerDefinitionError), exc_info=True)
            
            raise

    def start_fine_tuning(self) -> AutoModelForSeq2SeqLM:
        """
        Starts the fine-tuning process.
        """
        try:
            if not self.trainer:
                raise ValueError("Trainer must be defined before starting fine-tuning")
                
            self.trainer.train()
            if self.logger:
                self.logger.info("Fine-tuning completed successfully")
            
            return self.model

        except Exception as FineTuningStartError:
            if self.logger:
                self.logger.error(repr(FineTuningStartError), exc_info=True)
            
            raise

def flan_t5_finetuner_main(logger: LoggerSetup) -> None:
    """
    Main function to run the fine tuner for FLAN-T5-Large.
    """
    
    set_global_seed(logger=logger, seed=42)
    
    try:
        model_path             = 'src/base_models/flan_t5_base/model'
        tokenizer_path         = 'src/base_models/flan_t5_base/tokenizer'

        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            FLAN_T5_LOGGER.info("Loading existing model and tokenizer...")
            model              = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer          = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            FLAN_T5_LOGGER.info("Downloading model and tokenizer from Hugging Face...")
            model              = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
            tokenizer          = AutoTokenizer.from_pretrained('google/flan-t5-base')

            os.makedirs(model_path, exist_ok=True)
            os.makedirs(tokenizer_path, exist_ok=True)
            
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(tokenizer_path)

        tokenizer.pad_token    = tokenizer.pad_token or tokenizer.eos_token

        with open(Config.MIXED_CURATED_DATA_PATH, "r", encoding = "utf-8") as file:
            data               = json.load(file)

        for item in data:
            item.pop('image_paths', None)

        data_list              = [llm_finetuning_prep(item) for item in data]
        dataset                = Dataset.from_dict({'texts': data_list})

        def tokenize_function(examples):
            inputs             = tokenizer(examples["texts"], 
                                           padding        = "max_length", 
                                           truncation     = True, 
                                           max_length     = 128, 
                                           return_tensors = 'pt')
            
            inputs["labels"]   = inputs["input_ids"].clone().detach()
            
            return inputs
        
        tokenized_dataset      = dataset.map(tokenize_function, batched = True)
        split_dataset          = tokenized_dataset.train_test_split(test_size = 0.2)

        output_dir = 'results/llm_results/flan_t5_base_fine_tuning_results_v1'
        
        os.makedirs(output_dir, exist_ok = True)
        
        training_args          = {'output_dir'                   : output_dir,
                                  'learning_rate'                : 5e-5,
                                  'warmup_steps'                 : 100,
                                  'logging_first_step'           : True,
                                  'logging_steps'                : 5,
                                  'per_device_train_batch_size'  : 2,
                                  'per_device_eval_batch_size'   : 2,
                                  'gradient_accumulation_steps'  : 8,
                                  'num_train_epochs'             : 20,
                                  'logging_dir'                  : 'logs/llm_finetune_logs/flan_t5_v1',
                                  'weight_decay'                 : 0.01,
                                  'bf16'                         : torch.cuda.is_bf16_supported(),
                                  'fp16'                         : torch.cuda.is_available(),
                                  'evaluation_strategy'          : "epoch",
                                  'save_strategy'                : "epoch",
                                  'load_best_model_at_end'       : True,
                                  }

        lora_config            = LoraConfig(r              = 8,
                                            lora_alpha     = 32,
                                            target_modules = ["q", "v"],  
                                            lora_dropout   = 0.05,
                                            bias           = "none",
                                            task_type      = "SEQ_2_SEQ_LM"
                                            )

        fine_tuner             = FLAN_T5_FineTuner(model_path       = model_path,
                                                   tokenizer_path   = tokenizer_path,
                                                   dataset          = split_dataset,
                                                   finetune_logger  = FLAN_T5_LOGGER
                                                   )

        model                  = prepare_model_for_kbit_training(model)
        model                  = get_peft_model(model, lora_config)
        
        fine_tuner.set_model(model)
        fine_tuner.define_training_args(**training_args)
        fine_tuner.define_trainer()
        fine_tuner.start_fine_tuning()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        lora_dict              = lora_config.to_dict()

        for key, value in lora_dict.items():
            if isinstance(value, set):
                lora_dict[key] = list(value)
        
        config_path = os.path.join(output_dir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(lora_dict, f, indent=4)

        FLAN_T5_LOGGER.info(f"Model, tokenizer, and config saved to {output_dir}")

    except Exception as MainError:
        FLAN_T5_LOGGER.error(f"Fine-tuning failed: {repr(MainError)}", exc_info=True)
        raise

if __name__ == '__main__':
    FLAN_T5_LOGGER.info("Starting Fine-Tuning Process...")
    flan_t5_finetuner_main(logger = FLAN_T5_LOGGER)
    FLAN_T5_LOGGER.info("Fine-Tuning Completed Successfully!")