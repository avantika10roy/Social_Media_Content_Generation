
import time
import torch
import pandas
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig, set_seed
from src.utils.set_seed import set_global_seed
from src.utils.logger import LoggerSetup
from src.prompts.prompts import llm_prompt
from src.model_inference.llm_inference import LLMInference

import gc

gc.collect()
torch.set_num_threads(1)

inference_logger = LoggerSetup(logger_name="llm_inference.py", log_filename_prefix="llm_inference").get_logger()
inference_logger.info("Logger Successfully Initialized")
set_global_seed(logger=inference_logger, seed=42)

# The topic should be decriptive enough so that the model does not hallucinate
prompt = llm_prompt(platform="Facebook", 
                    # topic="Holiday Greetings", 
                    company_name="Itobuz Private Limited",
                    extra_details="No year should be included in Hashtags",
                    occasion="Republic Day",
                    brief= "Well Wishes for the day",
                    target_audience="Followers")

infer = LLMInference(model_path='src/base_models/falcon1b/model',
                     tokenizer_path='src/base_models/falcon1b/tokenizer', 
                     lora_path='results/llm_results/pipeline_finetuning_v9')



start = time.time()
result = infer.generate(prompt=prompt)
end = time.time()
inference_logger.info(result)
inference_logger.info(f"\nTime Taken:{end-start}")