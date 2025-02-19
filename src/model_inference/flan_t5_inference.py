## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

import os
import torch
from peft import PeftModel
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration 

class FLAN_T5_Inference:
    
    def __init__(self) -> None:
        
        try:
            self.device      = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"Using device: {self.device}")
            
            self.model       = None
            self.tokenizer   = None
            self.device      = self.device
        
        except Exception as e:
            print(f"Error initializing FLAN-T5 inference: {str(e)}")
            
            raise

    @staticmethod   
    def load_model(base_model_id: str = None, local_model_path: str = None, local_tokenizer_path: str = None, lora_path: str = None):
        """
        Loads the model, tokenizer, and optionally LoRA weights from either a 
        local path or from HuggingFace. 

        Arguments:
        -----------
            base_model_id (str)                   : The identifier of the base model to load from HuggingFace. Default is "google/flan-t5-base".
            local_model_path (str, optional)      : Path to a locally saved model. If None, the model is loaded from HuggingFace.
            local_tokenizer_path (str, optional)  : Path to a locally saved tokenizer. If None, the tokenizer is loaded from HuggingFace.
            lora_path (str, optional)             : Path to LoRA weights for fine-tuning. If None, no LoRA weights are loaded.

        Returns:
        ---------
            model (transformers.T5ForConditionalGeneration)  : The loaded model (base model or LoRA-tuned model).
            tokenizer (transformers.T5Tokenizer)             : The tokenizer corresponding to the model.
            device (str)                                     : The device used for model inference ('cpu', 'cuda', or 'mps').

        Raises:
        --------
            Exception                                         : If there is an error loading the model, tokenizer, or LoRA weights.
        """
        
        device               = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
 
        model_kwargs         = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}

        if local_model_path and os.path.exists(local_model_path):
            print(f"Loading model from local path: {local_model_path}")
            
            base_model       = T5ForConditionalGeneration.from_pretrained(local_model_path, local_files_only=True, **model_kwargs)
        
        else:
            print(f"Loading model from HuggingFace: {base_model_id}")
            
            base_model       = T5ForConditionalGeneration.from_pretrained(base_model_id, **model_kwargs)

        if local_tokenizer_path and os.path.exists(local_tokenizer_path):
            print(f"Loading tokenizer from local path: {local_tokenizer_path}")
            
            tokenizer        = T5Tokenizer.from_pretrained(local_tokenizer_path, local_files_only=True)
       
        else:
            print(f"Loading tokenizer from HuggingFace: {base_model_id}")
            tokenizer        = T5Tokenizer.from_pretrained(base_model_id)

        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA weights from: {lora_path}")
            
            model            = PeftModel.from_pretrained(base_model, lora_path)
        
        else:
            print("No LoRA weights provided, using base model")
            
            model            = base_model

        model                = model.to(device)
        
        return model, tokenizer, device

    @staticmethod
    def set_prompt(platform: str, tone: str, target_audience: str, word_limit: int, topic: str) -> str:
        """
        Constructs a prompt with chain-of-thought reasoning and few-shot prompting for better structured output.
        
        """
        
        prompt = f"""
                
                Let's think about this step by step:

                1. First, analyze the example post structure:
                    - Starts with an attention-grabbing headline
                    - Follows with engaging content that connects with readers
                    - Ends with relevant hashtags and emojis

                2. Consider key elements for {platform} post:
                    - Platform   : {platform} requires {tone} tone
                    - Audience   : {target_audience} expects relatable content
                    - Topic      : {topic} should be central focus
                    - Length     : Stay within {word_limit} words

                3. Study this example format carefully:

                    This Republic Day, ITobuz Technologies salutes the innovative spirit of India.
                    Just as our nation builds its strength on unity and diversity, we craft solutions that empower businesses across industries. Wishing everyone a Happy Republic Day 2025!
                    #republicday2025 #26thJanuary2025 #republicindia #itobuztechnologies #freedom
                    🇮🇳 ✨

                4. Now, generate a new post following exactly:
                    - First line: Attention-grabbing headline
                    - Second line: Main content with engaging message
                    - Third line: Exactly 5 relevant hashtags
                    - Fourth line: 2-3 relevant emojis

                Do not repeat the example. Create an entirely new post now!
                Post:
        """
        
        return prompt

if __name__ == '__main__':
    BASE_MODEL_PATH           = os.path.abspath("./models/flan-t5-base")
    TOKENIZER_PATH            = os.path.abspath("./models/flan-t5-base")
    LORA_PATH                 = os.path.abspath("./results/llm_results/flan_t5_base_fine_tuning_results_v1/checkpoint-220")
    
    llm_model = FLAN_T5_Inference()
    
    model, tokenizer, device  = llm_model.load_model(base_model_id         = "google/flan-t5-base",
                                                     local_model_path      = BASE_MODEL_PATH if os.path.exists(BASE_MODEL_PATH) else None,
                                                     local_tokenizer_path  = TOKENIZER_PATH if os.path.exists(TOKENIZER_PATH) else None,
                                                     lora_path             = LORA_PATH if os.path.exists(LORA_PATH) else None)
    
    prompt                    = llm_model.set_prompt(platform         = "Facebook",
                                                     tone             = "Engaging",
                                                     target_audience  = "General Public",
                                                     word_limit       = 250,
                                                     topic            = "Republic Day")
    
    input_dict                = tokenizer(prompt,
                                          return_tensors      = "pt",
                                          padding             = True,
                                          truncation          = True,
                                          max_length          = 512,
                                          add_special_tokens  = True)
    

    input_dict                 = {k: v.to(device) for k, v in input_dict.items()}
    

    generation_config          = {"max_length"            : 200,          
                                  "min_length"            : 50,          
                                  "num_beams"             : 5,             
                                  "no_repeat_ngram_size"  : 3,   
                                  "num_return_sequences"  : 1,
                                  "temperature"           : 0.7,         
                                  "top_p"                 : 0.95,              
                                  "top_k"                 : 50,                
                                  "repetition_penalty"    : 1.2,  
                                  "length_penalty"        : 1.0,      
                                  "early_stopping"        : True,      
                                  "do_sample"             : True
                                  }
    
    with torch.no_grad():
        
        outputs                = model.generate(input_ids       = input_dict["input_ids"],
                                                attention_mask  = input_dict["attention_mask"],
                                                **generation_config
                                                )
    
    generated_text             = tokenizer.decode(outputs[0],
                                                  skip_special_tokens           = True,
                                                  clean_up_tokenization_spaces  = True
                                                  )
    
    generated_text             = generated_text.strip()
    
    print("\nGenerated Text:")
    print(generated_text)