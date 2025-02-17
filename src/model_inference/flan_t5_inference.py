## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


class FLAN_T5_Inference:
    """
    A class that handles inference using a fine-tuned FLAN-T5 model
    with LoRA adaptors.
    """
    def __init__(self, 
                 base_model_id        : str = "google/flan-t5-base",
                 local_model_path     : str = None,
                 local_tokenizer_path : str = None,
                 lora_path            : str = None):
        """
        Initialize the inference class with model paths.

        Arguments:
        ----------
            base_model_id: HuggingFace model ID or path to local base model
            local_model_path: Path to locally saved model (optional)
            local_tokenizer_path: Path to locally saved tokenizer (optional)
            lora_path: Path to the saved LoRA weights (optional)
        """
        try:

            self.device            = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            model_kwargs           = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
            }
            
            if local_model_path and os.path.exists(local_model_path):
                print(f"Loading model from local path: {local_model_path}")
                
                self.base_model    = AutoModelForSeq2SeqLM.from_pretrained(local_model_path,
                                                                           local_files_only = True,
                                                                           **model_kwargs)
            else:
                print(f"Loading model from HuggingFace: {base_model_id}")
                
                self.base_model    = AutoModelForSeq2SeqLM.from_pretrained(base_model_id,
                                                                           **model_kwargs
                                                                           )

            if local_tokenizer_path and os.path.exists(local_tokenizer_path):
                print(f"Loading tokenizer from local path: {local_tokenizer_path}")
                
                self.tokenizer     = AutoTokenizer.from_pretrained(local_tokenizer_path,
                                                                   local_files_only = True
                                                                   )
            else:
                print(f"Loading tokenizer from HuggingFace: {base_model_id}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

            if lora_path and os.path.exists(lora_path):
                print(f"Loading LoRA weights from: {lora_path}")
                
                self.model     = PeftModel.from_pretrained(self.base_model,
                                                       lora_path
                                                       )
            else:
                print("No LoRA weights provided, using base model")
                self.model     = self.base_model

            if self.device in ["mps", "cpu"]:
                self.model     = self.model.to(self.device)

        except Exception as e:
            print(f"Error initializing FLAN-T5 inference: {str(e)}")
            raise

    def generate(self, prompt: str, max_new_tokens: int = 500, **kwargs) -> str:
        """
        Generate text based on the input prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters for model.generate()

        Returns:
            str: Generated text
        """
        try:
            
            inputs              = self.tokenizer(prompt, 
                                                 return_tensors  = 'pt',
                                                 padding         = True,
                                                 truncation      = True)
            
            inputs              = {k: v.to(self.device) for k, v in inputs.items()}

            outputs             = self.model.generate(**inputs,
                                                      max_new_tokens = max_new_tokens,
                                                      **kwargs)

            generated_text      = self.tokenizer.decode(outputs[0],
                                                        skip_special_tokens          = True,
                                                        clean_up_tokenization_spaces = True
                                                        )

            return generated_text

        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            
            raise

if __name__ == '__main__':
    
    BASE_MODEL_PATH  = os.path.abspath("./models/flan-t5-base")
    TOKENIZER_PATH   = os.path.abspath("./models/flan-t5-base")
    LORA_PATH        = os.path.abspath("./results/llm_results/flan_t5_base_fine_tuning_results_v1/checkpoint-11")

    # Example usage
    prompt           = """Generate a high-quality, engaging social media post for a business in a descriptive format. Follow the example structure and ensure clarity, creativity, and audience engagement.

    Context:
    - Platform: facebook 
    - Topic: Republic Day
    - Language: English
    - Word Limit: 250

    Requirements:
    - Craft a compelling opening that grabs attention.  
    - Highlight key details about the business or occasion.  
    - Maintain a consistent and engaging tone throughout.  
    - Tone should be determined by the platform.
    - Use persuasive language and storytelling where applicable.  
    - Include a strong call to action (CTA) to encourage engagement.  

    Now, generate a social media post using the provided context."""

    try:

        inference    = FLAN_T5_Inference(base_model_id        = "google/flan-t5-base",
                                         local_model_path     = BASE_MODEL_PATH if os.path.exists(BASE_MODEL_PATH) else None,
                                         local_tokenizer_path = TOKENIZER_PATH if os.path.exists(TOKENIZER_PATH) else None,
                                         lora_path            = LORA_PATH if os.path.exists(LORA_PATH) else None
                                         )

        result       = inference.generate(prompt,
                                          max_new_tokens = 500,
                                          temperature    = 0.5,
                                          top_p          = 0.85,
                                          do_sample      = True)
        print("\nGenerated Text:")
        print(result)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")