## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

from config.config import Config
from src.model_inference.t5 import SocialMediaPostGenerator
from src.model_inference.t5_inference import T5LoRAInference
from src.model_inference.codet5_inference import CodeT5LoRAInference
# from src.model_finetuners.codet5_lora_finetuner import LLMFineTuningLoRA
from src.model_finetuners.llama_3_3b_lora_finetuning import LLMFineTuningLoRA


def main():

    # ## ----- FINE-TUNING - t5_lora_finetuning.py -----

    # MODEL_NAME     = "google-t5/t5-base" 
    # TOKENIZER_NAME = MODEL_NAME
    # DATA_PATH      = "data/mixed_curated/mixed_curated.json"
    # SAVE_PATH      = "models/t5_base_lora/"
    # BATCH_SIZE     = 1

    # print("\nInitializing fine-tuning process...")
    # fine_tuner     = LLMFineTuningLoRA(model_name     = MODEL_NAME, 
    #                                    tokenizer_name = TOKENIZER_NAME, 
    #                                    data_path      = DATA_PATH,
    #                                    batch_size     = BATCH_SIZE,
    #                                   )

    # print("\nStarting training...")
    # fine_tuner.train(num_epochs = 2)

    # print("\nSaving model and training history...")
    # fine_tuner.save_model(SAVE_PATH)
    
    ## ----- INFERENCE - t5_inference.py -----

    # inferencer        = T5LoRAInference("models/t5_base_lora")
    
    # generated_post    = inferencer.generate_post(platform         = "LinkedIn",
    #                                              topic            = "Artificial Intelligence",
    #                                              brief            = "Recent advances in generative AI and its impact on business",
    #                                              extras           = "Include mention of cost efficiency",
    #                                              word_limit       = 150,
    #                                              target_audience  = "professionals",
    #                                              tone             = "informative"
    #                                              )
    
    # print("Generated Post:")
    # print(generated_post)
    
    ## ----- MODEL DOWNLOADED FROM HUGGING FACE AND INFERENCE DIRECTLY - t5.py -----

    # generator = SocialMediaPostGenerator(model_id = "google-t5/t5-base", model_dir = "models/t5_base")
    # generator.download_model()
    
    # prompt    = generator.generate_prompt(occasion          = "Republic Day",
    #                                       brief             = "Celebrate the spirit of patriotism",
    #                                       platform          = "LinkedIn",
    #                                       target_audience   = "professionals",
    #                                       tone              = "inspirational",
    #                                       extra_details     = ""
    #                                       )
    
    # post     = generator.generate_post(prompt)
    # print("Generated Post:", post)
    
    
    ## ----- FINE-TUNING OF LLAMA-3.2 3B INSTRUCT - llama_3.2_3b_lora_finetuning.py -----
        
    MODEL_NAME     = "meta-llama/Llama-3.2-3B-Instruct"           # ENTER MODEL ID FROM HUGGING FACE
    TOKENIZER_NAME = MODEL_NAME
    DATA_PATH      = "data/mixed_curated/mixed_curated.json"      # DATASET PATH
    SAVE_PATH      = "models/llama_3_2_3b_instruct/"              # MODEL SAVE PATH
    BATCH_SIZE     = 1

    print("\nInitializing fine-tuning process...")
    fine_tuner     = LLMFineTuningLoRA(model_name     = MODEL_NAME, 
                                       tokenizer_name = TOKENIZER_NAME, 
                                       data_path      = DATA_PATH,
                                       batch_size     = BATCH_SIZE,
                                      )

    print("\nStarting training...")
    fine_tuner.train(num_epochs = 2)

    print("\nSaving model and training history...")
    fine_tuner.save_model(SAVE_PATH)
    
    ## ----- CODE-T5 BASE MODEL FINE-TUNING - codet5_lora_finetuner.py -----
    
    # MODEL_NAME     = "Salesforce/codet5-base" 
    # TOKENIZER_NAME = MODEL_NAME
    # DATA_PATH      = "data/mixed_curated/mixed_curated.json"
    # SAVE_PATH      = "models/codet5_base/"
    # BATCH_SIZE     = 1

    # print("\nInitializing fine-tuning process...")
    # fine_tuner     = LLMFineTuningLoRA(model_name     = MODEL_NAME, 
    #                                    tokenizer_name = TOKENIZER_NAME, 
    #                                    data_path      = DATA_PATH,
    #                                    batch_size     = BATCH_SIZE,
    #                                   )

    # print("\nStarting training...")
    # fine_tuner.train(num_epochs = 10)

    # print("\nSaving model and training history...")
    # fine_tuner.save_model(SAVE_PATH)
    
    
    ## ----- INFERENCE - codet5_inference.py -----

    # inferencer        = CodeT5LoRAInference("models/codet5_base")
    
    # generated_post    = inferencer.generate_post(occasion          = "Republic Day",
    #                                              brief             = "Celebrate the spirit of patriotism",
    #                                              platform          = "LinkedIn",
    #                                              target_audience   = "professionals",
    #                                              tone              = "inspirational",
    #                                              extra_details     = "Give 5 hastags"
    #                                              )
    
    # print("Generated Post:")
    # print(generated_post)

if __name__ == '__main__':
    main()