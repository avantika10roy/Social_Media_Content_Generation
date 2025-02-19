## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

# from config.config import Config
# from src.model_finetuners.t5_finetuner import T5LoRATrainer
# from src.model_finetuners.t5_lora_finetuning import LLMFineTuningLoRA
from src.model_inference.t5_inference import T5LoRAInference


def main():
    
    # trainer                 = T5LoRATrainer(model_name = "t5-base",local_model_path = "base_models")
    
    # posts                   = trainer.load_data(Config.MIXED_CURATED_DATA_PATH)
    # dataset                 = trainer.prepare_dataset(posts)
    
    # # Configure LoRA
    # lora_config             = trainer.configure_lora(r             = 16,
    #                                                  lora_alpha    = 32,
    #                                                  lora_dropout  = 0.05
    #                                                  )
    
    # trainer.train(dataset                        = dataset,
    #               output_dir                     = "results/llm_results/t5_llm_checkpoints",
    #               num_train_epochs               = 3,
    #               per_device_train_batch_size    = 2,
    #               learning_rate                  = 5e-5,
    #               lora_config                    = lora_config
    #               )
    
    
    
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
    # fine_tuner.train(num_epochs = 25)

    # print("\nSaving model and training history...")
    # fine_tuner.save_model(SAVE_PATH)
    
    
    inferencer        = T5LoRAInference("models/t5_base_lora")
    
    generated_post    = inferencer.generate_post(platform         = "LinkedIn",
                                                 topic            = "Artificial Intelligence",
                                                 brief            = "Recent advances in generative AI and its impact on business",
                                                 extras           = "Include mention of cost efficiency",
                                                 word_limit       = 150,
                                                 target_audience  = "professionals",
                                                 tone             = "informative"
                                                 )
    
    print("Generated Post:")
    print(generated_post)

if __name__ == '__main__':
    main()