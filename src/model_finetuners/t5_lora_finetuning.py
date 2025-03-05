# Dependencies
import os
import json
import time
import torch
from tqdm.auto import tqdm
from peft import PeftModel
from peft import LoraConfig
from torch.optim import AdamW
from datetime import datetime
from peft import get_peft_model
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from peft import prepare_model_for_kbit_training

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure directory exists
def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.labels         = labels


    def __len__(self):
        return len(self.input_ids)


    def __getitem__(self, idx):
        return {"input_ids"      : self.input_ids[idx],
                "attention_mask" : self.attention_mask[idx],
                "labels"         : self.labels[idx]  
               }



# LLM Trainer Class
class LLMTrainer:
    def __init__(self, tokenizer_path: str, data_path: str, batch_size: int = 4):

        self.tokenizer              = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = tokenizer_path)

        # Fix Padding
        self.tokenizer.pad_token    = self.tokenizer.eos_token  

        # Ensure Left Padding for Falcon
        self.tokenizer.padding_side = "left"  

        self.data_path              = data_path
        self.batch_size             = batch_size


    def load_data(self):
        """ 
        Load and preprocess JSON data dynamically 
        """
        with open(self.data_path, 'r') as file:
            data = json.load(file)

        formatted_samples           = [f"Platform: {item['platform']} \n Heading: {item['post_heading']} \n Post Content: {item['post_contents']} \n "
                                       f"Hashtags: {' '.join(item['hashtags'])} \n Emojis: {' '.join(item['emoji'])} \n\n"
                                       f"### Target Response:\n{item['raw_post_content']}" for item in data]

        # Tokenize the concatenated input-output pair
        tokenized                   = self.tokenizer(formatted_samples, 
                                                     padding        = True, 
                                                     truncation     = True, 
                                                     return_tensors = "pt", 
                                                     max_length     = 512)

        # Ensure consistent dtypes
        tokenized['input_ids']      = tokenized['input_ids'].to(torch.long)
        tokenized['attention_mask'] = tokenized['attention_mask'].to(torch.float32)

        return tokenized


    def get_dataloader(self):
        """ 
        Prepare the DataLoader for fine-tuning 
        """
        tokenized_data = self.load_data()

        dataset        = CustomDataset(tokenized_data['input_ids'],
                                       tokenized_data['attention_mask'],
                                       tokenized_data['input_ids'].clone(),  # Clone input_ids as labels
                                      )

        dataloader     = DataLoader(dataset    = dataset, 
                                    batch_size = self.batch_size, 
                                    shuffle    = True)

        return dataloader


# LLMFineTuning Class Using LoRA
class LLMFineTuningLoRA:
    def __init__(self, model_name:str, tokenizer_name:str, data_path:str, learning_rate:float = 5e-5, batch_size:int=1):
        
        self.tokenizer              = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = tokenizer_name)

        # Use the EOS token as padding
        self.tokenizer.pad_token    = self.tokenizer.eos_token  

        # Explicitly set the padding size
        self.tokenizer.padding_side = "left"

        # Initialize the trainer
        self.trainer                = LLMTrainer(tokenizer_path = tokenizer_name, 
                                                 data_path      = data_path,
                                                 batch_size     = batch_size)
        
        # Load Falcon-7B model
        print("\nLoading model...")
        self.model                  = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path = model_name, 
                                                                            torch_dtype                   = torch.float16, 
                                                                            device_map                    = "auto"
                                                                            )

        # Convert the model to optimized format
        print("\n\nPreparing model for training...")
        self.model                  = prepare_model_for_kbit_training(self.model)

        # LoRA Configuration
        target_finetuning_modules   = ["q", "k", "v", "o"]

        lora_config                 = LoraConfig(r              = 8,                         # Rank (controls number of trainable parameters)
                                                 lora_alpha     = 8,                         # Scaling factor
                                                 lora_dropout   = 0.05,                      # Regularization
                                                 target_modules = target_finetuning_modules, # Fine-tuning specific attention layers
                                                 bias           = "none",
                                                 task_type      = "SEQ_2_SEQ_LM",
                                                )

        # Apply LoRA to Falcon-7B
        self.model                  = get_peft_model(model       = self.model, 
                                                     peft_config = lora_config) 

        # Move model to appropriate dtype
        self.model                  = self.model.to(torch.float16)

        # Optimizer
        self.optimizer              = AdamW(params = self.model.parameters(), 
                                            lr     = learning_rate,
                                           )

        # Initialize lists to store metrics
        self.training_history      = {'epoch_losses' : list(),
                                      'batch_losses' : list(),
                                      'epoch_times'  : list(),
                                     }


    def train(self, num_epochs: int):

        # Safe Fallback to MPS or CPU
        device     = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

         # Ensure device and model dtype matches
        if (device == "mps"):
            self.model = self.model.to(torch.float32)
        
        elif (device == 'cuda'):
            self.model = self.model.to(torch.float16)
        
        else:
            self.model = self.model

        # Get the dataloader
        dataloader         = self.trainer.get_dataloader()

        self.model.to(device)
        self.model.train()

        # Create epoch progress bar
        epoch_progress_bar = tqdm(range(num_epochs), 
                                  desc     = "Epochs", 
                                  position = 0)

        for epoch in epoch_progress_bar:
            epoch_start_time   = time.time()
            total_loss         = 0
            batch_losses       = list()

            # Create batch progress bar
            batch_progress_bar = tqdm(dataloader, 
                                      desc     = f"Epoch {epoch+1}", 
                                      position = 1, 
                                      leave    = False)

            for batch_idx, batch in enumerate(batch_progress_bar):
                # Ensure consistent dtypes
                input_ids      = batch["input_ids"].to(device).long()
                attention_mask = batch["attention_mask"].to(device).float()
                labels         = batch["labels"].to(device).long() 

                # Clear gradients
                self.optimizer.zero_grad()
                
                # Forward Pass
                outputs        = self.model(input_ids      = input_ids, 
                                            attention_mask = attention_mask, 
                                            labels         = labels)

                # Calculate the loss for this epoch
                loss           = outputs.loss

                # For device compatibility, convert loss to specific format
                if (device == "mps"):
                    loss = loss.to(torch.float32)
                
                elif (device == "cuda"):
                    loss = loss.to(torch.float16) 
                
                else:
                    loss = loss
                
                # Backward Pass
                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()
                total_loss  += current_loss
                batch_losses.append(current_loss)

                # Update batch progress bar
                batch_progress_bar.set_postfix({'loss'     : f'{current_loss:.4f}',
                                                'avg_loss' : f'{total_loss/(batch_idx+1):.4f}'
                                              })

            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start_time
            avg_loss   = total_loss / len(dataloader)
            
            # Store metrics
            self.training_history['epoch_losses'].append(avg_loss)
            self.training_history['batch_losses'].extend(batch_losses)
            self.training_history['epoch_times'].append(epoch_time)
            
            # Update epoch progress bar
            epoch_progress_bar.set_postfix({'avg_loss' : f'{avg_loss:.4f}',
                                            'time'     : f'{epoch_time:.1f}s'
                                          })

            # Print detailed epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Min Batch Loss: {min(batch_losses):.4f}")
            print(f"  Max Batch Loss: {max(batch_losses):.4f}")
            print(f"  Time Taken: {epoch_time:.1f} seconds")
            print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 150)
            print("\n\n")
                

    def save_model(self, save_path: str):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save the model
        self.model.save_pretrained(save_path)
        
        # Save tokenizer along with the adapter
        self.tokenizer.save_pretrained(save_path) 
        
        # Save training history
        history_path = os.path.join(save_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)
        
        print(f"LoRA fine-tuned model saved at: {save_path}")
        print(f"Training history saved at: {history_path}")



# Main Script Execution
if __name__ == "__main__":

    MODEL_NAME     = "tiiuae/Falcon3-1B-Base" 
    TOKENIZER_NAME = MODEL_NAME
    DATA_PATH      = "data/mixed_curated/mixed_curated.json"
    SAVE_PATH      = "models/falcon_1b_base_lora/"
    BATCH_SIZE     = 1

    print("\nInitializing fine-tuning process...")
    fine_tuner     = LLMFineTuningLoRA(model_name     = MODEL_NAME, 
                                       tokenizer_name = TOKENIZER_NAME, 
                                       data_path      = DATA_PATH,
                                       batch_size     = BATCH_SIZE,
                                      )

    print("\nStarting training...")
    fine_tuner.train(num_epochs = 25)

    print("\nSaving model and training history...")
    fine_tuner.save_model(SAVE_PATH)