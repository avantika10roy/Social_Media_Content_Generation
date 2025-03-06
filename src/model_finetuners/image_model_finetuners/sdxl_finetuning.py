## ----- DONE BY AVANTIKA ROY -----

# Dependencies
import os
import json
import time
import torch
from PIL import Image
from tqdm.auto import tqdm
from peft import PeftModel
from peft import LoraConfig
from torch.optim import AdamW
from datetime import datetime
from peft import get_peft_model
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import CLIPTextModel
from transformers import CLIPTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from diffusers import UNet2DConditionModel
from diffusers import StableDiffusionXLPipeline 
from peft import prepare_model_for_kbit_training


# Clear MPS Cache
torch.mps.empty_cache()

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Custom Dataset Class
class ImageCaptionDataset(Dataset):
    """
    A custom dataset class to handle image-caption pairs for fine-tuning SDXL-Turbo
    """

    def __init__(self, json_path: str):
        """
        Initializes the dataset by loading the JSON file and defining image transformations

        Arguments:
        ----------
            json_path { str } : Path to the dataset JSON file
        """
        with open(json_path, "r", encoding="utf-8") as fp:
            self.data = json.load(fp)

        self.transform = transforms.Compose([transforms.Resize((512, 512)),
                                             transforms.ToTensor(),
                                           ])


    def __len__(self):
        """
        Returns the total number of items in the dataset
        """
        return len(self.data)


    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves an image-caption pair

        Arguments:
        ----------
            idx { int } : Index of the data item

        Returns:
        --------
            { tuple }   : (Processed image tensor, caption string)
        """
        item             = self.data[idx]

        # Load the image from the provided path
        image_path       = item["image_path"]
        image            = Image.open(image_path).convert("RGB")
        image            = self.transform(image)

        # Construct the text caption
        text_description = f"{item['post_heading']}. {item['post_content']}."
        context_info     = item["context"]

        # Additional metadata (logo presence, color theme, layout)
        color_theme      = f"Color Theme: {item['color_theme']}" if "color_theme" in item else ""
        layout_info      = f"Image Layout: {item['image_layout']}" if "image_layout" in item else ""

        # Final caption with metadata
        caption          = f"{text_description}. {context_info}. {color_theme}. {layout_info}"

        return image, caption


# Trainer Class for SDXL
class SDXLTrainer:
    """
    Handles tokenization and data loading for fine-tuning SDXL
    """

    def __init__(self, tokenizer_name: str, data_path: str, train_batch_size: int,val_batch_size: int, val_split: float = 0.1):
        """
        Initializes the tokenizer and splits the dataset into training and validation

        Arguments:
        ----------
            tokenizer_name   { str }  : Pretrained tokenizer model name

            data_path        { str }  : Path to the dataset JSON file

            train_batch_size { int }  : Number of samples per batch for training

            val_batch_size   { int }  : Number of samples per batch for validation

            val_split       { float } : Fraction of data to use for validation (default: 0.2)
        """
        print("\nLoading tokenizer & dataset...")

        # Load the CLIP Tokenizer
        self.tokenizer                       = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path = tokenizer_name)

        # Read the dataset from local JSON file
        full_dataset                         = ImageCaptionDataset(json_path = data_path)

        # Compute sizes for train/validation split
        total_size                           = len(full_dataset)
        val_size                             = int(total_size * val_split)
        train_size                           = total_size - val_size

        # Split datasets
        self.train_dataset, self.val_dataset = random_split(dataset = full_dataset, 
                                                            lengths = [train_size, val_size])

        # Set Training and Validation Batch Sizes
        self.train_batch_size                = train_batch_size
        self.val_batch_size                  = val_batch_size

        # Create dataloaders
        self.train_dataloader                = DataLoader(dataset    = self.train_dataset, 
                                                          batch_size = self.train_batch_size, 
                                                          shuffle    = True)

        self.val_dataloader                  = DataLoader(dataset    = self.val_dataset, 
                                                          batch_size = self.val_batch_size, 
                                                          shuffle    = False)

        print(f"Dataset split: {train_size} training samples, {val_size} validation samples\n")


    def get_dataloaders(self):
        """
        Returns the train and validation dataloaders
        """
        return self.train_dataloader, self.val_dataloader



# Fine-Tuning Class for SDXL LoRA
class SDXLFineTuningLoRA:
    """
    Implements LoRA fine-tuning for Stable Diffusion XL (SDXL-Turbo)
    """
    def __init__(self, model_name: str, tokenizer_name: str, text_encoder_1: str, text_encoder_2: str, data_path: str, train_batch_size: int,
                 val_batch_size: int, validation_size: float, learning_rate: float, gradient_accumulation_steps: int, gradient_clip: float, 
                 lora_rank: int, lora_alpha: int, lora_dropout: float, target_modules: list, save_path: str, device: str):
        """
        Initializes the fine-tuning pipeline.

        Arguments:
        ----------
            model_name                  { str } : Name of the pretrained SDXL model

            tokenizer_name              { str } : Tokenizer associated with the model

            text_encoder_1              { str } : First level text encoder name or path for SDXL 

            text_encoder_2              { str } : Second level text encoder name or path for SDXL 

            data_path                   { str } : Path to the dataset JSON file

            train_batch_size            { int } : Number of samples per batch for training

            val_batch_size              { int } : Number of samples per batch for validation

            validation_size            { float} : Fraction of the total dataset to be used for validation

            learning_rate             { float } : Learning rate for optimizer

            gradient_accumulation_steps { int } : Number of steps to accumulate gradients

            gradient_clip             { float } : Gradient clipping value

            lora_rank                   { int } : LoRA rank

            lora_alpha                  { int } : LoRA scaling factor

            lora_dropout              { float } : LoRA dropout probability

            target_modules             { list } : Target layers for LoRA adaptation

            save_path                   { str } : Directory to save fine-tuned model

            device                       { str } : Device to run training (e.g., "cuda", "cpu" or "mps")
        """
        print("\nInitializing fine-tuning of SDXL model with LoRA Adapter...")

        self.device                      = device

        self.tokenizer                   = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path = tokenizer_name)

        print("\nPreparing SDXL Traner Class...")
        # Initialize dataset trainer with separate train & validation batch sizes
        self.trainer                     = SDXLTrainer(tokenizer_name   = tokenizer_name, 
                                                       data_path        = data_path, 
                                                       train_batch_size = train_batch_size, 
                                                       val_batch_size   = val_batch_size,
                                                       val_split        = validation_size)

        print("\nLoading SDXL Pipeline, Unet and Text Encoders...")
        self.pipeline                    = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path = model_name, 
                                                                                     torch_dtype                   = torch.float16).to(self.device)

        self.unet                        = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path = model_name, 
                                                                                subfolder                     = "unet").to(self.device)

        self.text_encoder_1              = CLIPTextModel.from_pretrained(pretrained_model_name_or_path = text_encoder_1).to(self.device)
        
        self.text_encoder_2              = CLIPTextModel.from_pretrained(pretrained_model_name_or_path = text_encoder_2).to(self.device)


        print("\nPreparing SDXL model for training...")
        self.unet                        = prepare_model_for_kbit_training(self.unet)

        # Add a linear projection layer for text embeddings
        self.text_projection             = torch.nn.Linear(in_features  = 2048, 
                                                           out_features = 1280).to(self.device)

        # LoRA Configuration
        self.lora_config                 = LoraConfig(r              = lora_rank,
                                                      lora_alpha     = lora_alpha,
                                                      lora_dropout   = lora_dropout,
                                                      target_modules = target_modules,
                                                      bias           = "lora_only",
                                                     )
        
        # Apply LoRA to SDXL-Turbo
        self.unet                        = get_peft_model(model       = self.unet, 
                                                          peft_config = self.lora_config)

        # Set-up the Optimizer
        self.optimizer                   = AdamW(params = self.unet.parameters(), 
                                                 lr     = learning_rate)

        # Add gradient clipping
        self.gradient_clip               = gradient_clip

        # Add gradient accumulation steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.save_path                   = save_path
        
        # Track training and validation loss
        self.training_history            = {"epoch_losses"      : list(), 
                                            "batch_losses"      : list(), 
                                            "validation_losses" : list(), 
                                            "epoch_times"       : list()}

    def train(self, num_epochs:int):
        """
        Trains the SDXL model with LoRA

        Arguments:
        ----------
            num_epochs  { int } : Number of training epochs
        """
        # Set correct dtype based on the device
        dtype                                      = torch.float32 if self.device == "mps" else torch.float16
        
        # Move the model components to the correct device and dtype
        self.unet.to(self.device, dtype = dtype)
        self.pipeline.vae.to(self.device, dtype = dtype)
        self.text_encoder_1.to(self.device, dtype = dtype)
        self.text_encoder_2.to(self.device, dtype = dtype)
        self.text_projection.to(self.device, dtype = dtype)
        
        # Get training & validation dataloaders
        train_dataloader, val_dataloader           = self.trainer.get_dataloaders()
        
        # Initialize the best validation loss with a large value
        best_val_loss                              = float("inf")

        # Create epoch progress bar
        epoch_progress_bar                         = tqdm(range(num_epochs), 
                                                          desc     = "Epochs", 
                                                          position = 0)

        for epoch in epoch_progress_bar:
            epoch_start_time   = time.time()
            total_train_loss   = 0
            batch_train_losses = list()
            
            # Training Loop
            self.unet.train()

            # Create batch progress bar
            batch_progress_bar = tqdm(train_dataloader, 
                                      desc     = f"Training Epoch {epoch+1}", 
                                      position = 1, 
                                      leave    = False)
            
            for batch_idx, (images, captions) in enumerate(batch_progress_bar):
                # Move images to correct dtype and device
                images               = images.to(self.device, dtype = dtype)

                # Tokenize text
                tokenized_input      = self.tokenizer(captions, 
                                                      return_tensors = "pt", 
                                                      padding        = "max_length", 
                                                      truncation     = True, 
                                                      max_length     = 77).to(self.device)
                
                # Get text embeddings
                text_embeddings_1    = self.text_encoder_1(tokenized_input.input_ids)[0].to(dtype)
                text_embeddings_2    = self.text_encoder_2(tokenized_input.input_ids)[0].to(dtype)
                
                # Concatenate both embeddings and calculate the mean
                text_embeddings      = torch.cat([text_embeddings_1, text_embeddings_2], dim = -1)
                text_embeddings_mean = text_embeddings.mean(dim = 1).to(dtype)

                # Get image latents
                latents              = self.pipeline.vae.encode(images).latent_dist.sample()
                latents              = latents * self.pipeline.vae.config.scaling_factor
                latents              = latents.to(dtype)

                # Generate random timesteps
                timesteps            = torch.randint(0, 1000, (latents.shape[0],), device = self.device)
                timesteps            = timesteps.to(dtype)

                # Ensure correct shape for conditioning
                add_text_embeddings  = self.text_projection(text_embeddings_mean).to(dtype)

                # Prepare time embeddings
                add_time_ids         = torch.zeros((latents.shape[0], 1, 6), device = self.device, dtype = dtype) 

                # Clear gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                conditional_kwargs   = {"text_embeds" : add_text_embeddings,
                                        "time_ids"    : add_time_ids
                                       }

                # Forward pass
                outputs              = self.unet(latents, 
                                                 timesteps, 
                                                 encoder_hidden_states = text_embeddings, 
                                                 added_cond_kwargs     = conditional_kwargs,
                                                ).sample
                
                # Compute training loss
                train_loss           = torch.nn.functional.mse_loss(outputs, latents)
                
                # Scale loss for gradient accumulation
                scaled_train_loss    = train_loss / self.gradient_accumulation_steps  

                # Backward pass
                scaled_train_loss.backward()

                # Perform optimizer step after accumulating gradients for `gradient_accumulation_steps`
                if (((batch_idx + 1) % self.gradient_accumulation_steps == 0) or ((batch_idx + 1) == len(train_dataloader))):
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.gradient_clip)
                    
                    # Optimize gradients
                    self.optimizer.step()

                    # Clear gradients
                    self.optimizer.zero_grad()

                # Rescale loss for logging
                current_train_loss = scaled_train_loss.item() * self.gradient_accumulation_steps 
                
                # Track training loss
                total_train_loss  += current_train_loss
                batch_train_losses.append(current_train_loss)

                # Update batch progress bar
                batch_progress_bar.set_postfix({'current_train_loss' : f'{current_train_loss:.4f}', 
                                                'avg_train_loss'     : f'{total_train_loss / (batch_idx + 1):.4f}'
                                              })
                
            # Compute average training loss
            avg_train_loss = total_train_loss / len(train_dataloader)
            self.training_history["epoch_losses"].append(avg_train_loss)

            # Call validation
            avg_val_loss   = self.evaluate(val_dataloader, dtype)

            # Store validation loss
            self.training_history["validation_losses"].append(avg_val_loss)

            # Calculate epoch time
            epoch_time     = time.time() - epoch_start_time

            # Update epoch progress bar
            epoch_progress_bar.set_postfix({'avg_train_loss' : f'{avg_train_loss:.4f}', 
                                            'avg_val_loss'   : f'{avg_val_loss:.4f}', 
                                            'time_taken'     : f'{epoch_time:.1f}s'
                                          })
            
            # Print summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Average Training Loss: {avg_train_loss:.4f}")
            print(f"  Average Validation Loss: {avg_val_loss:.4f}")
            print(f"  Time Taken: {epoch_time:.1f} seconds")
            print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 150)
            print("\n")

            # Save only the best model
            if (avg_val_loss < best_val_loss):
                best_val_loss = avg_val_loss
                self.save_model()
                print(f"Best Model Updated at {self.save_path} (Validation Loss: {best_val_loss:.4f})")
            

    def evaluate(self, val_dataloader, dtype):
        """
        Evaluates the model on validation data and returns average validation loss

        Arguments:
        ----------
            val_dataloader { DataLoader }  : Validation data loader

            dtype          { torch.dtype } : Data type for model computations

        Returns:
        --------
            avg_val_loss { float }         : Average validation loss
        """
        # Set correct dtype based on the device
        dtype                                      = torch.float32 if self.device == "mps" else torch.float16

        # Move the model components to the correct device and dtype
        self.unet.to(self.device, dtype = dtype)
        self.pipeline.vae.to(self.device, dtype = dtype)
        self.text_encoder_1.to(self.device, dtype = dtype)
        self.text_encoder_2.to(self.device, dtype = dtype)
        self.text_projection.to(self.device, dtype = dtype)

        self.unet.eval()
        total_val_loss   = 0
        batch_val_losses = list()

        with torch.no_grad():
            batch_progress_bar = tqdm(val_dataloader, 
                                      desc     = "Validation", 
                                      position = 2, 
                                      leave    = False)

            for batch_idx, (images, captions) in enumerate(batch_progress_bar):
                images               = images.to(self.device, dtype = dtype)

                # Tokenize text
                tokenized_input      = self.tokenizer(captions, 
                                                      return_tensors = "pt", 
                                                      padding        = "max_length", 
                                                      truncation     = True, 
                                                      max_length     = 77).to(self.device)

                # Encode text
                text_embeddings_1    = self.text_encoder_1(tokenized_input.input_ids)[0].to(dtype)
                text_embeddings_2    = self.text_encoder_2(tokenized_input.input_ids)[0].to(dtype)

                # Concatenate both embeddings and calculate the mean
                text_embeddings      = torch.cat([text_embeddings_1, text_embeddings_2], dim = -1)
                text_embeddings_mean = text_embeddings.mean(dim = 1).to(dtype)

                # Get image latents
                latents              = self.pipeline.vae.encode(images).latent_dist.sample()
                latents              = latents * self.pipeline.vae.config.scaling_factor
                latents              = latents.to(dtype)

                # Generate random timesteps
                timesteps            = torch.randint(0, 1000, (latents.shape[0],), device = self.device)
                timesteps            = timesteps.to(dtype)

                # Prepare additional embeddings
                add_text_embeddings  = self.text_projection(text_embeddings_mean).to(dtype)
                add_time_ids         = torch.zeros((latents.shape[0], 1, 6), device = self.device, dtype = dtype)

                # Forward pass (Validation)
                conditional_kwargs   = {"text_embeds" : add_text_embeddings, 
                                        "time_ids"    : add_time_ids
                                       }
                outputs              = self.unet(latents, 
                                                 timesteps, 
                                                 encoder_hidden_states = text_embeddings, 
                                                 added_cond_kwargs     = conditional_kwargs,
                                                ).sample

                # Compute validation loss
                val_loss            = torch.nn.functional.mse_loss(outputs, latents)
                total_val_loss     += val_loss.item()
                batch_val_losses.append(val_loss.item())

                # Update validation batch progress bar
                batch_progress_bar.set_postfix({'current_val_loss' : f'{val_loss.item():.4f}', 
                                                'avg_val_loss'     : f'{total_val_loss / (batch_idx + 1):.4f}'})

        # Compute average validation loss
        avg_val_loss = total_val_loss / len(val_dataloader)

        return avg_val_loss


    def save_model(self):
        # Remove previous checkpoint
        if os.path.exists(self.save_path):
            os.system(f"rm -rf {self.save_path}")  
        
        # Create directory if not exists
        os.makedirs(self.save_path, 
                    exist_ok = True)  

        # Save LoRA-adapted model and base model separately
        self.unet.save_pretrained(self.save_path)
        self.unet.base_model.save_pretrained(os.path.join(self.save_path, "base_model"))

        # Save tokenizer along with the adapter
        self.tokenizer.save_pretrained(self.save_path)
        
        # Save training history
        history_path = os.path.join(self.save_path, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent = 4)

        print(f"SDXL-LoRA fine-tuned model saved at: {self.save_path}")
        print(f"Training history saved at: {history_path}")



#  Train the Model 
if __name__ == "__main__":

    # Global Configuration 
    DEVICE                       = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    MODEL_NAME                   = "stabilityai/stable-diffusion-xl-base-1.0"
    TOKENIZER_NAME               = "openai/clip-vit-large-patch14"
    TEXT_ENCODER_1_NAME          = "openai/clip-vit-large-patch14"
    TEXT_ENCODER_2_NAME          = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    DATA_PATH                    = "data/preprocessed_data/preprocessed_data.json"
    SAVE_PATH                    = "models/sdxl_lora_finetuned/"
    TRAIN_BATCH_SIZE             = 1
    VAL_BATCH_SIZE               = 1
    VALIDATION_SIZE              = 0.1
    NUM_EPOCHS                   = 5
    LEARNING_RATE                = 1e-5
    GRADIENT_ACCUMULATION_STEPS  = 4
    GRADIENT_CLIP                = 0.5

    # LoRA Hyperparameters
    LORA_RANK                    = 8
    LORA_ALPHA                   = 16
    LORA_DROPOUT                 = 0.05
    TARGET_MODULES               = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]  # Applies LoRA to attention layers only

    # Initialize Fine-Tuning
    fine_tuner                   = SDXLFineTuningLoRA(model_name                  = MODEL_NAME,
                                                      tokenizer_name              = TOKENIZER_NAME,
                                                      text_encoder_1              = TEXT_ENCODER_1_NAME,
                                                      text_encoder_2              = TEXT_ENCODER_2_NAME,
                                                      data_path                   = DATA_PATH,
                                                      train_batch_size            = TRAIN_BATCH_SIZE,
                                                      val_batch_size              = VAL_BATCH_SIZE,
                                                      validation_size             = VALIDATION_SIZE,
                                                      learning_rate               = LEARNING_RATE,
                                                      gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
                                                      gradient_clip               = GRADIENT_CLIP,
                                                      lora_rank                   = LORA_RANK,
                                                      lora_alpha                  = LORA_ALPHA,
                                                      lora_dropout                = LORA_DROPOUT,
                                                      target_modules              = TARGET_MODULES,
                                                      save_path                   = SAVE_PATH,
                                                      device                      = DEVICE,
                                                     )
    # Finetune the model
    fine_tuner.train(num_epochs = NUM_EPOCHS)
    
    # Save the model checkpoint
    fine_tuner.save_model()
