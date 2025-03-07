# Model Fine-Tuners

## Overview

The `src/model_finetuners` module contains scripts for fine-tuning both image generation and language models using PEFT (Parameter-Efficient Fine-Tuning) techniques such as LoRA. These scripts help adapt pre-trained models to domain-specific tasks while minimizing the number of trainable parameters and computational overhead.

This module is organized into two main sections:
- **Image Model Fine-Tuners:** Contains scripts to fine-tune image generation models like Stable Diffusion XL (SDXL) using LoRA.
- **LLM Fine-Tuners:** Contains scripts for fine-tuning large language models (LLMs), including generic LLM fine-tuning, T5-based models using LoRA, and FLAN-T5 fine-tuning.

---

## Directory Structure

```plaintext
src/model_finetuners
├── image_model_finetuners
│   ├── __init__.py
│   └── sdxl_finetuning.py         # Fine-tuning script for Stable Diffusion XL using LoRA.
└── llm_finetuners
    ├── __init__.py
    ├── flan_t5_finetuner.py       # Fine-tuning script for FLAN-T5 using LoRA.
    ├── llm_fine_tuner.py          # Generic LLM fine-tuning using PEFT techniques.
    └── t5_lora_finetuning.py      # Fine-tuning script for T5-based models using LoRA.
```

## Components

### 1. Image Model Fine-Tuners

#### SDXL Fine-Tuning

- **File:** `image_model_finetuners/sdxl_finetuning.py`

- **Description:**  
  Fine-tunes the Stable Diffusion XL model using LoRA adaptation. This script:
  - Loads a custom dataset of image–caption pairs.
  - Implements an `SDXLTrainer` class for data loading and splitting.
  - Provides an `SDXLFineTuningLoRA` class that applies LoRA to the UNet and trains the model.

- **Usage:**  
  Run the script as a standalone module. Adjust training hyperparameters (batch sizes, learning rate, LoRA rank, etc.) and dataset paths as needed.

- **Example:**  
  ```bash
  python src/model_finetuners/image_model_finetuning.py
  ```

### 2. LLM Fine-Tuners

#### Generic LLM Fine-Tuning

- **File:** `llm_finetuners/llm_fine_tuner.py`

- **Description:**
  Provides a generic fine-tuning framework for large language models using PEFT. The LLMFineTuner class:
  - Loads a pre-trained LLM and its tokenizer.
  - Configures LoRA settings and fine-tuning parameters.
  - Utilizes Hugging Face's Trainer API to train and save the model.

- **Usage:**
  Instantiate and run the fine-tuning process within your training pipeline. Adjust dataset, model paths, and training arguments as required.


#### T5 LoRA Fine-Tuning

- **File:** `llm_finetuners/t5_lora_finetuning.py`

- **Description:**
  Fine-tunes T5-based models (e.g., Falcon, T5 variants) using LoRA. This script:

  - Defines a custom dataset for processing input–output pairs.
  - Implements an LLMTrainer class to handle tokenization and data loading.
  - Provides an LLMFineTuningLoRA class that applies LoRA, sets up an optimizer, and trains the model.

- **Usage:**
  Run the script with your custom data and adjust training hyperparameters such as learning rate, batch size, and number of epochs.

- **Example:**
  ```bash
  python src/model_finetuners/llm_finetuners/t5_lora_finetuning.py
  ```   

#### FLAN-T5 Fine-Tuning

- **File:** `llm_finetuners/flan_t5_finetuner.py`

- **Description:**
  Specifically designed to fine-tune FLAN-T5 models using PEFT techniques. The FLAN_T5_FineTuner class:

  - Loads the FLAN-T5 model and its tokenizer.
  - Applies a LoRA configuration to adapt the model.
  - Sets up the Trainer with specified training arguments and dataset splits.

- **Usage:**
  Configure the training arguments and LoRA parameters, then run the fine-tuning process.

- **Example:**

  ```bash
  python src/model_finetuners/llm_finetuners/flan_t5_finetuner.py
  ```


## Dependencies
All fine-tuning scripts require the following libraries:

- PyTorch: For deep learning model training.
- Transformers: From Hugging Face for model loading and training.
- PEFT: For applying LoRA and other parameter-efficient fine-tuning methods.
- Datasets: For data handling.
- TQDM: For progress visualization during training.
- Pillow: For image processing.
- Other Utilities: Such as os, json, time, and logging utilities.

Install the required packages using pip:

```bash
pip install torch transformers peft datasets tqdm pillow
```

## Final Notes
The scripts in this module enable efficient adaptation of pre-trained models to your specific tasks using LoRA. Adjust configuration parameters, dataset paths, and training hyperparameters as needed to suit your use case. For detailed usage and further customization, refer to the inline documentation within each script.