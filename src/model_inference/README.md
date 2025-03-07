# Model Inference

## Overview

The `src/model_inference` module provides the inference pipelines for both image generation and text (LLM) generation. This module allows you to generate AI-powered outputs using fine-tuned models and LoRA adapters. It is divided into two main sections:

- **Image Model Inference:** Uses a fine-tuned SDXL model with LoRA adapters to generate images.
- **LLM Inference:** Contains inference scripts for generating text using various fine-tuned language models, including FLAN-T5, generic LLMs, and T5-based models with LoRA.

---

## Directory Structure

```plaintext
src/model_inference
├── README.md
├── image_model_inference
│   └── sdxl_inference.py         # Performs image generation using the SDXL pipeline with LoRA adapters and optional post-processing (text overlay and logo placement).
└── llm_inference
    ├── flan_t5_inference.py       # Inference class for FLAN-T5 that loads a model (optionally with LoRA weights) and generates text based on prompts.
    ├── llm_inference.py           # Generic LLM inference using PEFT; supports models like Mistral-7B-Instruct and Falcon-1B-Instruct.
    └── t5_inference.py            # Inference class for T5-based models with LoRA adapters, optimized for generating social media posts.
```

## Components

### 1. Image Model Inference

#### SDXL Inference

- **File:** `image_model_inference/sdxl_inference.py`

- **Description:**
  This script leverages a fine-tuned SDXL model with LoRA adapters to generate images. It performs the following:
  - Loads the base SDXL model and applies the fine-tuned LoRA adapter.
  - Uses a refiner model to improve image quality.
  - Supports post-processing steps such as overlaying custom text (with transparent text overlays) and logos on the generated image.
  - Optionally uses a prompt enhancer (via a Zephyr model) to format the positive and negative prompts.

- **Usage:**
  Run the script as a standalone module. Adjust model paths, image dimensions, prompts, and additional options (like logo placement and text overlay) as needed.

- **Example:**
  ```bash
  python src/model_inference/image_model_inference/sdxl_inference.py
  ```

### 2. LLM Inference

#### FLAN-T5 Inference

- **File:** `llm_inference/flan_t5_inference.py`

- **Description:**
  Provides an inference class (FLAN_T5_Inference) that:
  - Loads a FLAN-T5 model and its tokenizer, optionally with LoRA weights.
  - Contains methods to set up prompts using chain-of-thought reasoning.
  - Generates text based on the given prompt and configuration.

- **Usage:**
  Configure model paths and prompt parameters, then run the script to generate text output.

- **Example:**
  ```bash
  python src/model_inference/llm_inference/flan_t5_inference.py
  ```


#### Generic LLM Inference

- **File:** `llm_inference/llm_inference.py`

- **Description:**
  This script provides a generic inference framework for language models using PEFT. The LLMInference class:
  - Loads a base LLM along with LoRA weights.
  - Processes an input prompt to generate text outputs.
  - Supports models such as Mistral-7B-Instruct and Falcon-1B-Instruct.

- **Usage:**
  Simply instantiate the LLMInference class with appropriate model, tokenizer, and LoRA paths, and call the generate method with your prompt.

- **Example:**
  ```python
  from src.model_inference.llm_inference import LLMInference

  inference_engine = LLMInference(model_path     = "path/to/model",
                                  tokenizer_path = "path/to/tokenizer",
                                  lora_path      = "path/to/lora")

  output_text      = inference_engine.generate("Your prompt here")
  
  print(output_text)
  ```

#### T5 Inference

**File:** `llm_inference/t5_inference.py`

- **Description:**
  The T5LoRAInference class is designed for generating social media posts using a fine-tuned T5 model with LoRA adapters. It:
  - Loads the T5 model and its tokenizer from a specified directory.
  - Validates the presence of required adapter files.
  - Generates text posts based on a structured prompt, ensuring that the output meets the expected format.

- **Usage:**
  Initialize the T5LoRAInference class with the model directory and device, then use the generate_post method to create a post.

- **Example:**    
  ```bash
  python src/model_inference/llm_inference/t5_inference.py
  ```


## Dependencies
The inference modules require several Python packages. Ensure you have installed the following:

- PyTorch: For model computations.
- Transformers: From Hugging Face for model loading and tokenization.
- PEFT: For applying LoRA to pre-trained models.
- Diffusers: For image generation pipelines.
- Llama_cpp: For local Zephyr model inference (if used).
- Pillow: For image processing.
- Other utilities: Such as os, json, re, time, numpy, and logging packages.

You can install the required packages with:

```bash
pip install torch transformers peft diffusers llama-cpp pillow numpy
```

## Final Notes
The src/model_inference module is designed to seamlessly integrate AI-driven image and text generation into your application. Customize model paths, hyperparameters, and prompts as necessary to suit your needs.