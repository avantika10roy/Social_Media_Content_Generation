# Model Evaluators

## Overview

The `src/model_evaluators` module provides functionality to evaluate both image generation models and large language models (LLMs). It is divided into two main components:

- **Image Model Evaluation:** Evaluates image generation quality using metrics such as CLIP Score, FID Score, LPIPS Score, and inference time measurement.
- **LLM Evaluation:** Assesses text generation performance using metrics including perplexity, semantic coherence, and BLEU score.

---

## Directory Structure

```plaintext
src/model_evaluators
├── __init__.py
├── image_model_evaluation.py  # Contains the ImageGenerationEvaluator class and evaluation functions for image models.
└── llm_evaluation.py          # Provides evaluation scripts and metrics for large language models.
```

### 1. Image Model Evaluation
#### Overview
The image_model_evaluation.py module evaluates image generation models by computing various quality metrics:

- **CLIP Score**: Measures text-image alignment using a CLIP model.
- **FID Score**: Computes the Fréchet Inception Distance between real and generated images.
- **LPIPS Score**: Evaluates perceptual similarity between images.
- **Inference Time**: Measures how long it takes for a model to perform inference.

#### Key Components
- ImageGenerationEvaluator:
- Initializes required models (CLIP, LPIPS, InceptionV3) for evaluation.
- Methods:
  - compute_clip_score(image_path, prompt): Returns a similarity score between the image and prompt.
  - compute_fid(real_image, generated_image): Computes FID between a pair of images.
  - compute_lpips(real_image_path, generated_image_path): Computes the LPIPS perceptual similarity.
  - measure_inference_time(model, input_tensor): Measures inference time for a given model and input.

#### Evaluation Workflow:
- Loads test data from a JSON file.
- Generates images (placeholder function generate_image should be replaced with actual generation logic).
- Computes all evaluation metrics.
- Saves results in CSV format and updates the JSON file.

#### Usage
Run the module as a standalone script to evaluate image generation performance:

```bash
python src/model_evaluators/image_model_evaluation.py
```
Note: Update the generate_image function and file paths as needed to integrate with your actual image generation workflow.

### 2. LLM Evaluation
#### Overview
The llm_evaluation.py module evaluates large language models (LLMs) based on several metrics:

- **Perplexity**: Measures fluency by computing the model's loss.
- **Coherence Score**: Uses Sentence Transformer embeddings to compute semantic similarity between the input prompt and the generated text.
- **BLEU Score**: Evaluates word-level overlap between reference and generated text.

#### Key Components
- Model & Tokenizer Loading:
  - Loads a Falcon-based LLM along with LoRA fine-tuning weights.
  - Uses Hugging Face's Transformers for tokenization and inference.

- Evaluation Metrics:
  - For each test sample, the script:
  - Tokenizes the prompt and generates text.
  - Computes the loss to derive perplexity.
  - Uses Sentence Transformers to compute semantic similarity.
  - Calculates the BLEU score for word-level coherence.
  - Aggregates results and prints average metrics for the dataset.

#### Usage
Run the module as a standalone script to evaluate LLM performance:

```bash
python src/model_evaluators/llm_evaluation.py
```
Note: Ensure that the model, tokenizer, LoRA weights, and test dataset paths are correctly configured in the script.

## Dependencies
Both evaluation modules rely on several external libraries. Install the required packages using pip:

```bash
pip install torch torchvision transformers peft sentence-transformers nltk lpips clip-by-openai scipy pandas pillow
```

## Final Notes
The evaluator modules are integral to the evaluation pipeline of the AI-Powered Social Media Content Generation System. They provide quantitative metrics to assess the quality of both image and text generation outputs, ensuring that generated content meets the desired standards.