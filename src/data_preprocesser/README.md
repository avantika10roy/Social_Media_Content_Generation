# Data Preprocessor Module

## Overview

The `src/data_preprocesser` module is responsible for transforming, augmenting, and standardizing both text and image data. This module ensures that the dataset is optimized for AI-powered social media content generation by performing data cleaning, feature extraction, augmentation, and logo identification.

The module is divided into two main packages:
- **Text Preprocessor:** Cleans, tokenizes, and structures text data, including preparing curated content for LLM fine-tuning.
- **Image Preprocessor:** Processes and augments images, extracts visual features using different models, and identifies logos in images.

---

## Directory Structure

```plaintext
src/data_preprocesser
├── README.md
├── image_preprocessor
│   ├── __init__.py
│   ├── blip_feature_extraction.py         # Extracts image and text features using the BLIP model.
│   ├── clip_feature_extraction.py           # Extracts image features using a pretrained ResNet50 and simple text embeddings.
│   ├── image_augmentor.py                   # Performs image augmentation, dominant color extraction, and layout determination.
│   └── logo_identification.py               # Identifies logos in images by uploading them and cleaning the logo information.
└── text_preprocessor
    ├── __init__.py
    ├── llm_finetune_data_preprocessor.py    # Merges and structures text data from multiple sources for LLM fine-tuning.
    └── text_preprocessing.py                # Cleans, tokenizes, and normalizes raw text data.
```

## Preprocessors
### 1. Text Preprocessing
#### Description
The text_preprocessing.py module cleans and normalizes raw text by removing HTML tags, special characters, and stopwords. It also tokenizes and lemmatizes the input text.

#### Features
- Removal of HTML tags and special characters.
- Tokenization and lemmatization.
- Stopword filtering using NLTK.
- Logging of preprocessing steps and potential errors.

#### Dependencies
- `nltk`
- `re`
- `pandas`
- `src/utils/logger (for logging)`

#### Usage
```python
from src.data_preprocesser.text_preprocessor.text_preprocessing import TextPreprocessing

preprocessor = TextPreprocessing()
cleaned_df = preprocessor.preprocess(df, text_column="post_content")
```

### 2. LLM Fine-Tune Data Preprocessor
#### Description
The llm_finetune_data_preprocessor.py module structures curated text data for fine-tuning Large Language Models. It merges post contents from LinkedIn, Facebook, and Instagram into a unified format, appending raw content to the curated dataset.

#### Features
- Merging of post data from multiple JSON files.
- Extraction and integration of post contents into a curated JSON structure.
- Saving the final curated data for use in model fine-tuning.

#### Dependencies
- `json`

#### Usage
```python
from src.data_preprocesser.text_preprocessor.llm_finetune_data_preprocessor import merge_post_contents

merge_post_contents(
    linkedin_json="data/linkedin.json",
    fb_json="data/facebook.json",
    insta_json="data/instagram.json",
    curated_json="data/curated.json",
    output_json="data/final_curated.json"
)
```


### 3. BLIP Feature Extraction
#### Description
The blip_feature_extraction.py module leverages Salesforce's BLIP model to extract features from images and text, as well as combined features. The outputs are saved into a JSON file for further use.

#### Features
- Image feature extraction using the BLIP vision model.
- Text feature extraction using the BLIP language model.
- Combined feature extraction from both modalities.
- Saving extracted features to a structured JSON file.

#### Dependencies
- `transformers (BlipProcessor, BlipModel, BlipForConditionalGeneration)`
- `torch`
- `PIL`
- `json`
- `os`

#### Usage
Run the module directly or integrate it as part of a feature extraction pipeline to process raw LinkedIn data.

### 4. CLIP Feature Extraction
#### Description
The clip_feature_extraction.py module uses a pretrained ResNet50 (via PyTorch) to extract image features. It also computes simple text features and averages them with the image features to obtain combined features.

#### Features
- Image feature extraction using ResNet50.
- Basic text feature extraction.
- Combined feature calculation.
- Reads raw JSON data and saves the output with extracted features.

#### Dependencies
- `torch`
- `torchvision`
- `PIL`
- `json`
- `os`

#### Usage
Run the module to process raw LinkedIn data and save the extracted features into a designated JSON file.

### 5. Image Augmentor
#### Description
The image_augmentor.py module performs image augmentation and preprocessing. It:

- Converts dataset entries to separate rows based on individual image paths.
- Augments images using transformations such as brightness/contrast adjustments, rotation, and noise addition.
- Extracts dominant colors using clustering.
- Determines the layout (e.g., text-heavy vs. minimalist) of the images.

#### Dependencies
- `albumentations`
- `cv2 (OpenCV)`
- `numpy`
- `pandas`
- `sklearn (MiniBatchKMeans)`
- `src/utils/logger`
- `src/utils/color_themes`

#### Usage
```python
from src.data_preprocesser.image_preprocessor.image_augmentor import PreProcessor

augmentor = PreProcessor(img_size=(1024, 1024))
augmented_df = augmentor.run_preprocessor(df, save_dir="data/augmented_images")
```


### 6. Logo Identification
#### Description
The logo_identification.py module identifies logos within images. It uploads images to a logo detection API, processes the response, and cleans the logo information to be appended to the dataset.

#### Features
- Uploads images to a remote API for logo detection.
- Generates and cleans logo information.
- Updates the dataset with logo details.

#### Dependencies
- `PIL`
- `requests`
- `base64`
- `io`
- `json`
- `os`
- `Custom configuration from config/config.py`

#### Usage
Run the module as a standalone script to process an existing JSON file containing image paths. The updated JSON will include logo information.

## Logging
Each module in the data preprocessor logs key steps, errors, and warnings to help with debugging. Logs are stored in the designated logs/ directory as configured in the src/utils/logger module.

