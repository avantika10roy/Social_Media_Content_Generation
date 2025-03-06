# Data Preprocessor Module

## Overview
The `src/data_preprocessor` module is responsible for transforming, augmenting, and standardizing both text and image data. This module ensures the dataset is optimized for AI-powered social media content generation by refining text inputs and enhancing images.

---

## Preprocessors

### 1. Text Preprocessing
#### Description
The `text_preprocessing.py` module performs text cleaning, tokenization, lemmatization, and normalization.

#### Features
- Removes special characters, HTML tags, and extra spaces.
- Tokenizes text into meaningful words.
- Applies lemmatization to standardize words.
- Removes stopwords for improved model efficiency.

#### Dependencies
- `nltk`
- `re`
- `pandas`

#### Usage
```python
from src.data_preprocessor.text_preprocessing import TextPreprocessing

preprocessor = TextPreprocessing()
cleaned_data = preprocessor.preprocess(df, text_column="post_content")
```

### 2. Image Preprocessing
#### Description
The image_preprocessing.py module processes images by resizing, normalizing, augmenting, and filtering.

#### Features
- Resizes images to a standard format.
- Filters out GIF images and images containing faces.
- Performs data augmentation (color jitter, flipping).
- Saves transformed images into structured directories.

#### Dependencies
- `torchvision`
- `PIL`
- `cv2`
- `os`

#### Usage
```python
from src.data_preprocessor.image_preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor("data/raw_images", "data/preprocessed_images")
preprocessor.preprocess_images()
```

### 3. Image Augmentor
#### Description
The image_augmentor.py module enhances images by applying transformations like brightness adjustment, rotation, and noise addition.

#### Features
- Performs structured image augmentation.
- Extracts dominant colors and layouts from images.
- Applies machine-learning-based clustering for color extraction.

#### Dependencies
- albumentations
- sklearn
- cv2
- numpy
- pandas

#### Usage
```python
from src.data_preprocessor.image_augmentor import PreProcessor

augmentor = PreProcessor(img_size=(1024, 1024))
augmented_data = augmentor.image_augmentation(df, save_dir="data/augmented_images")
```

### 4. LLM Fine-Tune Data Preprocessor
#### Description
The llm_finetune_data_preprocessor.py module structures curated text data for fine-tuning Large Language Models.

#### Features
- Merges cleaned post contents from LinkedIn, Instagram, and Facebook.
- Extracts and combines textual data into structured training files.

#### Dependencies
- `json`

#### Usage
```python
from src.data_preprocessor.llm_finetune_data_preprocessor import merge_post_contents

merge_post_contents(linkedin_json="data/linkedin.json",
                    fb_json="data/facebook.json",
                    insta_json="data/instagram.json",
                    curated_json="data/curated.json",
                    output_json="data/final_curated.json"
                   )
```

## Directory Structure
```plaintext
src/
├── data_preprocessor/
│   ├── text_preprocessing.py
│   ├── image_preprocessing.py
│   ├── image_preprocessing2.py
│   ├── image_augmentor.py
│   ├── llm_finetune_data_preprocessor.py
│   ├── __init__.py
```

## Logging
Each module logs errors and processing steps to improve debugging. Logs are stored in the logs/ directory.

