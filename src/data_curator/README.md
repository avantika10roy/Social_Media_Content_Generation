# Data Curation Module

## Overview
The `src/data_curation` module is responsible for merging, refining, and structuring cleaned social media data from different platforms (LinkedIn, Instagram, and Facebook). It ensures that the curated data maintains quality and consistency before being used for AI-powered content generation.

---

## Curators

### 1. Data Curation
#### Description
The `data_curation.py` module consolidates cleaned text and images from multiple platforms into a single structured dataset.

#### Features
- Reads cleaned JSON files from LinkedIn, Instagram, and Facebook.
- Merges them into a unified dataset.
- Saves the curated data to a structured JSON file.
- Copies and updates image paths for uniform storage.

#### Dependencies
- `pandas`
- `json`
- `shutil`
- `os`

#### Usage
```python
from src.data_curation.data_curation import DataCuration

curator = DataCuration(
    linkedin_cleaned_data_path="path/to/linkedin_cleaned_data.json",
    instagram_cleaned_data_path="path/to/instagram_cleaned_data.json",
    facebook_cleaned_data_path="path/to/facebook_cleaned_data.json"
)

curated_data = curator.text_curation()
curator.image_curation(json_path="path/to/curated_data.json", curated_images_dir="path/to/curated_images")
```

### 2. Mix Curator
#### Description
The mix_curator.py module further refines the curated data by standardizing text content, extracting metadata like hashtags and emojis, and integrating raw post references.

#### Features
- Cleans text by removing emojis, hashtags, extra spaces, and HTML tags.
- Extracts and organizes hashtags from raw posts.
- Merges data from raw and curated sources to provide additional context.
- Saves the processed data in a structured JSON format.

#### Dependencies
- `pandas`
- `json`
- `emoji`
- `re`
- `shutil`
- `os`

#### Usage
```python
from src.data_curation.mix_curator import clean_text

cleaned_text = clean_text("Example #post with an emoji 😊 and some <b>HTML</b>.")
print(cleaned_text)
```

## Directory Structure
```plaintext
src/
├── data_curation/
│   ├── data_curation.py
│   ├── mix_curator.py
│   ├── __init__.py
```

## Logging
Each curator generates log files for debugging and tracking data curation processes. Logs are stored in the logs/ directory.
