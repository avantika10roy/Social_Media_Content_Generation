# Data Cleaner Module

## Overview
The `src/data_cleaner` module is responsible for cleaning and preprocessing scraped social media content. It ensures the extracted data is structured, free from noise, and ready for AI-powered content generation. The module consists of two main components:

1. **Text Cleaner** - Cleans and processes text data by removing unnecessary elements such as emojis, hashtags, and HTML tags.
2. **Image Cleaner** - Filters and processes image data, ensuring only relevant images are retained.

---

## Cleaners

### 1. Text Cleaner
#### Description
The `text_cleaner.py` module processes text data from scraped social media posts. It extracts important elements like emojis and hashtags while cleaning the text to maintain consistency.

#### Features
- Extracts emojis and hashtags.
- Removes unnecessary characters like HTML tags, extra spaces, and special symbols.
- Splits text into headings and content.
- Cleans LinkedIn and Instagram post content.

#### Dependencies
- `pandas`
- `re`
- `emoji`
- `shutil`
- `logging`

#### Usage
```python
from src.data_cleaner.text_cleaner import DataCleaner

cleaner = DataCleaner()
cleaned_data = cleaner.clean_text(df["post_contents"])
```

### 2. Image Cleaner
#### Description
The image_cleaner.py module handles image processing tasks such as filtering irrelevant images and copying valid ones to structured directories.

#### Features
- Filters images based on specified keywords.
- Removes invalid image formats (e.g., GIFs mistakenly saved as JPGs or PNGs).
- Copies relevant images to a structured directory.

#### Dependencies
- `pandas`
- `PIL (Pillow)`
- `shutil`
- `os`

#### Usage
```python
from src.data_cleaner.image_cleaner import ImageCleaner

cleaner = ImageCleaner()
cleaned_images = cleaner.filter_and_copy_images(df, platform="linkedin")
```

## Directory Structure
```plaintext
src/
├── data_cleaner/
│   ├── text_cleaner.py
│   ├── image_cleaner.py
│   ├── __init__.py
```

## Logging
Both modules generate log files for debugging and tracking cleaning operations. Logs are stored in the logs/ directory.
