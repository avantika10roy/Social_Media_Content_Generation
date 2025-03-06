# Utils Module

## Overview
The `src/utils` module provides utility functions and helper classes for logging, data management, Google Drive downloads, color themes, and setting random seeds. These utilities are used across different modules in the AI-powered social media content generation system.

---

## Utilities

### 1. Logger
#### Description
The `logger.py` module provides a standardized logging setup for the entire project.

#### Features
- Logs messages with timestamps and module information.
- Supports different logging levels (INFO, WARNING, ERROR, DEBUG).
- Saves logs to a structured `logs/` directory.

#### Dependencies
- `logging`
- `sys`
- `datetime`
- `pathlib`

#### Usage
```python
from src.utils.logger import LoggerSetup

logger = LoggerSetup(logger_name="my_logger", log_filename_prefix="app").get_logger()
logger.info("This is an informational message.")
```

### 2. Data Saver
#### Description
The data_saver.py module provides functions to save and load JSON data efficiently.

#### Features
- Saves Pandas DataFrames and lists to JSON files.
- Reads JSON files and converts them into DataFrames.
- Handles missing files and invalid formats.

####Dependencies
- `json`
- `pandas`
- `os`

#### Usage
```python
from src.utils.data_saver import DataSaver

data = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
DataSaver.data_saver(data, "output.json")

df = DataSaver.data_reader("output.json")
print(df)
```

### 3. Google Drive Downloader
#### Description
The download_from_drive.py module facilitates downloading files and folders from Google Drive using the PyDrive2 library.

#### Features
- Authenticates with Google Drive API.
- Downloads entire folders and saves them locally.
- Extracts folder IDs from Google Drive URLs.

#### Dependencies
- `pydrive2`
- `os`

#### Usage
```python
from src.utils.download_from_drive import DownloadData

downloader = DownloadData(client_secrets_path="client_secrets.json")
downloader.download_google_drive_folder("https://drive.google.com/drive/folders/your-folder-id", "local_folder")
```

### 4. Color Themes
#### Description
The color_themes.py module provides a predefined dictionary of named colors and their RGB values.

#### Features
- Defines a set of common colors.
- Maps color names to their respective RGB values.
- Used for image processing and clustering.

#### Dependencies
- `None`

#### Usage
```python
from src.utils.color_themes import get_color_themes

colors = get_color_themes()
print(colors["red"])  # Output: (255, 0, 0)
```

### 5. Random Seed Setter
#### Description
The set_seed.py module sets a global random seed for reproducibility across NumPy, PyTorch, and Python's random module.

#### Features
- Ensures deterministic results by fixing random seeds.
- Works across multiple libraries (NumPy, PyTorch, Transformers).
- Helps in model training and testing reproducibility.

#### Dependencies
- `torch`
- `numpy`
- `random`
- `transformers`

#### Usage
```python
from src.utils.logger import LoggerSetup
from src.utils.set_seed import set_global_seed

logger = LoggerSetup(logger_name="seed_logger").get_logger()
set_global_seed(logger, seed=42)
```

## Directory Structure
```plaintext
src/
├── utils/
│   ├── logger.py
│   ├── data_saver.py
│   ├── download_from_drive.py
│   ├── color_themes.py
│   ├── set_seed.py
│   ├── __init__.py
```

## Logging
Each utility module logs its actions for easier debugging and monitoring. Logs are stored in the logs/ directory.
