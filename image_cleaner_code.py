import os
import shutil
import pandas as pd
import logging

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def validate_path(path, base_dir):
    """Validate and resolve file path"""
    try:
        if path.startswith('./'):
            full_path = os.path.abspath(os.path.join(base_dir, path[2:]))
        else:
            full_path = os.path.abspath(image_dir,path)
        
        # Check if the path exists and is a file
        if not os.path.isfile(full_path):
            logging.warning(f"File not found: {full_path}")
            return None
        return full_path
    except Exception as e:
        logging.error(f"Error validating path {path}: {e}")
        return None

def move_images(data, dest_folder, base_dir):
    """Move images to destination folder with error handling"""
    os.makedirs(dest_folder, exist_ok=True)
    moved_files = []
    failed_files = []

    for paths in data['image_paths'].dropna().values:
        for path in [p.strip() for p in paths]:
        #for path in [p.strip() for p in paths.split(',')]:
            valid_path = validate_path(path, base_dir)
            if valid_path:
                dest_path = os.path.join(dest_folder, os.path.basename(valid_path))
                try:
                    shutil.copy2(valid_path, dest_path)  # Using copy2 instead of move for safety
                    moved_files.append(valid_path)
                    logging.info(f"Copied: {valid_path} -> {dest_path}")
                except Exception as e:
                    failed_files.append(valid_path)
                    logging.error(f"Failed to copy {valid_path}: {e}")

    return moved_files, failed_files

def filter_images(data, keywords, base_dir):
    """Filter images based on keywords and move them to appropriate folders"""
    setup_logging()
    
    # Create regex pattern from keywords
    pattern = '|'.join(map(str.lower, keywords))
    
    # Create necessary directories
    invalid_dir = "./data/cleaned_data/instagram_cleaned_images/invalid_images/"
    valid_dir = "./data/cleaned_data/instagram_cleaned_images/cleaned_images/"
    
    # Identify invalid posts based on keywords in heading or content
    invalid_posts = data[
        data['post_heading'].str.lower().str.contains(pattern, na=False) |
        data['post_content'].str.lower().str.contains(pattern, na=False)
    ]
    
    # Move images from invalid posts
    logging.info("Moving invalid images...")
    invalid_moved, invalid_failed = move_images(invalid_posts, invalid_dir, base_dir)
    
    # Drop invalid posts from the original dataset
    cleaned_data = data.drop(invalid_posts.index)
    
    # Move remaining images
    logging.info("Moving valid images...")
    valid_moved, valid_failed = move_images(cleaned_data, valid_dir, base_dir)
    
    # Log summary
    logging.info(f"""
    Processing complete:
    - Invalid images moved: {len(invalid_moved)}
    - Invalid images failed: {len(invalid_failed)}
    - Valid images moved: {len(valid_moved)}
    - Valid images failed: {len(valid_failed)}
    """)
    
    return cleaned_data

# Example usage
keywords = [
    'celebrated', 'celebration', 'throwback', 'lunch',
    'dinner', 'anniversary', 'outing', 'enjoyed'
]

# Load data
try:
    #face = pd.read_json('./data/cleaned_data/facebook_cleaned_data.json')
    insta = pd.read_json('./data/cleaned_data/instagram_cleaned_data.json')
except Exception as e:
    logging.error(f"Error loading JSON file: {e}")
    exit(1)

# Define base directory
#base_dir = "/Users/it012305/Desktop/Project/social_media_content_generation"
base_dir = "/Users/soumik/Movies/social_media_content_generation"
#image_dir = "/Users/soumik/Desktop/facebook_raw_images"
image_dir = "/Users/soumik/Desktop/instagram_raw_images"

# Process data
#logging.info(f"Original data shape: {face.shape}")
#cleaned_data = filter_images(face, keywords, base_dir)
#logging.info(f"Cleaned data shape: {cleaned_data.shape}")

logging.info(f"Original data shape: {insta.shape}")
cleaned_data = filter_images(insta, keywords, base_dir)
logging.info(f"Cleaned data shape: {cleaned_data.shape}")
