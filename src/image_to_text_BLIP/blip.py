import os
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import sys
import time
import torch  # Import torch to handle device management

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.config import Config

# Set the device to MPS (Apple Silicon GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BLIP model and processor
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)  # Move model to MPS
    print("BLIP model and processor loaded successfully!")
except Exception as e:
    print(f"Error loading BLIP model: {e}")
    exit()

# Function to generate captions
def generate_caption(image_path):
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt").to(device)  # Move inputs to MPS
        
        out = model.generate(
            **inputs,
            max_length=100,
            num_beams=15,
            temperature=0.9,
            top_k=200,
            do_sample=True,
            early_stopping=True
        )
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return "Caption not available (error)"

# Path to the curated JSON file
curated_json_path = Config.CURATED_POST_DATA_PATH

# Load the JSON data
try:
    with open(curated_json_path, "r") as f:
        data = json.load(f)
    print("JSON data loaded successfully!")
except Exception as e:
    print(f"Error loading JSON file: {e}")
    exit()

# Counter for progress updates
image_counter = 0

# Process each item in the JSON
start_time = time.time()
for item in data:
    captions = []
    for image_path in item["image_paths"]:
        try:
            # Generate caption for the image
            caption = generate_caption(image_path)
            captions.append(caption)
            
            # Increment the counter
            image_counter += 1
            
            # Print progress after every 10 images
            if image_counter % 10 == 0:
                current_time = time.time()
                time_diff = current_time - start_time
                print(f"{image_counter} done, Running for {time_diff:.2f} sec")
                
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            captions.append("Caption not available (error)")
    
    # Add the captions to the JSON under the key "contexts"
    item["contexts"] = captions

# Create a new folder named "Blip_with_context" under the "data" folder
output_folder = Config.BLIP_OUTPUT_PATH
os.makedirs(output_folder, exist_ok=True)

# Save the updated JSON with captions
output_json_path = os.path.join(output_folder, "blip_image_context.json")
try:
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Captions added and JSON saved successfully at {output_json_path}!")
except Exception as e:
    print(f"Error saving JSON file: {e}")