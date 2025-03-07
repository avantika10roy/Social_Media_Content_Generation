import os
import json
import torch
from transformers import BlipProcessor, BlipModel, BlipForConditionalGeneration
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


json_path = os.path.join(os.path.dirname(__file__), "../../../data/raw_data/linkedin_raw_data.json")
image_base_path = os.path.join(os.path.dirname(__file__), "../../../data/raw_data/linkedin_raw_images")


try:
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded JSON file successfully: {json_path}")
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit(1)

def extract_image_features(image_path):
    """Extracts image features using BLIP vision model."""
    try:
        full_image_path = os.path.join(image_base_path, os.path.basename(image_path))
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            return None

        image = Image.open(full_image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.vision_model(**inputs).pooler_output
        
        return features.cpu().tolist()

    except Exception as e:
        print(f"Feature extraction failed for {image_path}: {e}")
        return None

def extract_text_features(text):
    """Extracts text features using BLIP language model."""
    try:
        inputs = processor(text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.text_model(**inputs).pooler_output
        
        return features.cpu().tolist()

    except Exception as e:
        print(f"Text feature extraction failed for: {text[:30]}... : {e}")
        return None

def extract_combined_features(image_path, text):
    """Extracts combined features using both image and text."""
    try:
        full_image_path = os.path.join(image_base_path, os.path.basename(image_path))
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            return None

        image = Image.open(full_image_path).convert("RGB")
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model(**inputs).vision_model_output.pooler_output
        
        return features.cpu().tolist()

    except Exception as e:
        print(f"Combined feature extraction failed for {image_path}: {e}")
        return None


for item in data:
    if "image_paths" not in item:
        print("Skipping entry, missing 'image_paths'.")
        continue

    image_paths = item["image_paths"].split(", ")
    post_text = item.get("post_contents", "").strip() 

    item["image_features"] = []
    item["text_features"] = extract_text_features(post_text) if post_text else None
    item["combined_features"] = []

    for image_path in image_paths:
        if not image_path.strip():
            continue
        
        img_features = extract_image_features(image_path)
        combined_features = extract_combined_features(image_path, post_text) if post_text else None

        if img_features is not None:
            item["image_features"].append(img_features)
        if combined_features is not None:
            item["combined_features"].append(combined_features)

# Saving
output_json_path = os.path.join(os.path.dirname(__file__), "../../../data/extracted_features_data/blip_output.json")

try:
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Feature extraction complete. Saved to {output_json_path}")
except Exception as e:
    print(f"Error saving features: {e}")
