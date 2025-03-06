import os
import json
import torch
from torchvision import models, transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    """Extracts image features using ResNet."""
    try:
        full_image_path = os.path.join(image_base_path, os.path.basename(image_path))
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            return None

        image = transform(Image.open(full_image_path).convert("RGB")).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model(image)
        
        return features.cpu().tolist()
    except Exception as e:
        print(f"Feature extraction failed for {image_path}: {e}")
        return None

def extract_text_features(text):
    """Extracts text features using a simple embedding."""
    try:
        text_tensor = torch.tensor([ord(c) for c in text[:512]], dtype=torch.float32).to(device)
        
        return text_tensor.mean().item()
    except Exception as e:
        print(f"Text feature extraction failed for: {text[:30]}... : {e}")
        return None

def extract_combined_features(image_path, text):
    """Extracts combined features by averaging image and text features."""
    try:
        image_features = extract_image_features(image_path)
        text_features = extract_text_features(text)
        
        if image_features is None or text_features is None:
            return None
        
        combined_features = [(x + text_features) / 2 for x in image_features[0]]
        return combined_features
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
output_json_path = os.path.join(os.path.dirname(__file__), "../../../data/extracted_features_data/clip_output.json")

try:
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Feature extraction complete. Saved to {output_json_path}")
except Exception as e:
    print(f"Error saving features: {e}")
