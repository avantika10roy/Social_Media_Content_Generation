import os
import requests
import base64
import io
import json
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.config import Config


def upload_image(img_path, api_key):
    image = Image.open(img_path).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode("ascii")

    upload_url = f"https://detect.roboflow.com/augmented-data-pq0mq/3?api_key={api_key}&name={os.path.basename(img_path)}"
    r = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })
    return r.json()


def generate_logo_info(image_path):
    return {"logo": f"Logo_for_{image_path.split('/')[-1]}"}


folder_path = ""
MY_KEY = "B41hvr6rdhF8sUZnxAbd"


with open(Config.CURATED_POST_DATA_PATH , 'r') as file:
    data = json.load(file)


for post in data:
    logo_info_list = []
    for image_path in post["image_paths"]:
        logo_info = generate_logo_info(image_path)
        img_path = os.path.join(folder_path, image_path)
        response = upload_image(img_path, MY_KEY)
        print(f"Response for {image_path}: {response}")
        logo_info_list.append(response)
    
    post["logo_info"] = logo_info_list
with open(Config.LOGO_INFO_OUTPUT_PATH, 'w') as output_file:
    json.dump(data, output_file, indent=4)

print("Logo info has been added, images uploaded, and the new file has been saved.")
