import json
import sys
import os
import pandas as pd
from apify_client import ApifyClient
import requests
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.config import Config

cf = Config()

class FacebookScraper:
    def __init__(self, api_token, page_url, results_limit=500, dataset_file="facebook_data.json"):
        self.client = ApifyClient(api_token)
        self.page_url = page_url
        self.results_limit = results_limit
        self.dataset_file = dataset_file
    
    def scrape_data(self):
        try:
            run_input = {"startUrls": [{"url": self.page_url}], "resultsLimit": self.results_limit}
            run = self.client.actor("apify/facebook-posts-scraper").call(run_input=run_input)
            dataset_id = run["defaultDatasetId"]
            items = list(self.client.dataset(dataset_id).iterate_items())
            with open(self.dataset_file, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=4)
            print(f"✅ Data saved to {self.dataset_file}")
        except Exception as e:
            print(f"❌ Error scraping data: {e}")

class FacebookDataProcessor:
    def __init__(self, dataset_file, csv_file, output_json, image_dir="./facebook_raw_images"):
        self.dataset_file = dataset_file
        self.csv_file = csv_file
        self.output_json = output_json
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)
    
    def process_data(self):
        try:
            with open(self.dataset_file, 'r') as file:
                data = json.load(file)
            
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['post_url', 'image_url', 'caption']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for post in data:
                    post_url = post['url']
                    caption = post['text']
                    for media in post.get('media', [])[1:]:
                        image_url = media.get('image', {}).get('uri', '')
                        writer.writerow({'post_url': post_url, 'image_url': image_url, 'caption': caption})
            print("✅ CSV file created successfully!")
        except Exception as e:
            print(f"❌ Error processing data: {e}")
    
    def download_images(self):
        try:
            df = pd.read_csv(self.csv_file)
            df = df.drop(columns=['post_url'])
            
            def download_image(image_url, index):
                try:
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        image_name = f"image_{index}.jpg"
                        image_path = os.path.join(self.image_dir, image_name)
                        with open(image_path, "wb") as file:
                            for chunk in response.iter_content(1024):
                                file.write(chunk)
                        return image_name
                    else:
                        return None
                except Exception as e:
                    print(f"Error downloading {image_url}: {e}")
                    return None
            
            df["image_name"] = df.apply(lambda row: download_image(row["image_url"], row.name), axis=1)
            df.to_csv(self.csv_file, index=False)
            print(f"✅ Images downloaded and CSV updated: {self.csv_file}")
        except Exception as e:
            print(f"❌ Error downloading images: {e}")
    
    def format_json(self):
        try:
            df = pd.read_csv(self.csv_file)
            grouped_data = df.groupby("caption").agg(
                image_URLs=("image_url", lambda x: ", ".join(x)),
                image_paths=("image_name", lambda x: ", ".join([f"{self.image_dir}/{name}" for name in x]))
            ).reset_index()
            grouped_data.rename(columns={"caption": "post_contents"}, inplace=True)
            json_data = grouped_data.to_dict(orient="records")
            with open(self.output_json, "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, indent=4, ensure_ascii=False)
            print(f"✅ JSON saved as {self.output_json}")
        except Exception as e:
            print(f"❌ Error formatting JSON: {e}")
