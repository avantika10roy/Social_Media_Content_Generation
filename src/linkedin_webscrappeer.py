from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
from bs4 import BeautifulSoup
import re
import os 
import requests

class LinkedInScraper:
    def __init__(self, username, password, profile_url):
        self.username = username
        self.password = password
        self.profile_url = profile_url
        self.chrome_options = Options()
        self.chrome_options.add_argument("--start-maximized")
        self.chrome_options.add_argument("--disable-notifications")
        self.service = Service("/usr/local/bin/chromedriver")
        self.driver = webdriver.Chrome(service=self.service, options=self.chrome_options)
        self.wait = WebDriverWait(self.driver, 10)

    def login(self):
        self.driver.get("https://www.linkedin.com/login")
        self.wait.until(EC.presence_of_element_located((By.ID, "username")))

        email_field = self.driver.find_element(By.ID, "username")
        password_field = self.driver.find_element(By.ID, "password")

        email_field.send_keys(self.username)
        password_field.send_keys(self.password)
        password_field.send_keys(Keys.RETURN)

        self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "feed-shared-update-v2")))

    def scroll_down(self):
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        posts_count = len(self.driver.find_elements(By.CLASS_NAME, "feed-shared-update-v2"))
        previous_count = 0
        no_change_count = 0

        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)

            try:
                self.wait.until(lambda driver: len(driver.find_elements(By.CLASS_NAME, "feed-shared-update-v2")) > posts_count)
            except:
                if posts_count == previous_count:
                    no_change_count += 1
                else:
                    no_change_count = 0

                if no_change_count >= 3:
                    break

            previous_count = posts_count
            posts_count = len(self.driver.find_elements(By.CLASS_NAME, "feed-shared-update-v2"))

            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    @staticmethod
    def extract_hashtags(text):
        hashtags = re.findall(r'(?:hashtag)?#\w+', text)
        return list(set([tag.replace('hashtag#', '#') for tag in hashtags]))

    @staticmethod
    def extract_emojis(text):
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            "\U0001F680-\U0001F6FF"  # Transport & Map
            "\U0001F700-\U0001F77F"  # Alchemical Symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols, etc.
            "\U0001FA70-\U0001FAFF"  # More miscellaneous symbols
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # Enclosed characters
            "]+", flags=re.UNICODE)

        return emoji_pattern.findall(text)

    @staticmethod
    def clean_text(text):
        text = re.sub(r'hashtag#\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\.{3,}', '.', text)
        text = ' '.join(text.split())
        return text.strip()

    @staticmethod
    def split_at_first_delimiter(text):
        match = re.search(r'[.!?]', text)

        if match:
            split_index = match.start() + 1
            heading = text[:split_index].strip()
            content = text[split_index:].strip()

            content = re.sub(r'^[.!?\s]+', '', content)

            return heading, content
        else:
            return text.strip(), ""

    @staticmethod
    def create_image_folder():
        folder_path = "../data/linkedin/linkedin_images"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def is_valid_image(img_element):
        src = img_element.get('src', '').lower()
        invalid_indicators = ['logo', 'profile', 'company', 'brand', 'avatar', '8fz8rainn3wh49ad6ef9gotj1']
        if any(indicator in src for indicator in invalid_indicators):
            return False
        return True

    def download_image(self, image_url, post_id):
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                image_name = f"{post_id}_{image_url.split('/')[-1].split('?')[0]}.png"
                image_path = os.path.join("../data/linkedin/linkedin_images", image_name)

                with open(image_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"Downloaded: {image_name}")
                return image_path
            else:
                print(f"Failed to download image from {image_url}")
                return None
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

    def scrape_posts(self):
        self.driver.get(self.profile_url)
        self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "feed-shared-update-v2")))
        self.scroll_down()

        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        posts = soup.find_all('div', {'class': ['feed-shared-update-v2', 'update-components-text']})
        scraped_data = []

        self.create_image_folder()

        for post_id, post in enumerate(posts, start=1):
            try:
                content_element = post.find('span', {'class': 'break-words'})
                if not content_element:
                    continue

                full_text = content_element.get_text(strip=True)

                hashtags = self.extract_hashtags(full_text)
                emojis = self.extract_emojis(full_text)
                cleaned_text = self.clean_text(full_text)
                heading, content = self.split_at_first_delimiter(cleaned_text)

                image_elements = post.find_all('img', {'class': 'ivm-view-attr__img--centered'})
                valid_images = [img for img in image_elements if self.is_valid_image(img)]
                image_urls = list(set([img['src'] for img in valid_images if img.get('src')]))

                if not image_urls:
                    continue

                image_paths = []
                for image_url in image_urls:
                    image_path = self.download_image(image_url, post_id)
                    if image_path:
                        image_paths.append(image_path)

                scraped_data.append({
                    "Post Caption/Heading": heading,
                    "Post Content": content if content else "No content",
                    "Hashtags": ', '.join(hashtags) if hashtags else "No hashtags",
                    "Emojis": ', '.join(emojis) if emojis else "No emojis",
                    "Image URLs": ', '.join(image_urls),
                    "Image Paths": ', '.join(image_paths)
                })

            except Exception as e:
                print(f"Error extracting post: {e}")
                continue

        return scraped_data

    def run(self):
        try:
            self.login()
            scraped_data = self.scrape_posts()

            df = pd.DataFrame(scraped_data)
            df.to_csv('../data/linkedin/linkedin_posts.csv', index=False)
            print("Data saved successfully to linkedin_posts.csv")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            self.driver.quit()

if __name__ == "__main__":
    scraper = LinkedInScraper(
        username="@itobuz.com",
        password="Avantika@2930",
        profile_url="https://www.linkedin.com/company/itobuz-technologies-pvt-ltd/posts/?feedView=all"
    )
    scraper.run()
