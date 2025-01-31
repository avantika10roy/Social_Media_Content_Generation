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

# LinkedIn Credentials
LINKEDIN_USERNAME = "subhas@itobuz.com"
LINKEDIN_PASSWORD = "Itobuz@2004"
PROFILE_URL = "https://www.linkedin.com/company/itobuz-technologies-pvt-ltd/posts/?feedView=all"

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-notifications")

# Set up the WebDriver
service = Service("/usr/local/bin/chromedriver")
driver = webdriver.Chrome(service=service, options=chrome_options)
wait = WebDriverWait(driver, 10)

def linkedin_login():
    driver.get("https://www.linkedin.com/login")
    wait.until(EC.presence_of_element_located((By.ID, "username")))
    
    email_field = driver.find_element(By.ID, "username")
    password_field = driver.find_element(By.ID, "password")
    
    email_field.send_keys(LINKEDIN_USERNAME)
    password_field.send_keys(LINKEDIN_PASSWORD)
    password_field.send_keys(Keys.RETURN)
    
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "feed-shared-update-v2")))

def scroll_down():
    last_height = driver.execute_script("return document.body.scrollHeight")
    posts_count = len(driver.find_elements(By.CLASS_NAME, "feed-shared-update-v2"))
    previous_count = 0
    no_change_count = 0
    
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        
        try:
            wait.until(lambda driver: len(driver.find_elements(By.CLASS_NAME, "feed-shared-update-v2")) > posts_count)
        except:
            if posts_count == previous_count:
                no_change_count += 1
            else:
                no_change_count = 0
                
            if no_change_count >= 3:
                break
        
        previous_count = posts_count
        posts_count = len(driver.find_elements(By.CLASS_NAME, "feed-shared-update-v2"))
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def extract_hashtags(text):
    hashtags = re.findall(r'(?:hashtag)?#\w+', text)
    cleaned_hashtags = list(set([
        tag.replace('hashtag#', '#') for tag in hashtags
    ]))
    return cleaned_hashtags

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
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.findall(text)

def clean_text(text):
    text = re.sub(r'hashtag#\w+', '', text)  # Remove 'hashtag#' prefix
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\.{3,}', '.', text)  # Replace multiple dots
    text = ' '.join(text.split())  # Remove extra whitespace
    return text.strip()

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

def create_image_folder():
    folder_path = "linkedin_images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def is_logo_image(img_element):
    """Check if the image is a logo based on various attributes"""
    # Check alt text for logo indicators
    alt_text = img_element.get('alt', '').lower()
    logo_indicators = ['logo', 'profile', 'company', 'brand', 'avatar']
    if any(indicator in alt_text for indicator in logo_indicators):
        return True
    
    # Check parent classes for logo indicators
    parent = img_element.parent
    for _ in range(3):  # Check up to 3 levels of parent elements
        if parent and hasattr(parent, 'get'):
            classes = parent.get('class', [])
            if isinstance(classes, str):
                classes = [classes]
            if any(indicator in ' '.join(str(c) for c in classes).lower() 
                  for indicator in logo_indicators):
                return True
            parent = parent.parent
        else:
            break
    
    # Check image URL for logo indicators
    src = img_element.get('src', '').lower()
    if any(indicator in src for indicator in logo_indicators):
        return True
    
    return False

def download_image(image_url, post_id):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            image_name = f"{post_id}_{image_url.split('/')[-1].split('?')[0]}.png"
            image_path = os.path.join("linkedin_images", image_name)
            
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

def scrape_posts():
    driver.get(PROFILE_URL)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "feed-shared-update-v2")))
    scroll_down()
    
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    posts = soup.find_all('div', {'class': ['feed-shared-update-v2', 'update-components-text']})
    scraped_data = []

    create_image_folder()

    for post_id, post in enumerate(posts, start=1):
        try:
            content_element = post.find('span', {'class': 'break-words'})
            if not content_element:
                continue

            full_text = content_element.get_text(strip=True)
            
            hashtags = extract_hashtags(full_text)
            emojis = extract_emojis(full_text)
            cleaned_text = clean_text(full_text)
            heading, content = split_at_first_delimiter(cleaned_text)

            # Filter out logo images
            image_elements = post.find_all('img', {'class': 'ivm-view-attr__img--centered'})
            non_logo_images = [img for img in image_elements if not is_logo_image(img)]
            image_urls = list(set([img['src'] for img in non_logo_images if img.get('src')]))

            if not image_urls:
                continue

            image_paths = []
            for image_url in image_urls:
                image_path = download_image(image_url, post_id)
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

def main():
    try:
        linkedin_login()
        scraped_data = scrape_posts()
        
        # Save to CSV
        df = pd.DataFrame(scraped_data)
        df.to_csv('linkedin_posts.csv', index=False)
        print("Data saved successfully to linkedin_posts.csv")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()