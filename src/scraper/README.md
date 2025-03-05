# Scraper Module

## Overview
The `src/scraper` module is responsible for extracting social media content from LinkedIn, Instagram, and Facebook. These scrapers help collect posts, images, and metadata, which are later processed for AI-powered content generation. The scrapers leverage Selenium, Instaloader, and Apify API to retrieve structured data.

## Scrapers

### 1. LinkedIn Scraper
#### Description
The LinkedIn Scraper automates the extraction of posts, including text content, hashtags, and images, while maintaining brand consistency.

#### Features
- Uses Selenium for web scraping.
- Supports LinkedIn login automation.
- Scrolls and fetches multiple posts.
- Extracts post text and associated images.
- Saves scraped data in JSON format.

#### Dependencies
- `selenium`
- `beautifulsoup4`
- `pandas`
- `requests`

#### Usage
```python
from src.scraper.linkedin_scraper import LinkedinScraper

scraper = LinkedinScraper(username="your_email", password="your_password", profile_url="profile_url")
scraper.setup_driver(chromedriver_path="/path/to/chromedriver")
data = scraper.linkedin_scraper()
print(data)
```

### 2. Instagram Scraper
#### Description
The Instagram Scraper fetches posts, downloads images, and extracts metadata from a public profile using the Instaloader library.

#### Features
- Uses Instaloader for efficient data extraction.
- Downloads images and saves metadata.
- Handles profile access errors.
- Stores extracted data in JSON format.

#### Dependencies
- `instaloader`
- `json`
- `time`
- `random`

#### Usage
```python
from src.scraper.instagram_scraper import InstagramDataScraper

scraper = InstagramDataScraper(user="instagram_username")
scraper.scrape()
```

### 3. Facebook Scraper
#### Description
The Facebook Scraper fetches posts from a Facebook page using the Apify API.

#### Features
- Uses Apify API for structured data extraction.
- Saves posts and image URLs.
- Converts data into structured JSON and CSV formats.

#### Dependencies
- `apify-client`
- `requests`
- `pandas`
- `json`

#### Usage
```python
from src.scraper.facebook_scraper import FacebookScraper

scraper = FacebookScraper(api_token="your_apify_token", page_url="https://facebook.com/page")
scraper.scrape_data()
```

## Directory Structure

```plaintext
src/
├── scraper/
│   ├── linkedin_scraper.py
│   ├── instagram_scraper.py
│   ├── facebook_scraper.py
│   ├── __init__.py
```

## Logging
Each scraper logs important events and errors to help in debugging. Logs are stored in the logs/ directory with separate log files for each platform.
