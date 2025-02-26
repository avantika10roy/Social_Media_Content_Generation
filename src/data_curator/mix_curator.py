import os
import re
import json
import emoji
import shutil
import pandas as pd
# from ...config.config import Config


linkedin_raw_data_path = './data/raw_data/linkedin_raw_data.json' 
facebook_raw_data_path = "./data/raw_data/facebook_raw_data.json" 
instagram_raw_data_path = "./data/raw_data/instagram_raw_data.json"
curated_data_path       = "./data/curated_data/curated_data.json"


HASHTAG_PATTERN = re.compile(r"#\w+")
HTML_TAG_PATTERN = re.compile(r'<[^>]*>')
NEWLINE_PATTERN = re.compile(r'\n')
EXTRA_DOTS_PATTERN = re.compile(r'\.{2,}')
EXTRA_SPACES_PATTERN = re.compile(r'\s+')

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
curated_data = load_json(curated_data_path)
instagram_data = load_json(instagram_raw_data_path)
linkedin_data = load_json(linkedin_raw_data_path)
facebook_data = load_json(facebook_raw_data_path)

def clean_text(text):
    removed_emoji = emoji.replace_emoji(text,replace='')  # Remove emojis
    removed_hashtags = re.sub(HASHTAG_PATTERN, '', removed_emoji)  # Remove hashtags
    removed_html_tags = re.sub(HTML_TAG_PATTERN, '', removed_hashtags)  # Remove HTML tags
    removed_n = re.sub(NEWLINE_PATTERN, '', removed_html_tags)  # Remove '\n'
    removed_dots = re.sub(EXTRA_DOTS_PATTERN, '', removed_n)  # Remove extra dots
    cleaned_text = re.sub(r'\bhashtag\b','', removed_dots)
    final_text = re.sub(EXTRA_SPACES_PATTERN, ' ', cleaned_text)

    return final_text

print(clean_text(linkedin_data[0]['post_contents']))

# for item in linkedin_data:
#     item['platform'] = "linkedin"
#     item['hashtags'] = HASHTAG_PATTERN.findall(item["post_contents"])
#     item['emoji'] = emoji.distinct_emoji_list(item["post_contents"])


# for item in instagram_data:
#     item['platform'] = "instagram"
#     item['post_contents'] = item['post_heading']+" " +item['post_content']

# for item in facebook_data:
#     item['platform'] = "facebook"
#     item['hashtags'] = HASHTAG_PATTERN.findall(item["post_contents"])

# raw_data = facebook_data + linkedin_data + instagram_data

# raw_content_dict = {(tuple(sorted(entry["hashtags"])), entry["platform"]): entry["post_contents"] for entry in raw_data}


# for post in curated_data:
#     key = (tuple(sorted(post['hashtags'])), post['platform'])
#     post['raw_post_content'] = raw_content_dict.get(key, '')

# i = 0
# for item in raw_data:
#     if len(item['hashtags']) != 0:
#         i +=1 

# print(i)
        

# def save_json(data, file_path):
#     with open(file_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)

# save_json(curated_data,"updated.json")

# -- linkedin
# for i in linkedin_data:
#     hashtags_list = i['post_contents'].apply(lambda x: HASHTAG_PATTERN.findall(x))
#     if 

