# Written By Arnab Chatterjee

import json

def merge_post_contents(linkedin_json,fb_json, insta_json,  curated_json, output_json):
    # Load JSON files
    with open(fb_json, "r", encoding="utf-8") as fb_file, \
         open(insta_json, "r", encoding="utf-8") as insta_file, \
         open(linkedin_json, "r", encoding="utf-8") as linkedin_file, \
         open(curated_json, "r", encoding="utf-8") as curated_file:
        
        fb_data = json.load(fb_file)
        insta_data = json.load(insta_file)
        linkedin_data = json.load(linkedin_file)
        curated_data = json.load(curated_file)

    # Extract post_contents from each JSON
    fb_posts = [post.get("post_contents", "") for post in fb_data]
    insta_posts = [post.get("post_heading", "") + post.get("post_content", "") for post in insta_data]
    linkedin_posts = [post.get("post_contents", "") for post in linkedin_data]

    # Combine all post_contents
    all_posts =  linkedin_posts + fb_posts + insta_posts

    # Add post_contents to curated JSON
    for i, post in enumerate(curated_data):
        if i < len(all_posts):
            post["raw_content"] = all_posts[i]  # Assign content from the list

    # Save the updated curated JSON to a new file
    with open(output_json, "w", encoding="utf-8") as output_file:
        json.dump(curated_data, output_file, indent=4, ensure_ascii=False)

    print(f"Updated curated JSON saved as '{output_json}' with {len(curated_data)} posts.")



