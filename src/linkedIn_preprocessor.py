import json
from text_preprocessing import TextPreprocessing

# Initialize the text preprocessing class
preprocessor = TextPreprocessing()

json_data  = '../data/linkedin_data/post_data.json'
output_data = '../data/clean_data/linkedin_data/preprocessed_data.json'

try:
    posts_data = json.load(open(json_data))

    preprocessed_data = []

    # Process each post
    for post in posts_data:
        cleaned_heading = preprocessor.clean_text(post.get("Post_Heading", ""))
        cleaned_content = preprocessor.clean_text(post.get("Post_Content", ""))

        processed_post = {
            "Cleaned_Heading": cleaned_heading,
            "Cleaned_Content": cleaned_content,
            "Original_Hashtags": post.get("Hashtags", ""),
            "Image_URLs": post.get("Image URLs", ""),
            "Image_Paths": post.get("Image Paths", "")
        }
        preprocessed_data.append(processed_post)

    # Save preprocessed data to preprocessed_data.json
    with open(output_data, 'w') as outfile:
        json.dump(preprocessed_data, outfile, indent=4)

    print(f"Preprocessed data saved to {output_data}")

except FileNotFoundError:
    print(f"Error: The file {json_data} was not found.")

except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from {json_data}.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

