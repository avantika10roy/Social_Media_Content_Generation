import os
import base64
import streamlit as st

st.set_page_config(page_title = "Image Generation Documentation", 
                   page_icon  = "📚", 
                   layout     = "wide")

st.markdown(
    """
    <style>
        /* Set dark background */
        body {
            background-color: #121212;
            color: #ffffff;
        }

        /* Change font to match logo style */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }

        /* Customize title */
        .title {
            color: #d4af37;
            font-weight: bold;
            font-size: 36px;
        }

        /* Style text */
        .highlight {
            color: #c9a86a;
            font-weight: bold;
        }

        /* Beautify the content box */
        .content-box {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
        }
    </style>
    """,
    unsafe_allow_html=True
)


logo_path = "/Users/it012303/Project/social_media_content_generation/web/assets/logo.png"  

with open(logo_path, "rb") as img_file:
    encoded_logo = base64.b64encode(img_file.read()).decode()

st.markdown(
    f"""
    <style>
        .logo-container img {{
            height: 200px !important;  /* Adjust height as needed */
            width: 1000;  /* Maintain aspect ratio */
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{encoded_logo}" alt="BrandSync AI Logo">
    </div>
    """,
    unsafe_allow_html=True
) 


# Hide only the default Streamlit sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebarNav"] {
            display: none !important;
        }
    </style>
""", 
unsafe_allow_html=True
)


# Get the absolute path of the current script (pages/image_generation_doc.py)
current_dir = os.path.dirname(__file__)

# Construct the correct path to the Markdown file
md_file_path = os.path.join(current_dir, "image_generation_doc.md")

# Read and display the Markdown content
try:
    with open(md_file_path, "r", encoding="utf-8") as file:
        markdown_content = file.read()
    st.markdown(markdown_content)
except FileNotFoundError:
    st.error("⚠️ Error: The file `image_generation_doc.md` was not found. Please check the file path.")
    
st.markdown("<hr>", unsafe_allow_html = True) 

st.markdown("""
<p style='text-align: center; font-weight: bold; opacity: 0.6;'>
    © 2025 Made by Itobuz Technologies | All Rights Reserved
</p>
""", unsafe_allow_html = True)