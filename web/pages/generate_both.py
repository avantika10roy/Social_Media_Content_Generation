## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

import os
import sys
import time
import base64
import requests
import pyperclip
from PIL import Image
from io import BytesIO
import streamlit as st

# Get the absolute path of the root directory (two levels up from the current script)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Append the root directory to sys.path
sys.path.append(ROOT_DIR)

from config.config import Config

# Set page configuration
st.set_page_config(page_title = "Generating Both", 
                   page_icon  = "✨", 
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

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

with open(LOGO_PATH, "rb") as img_file:
    encoded_logo = base64.b64encode(img_file.read()).decode()

st.markdown(
    f"""
    <style>
        .logo-container img {{
            height: 180px !important;  /* Adjust height as needed */
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


st.markdown("""
    <style>
        section[data-testid="stSidebarNav"] {
            display: none !important;
        }
    </style>
""", 
unsafe_allow_html=True
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up from 'pages' to 'web'
FONT_DIR = os.path.join(BASE_DIR, 'fonts')

# Load available fonts from the directory
def get_font_families():
    fonts = [f for f in os.listdir(FONT_DIR) if f.endswith(".ttf")]
    return {font.split(".")[0]: os.path.join(FONT_DIR, font) for font in fonts}

# Get available fonts
font_families = get_font_families()
font_options  = list(font_families.keys())

st.title("Generate Both")

## ----- TEXT GENERATION SECTION -----

with st.expander("Text Generation Section", expanded = False):
    
    st.markdown("<div class = 'container'>", unsafe_allow_html = True)
    
    st.markdown("<h3 class='subtitle'>Generate your Text Here</h3>", unsafe_allow_html = True)

    company_name    = st.text_input("**Enter Your Company Name** :red[*]", 
                                    placeholder = "E.g. Itobuz Technologies")
    
    purpose         = st.radio("**For Which Purpose Do You Want To Generate This Post?** :red[*]", 
                               ["On A Specific Occasion", 
                                "On A Specific Topic"])

    if (purpose == "On A Specific Occasion"):
        
        occasion    = st.selectbox("**Select The Occasion For The Social Media Post** :red[*]", 
                                   ["Business & Professional Occasion", 
                                    "Marketing & Sales-Related Occasion", 
                                    "Social & Cultural Occasion", 
                                    "Industry-Specific Days", 
                                    "Personal & Community Engagement Occasions", 
                                    "Trend-Based & Fun Occasions"
                                    ]
                                   )
        
        topic        = ""
    
    elif (purpose == "On A Specific Topic"):
        
        topic = st.selectbox("**Select The Topic For The Social Media Post** :red[*]", 
                             ["Product & Feature-Based Topic", 
                              "Marketing & Branding Insights", 
                              "Engagement & Community", 
                              "Trend-Based Content", 
                              "Thought Leadership & Industry Trend", 
                              "Customer Success Stories & Testimonials", 
                              "Special Day & Holiday"
                              ]
                             )
        
        occasion    = ""

    brief           = st.text_area("**Please Provide A Brief Description For This Post** :red[*]", 
                                   placeholder = "Enter Brief for the Event")
    
    extra_details   = st.text_area("**Do You Want To Add Some Brand Specific Information, Keywords, Hashtags, Emojis, or Other Details?**", 
                                   placeholder = "Enter your Extra Details Here")
    
    platform        = st.selectbox("**Select Social Media Platform** :red[*]", 
                                   ["LinkedIn", 
                                    "Instagram", 
                                    "Facebook"
                                    ]
                                   )
    
    tone            = st.selectbox("**Select Tone & Style** :red[*]", 
                                   ["Formal", 
                                    "Casual", 
                                    "Promotional", 
                                    "Inspirational", 
                                    "Motivational", 
                                    "Informative", 
                                    "Educational", 
                                    "Humorous", 
                                    "Witty", 
                                    "Storytelling", 
                                    "Narrative", 
                                    "Authoritative", 
                                    "Thought Leadership"
                                    ]
                                   )
    
    target_audience = st.selectbox("**Select The Target Audience Type For This Post** :red[*]", 
                                   ["Small & Medium Businesses", 
                                    "Social Media Managers & Marketers", 
                                    "Influencers & Content Creators", 
                                    "Corporate & HR Teams", 
                                    "Agencies & Freelancers", 
                                    "Personal Brands & Entrepreneurs"
                                    ]
                                   )
    
    if "generated_text" not in st.session_state:
            st.session_state.generated_text     = ""

    if st.button("Generate Text"):
        missing_fields                          = []
        
        if not company_name:
            missing_fields.append("Company Name")
        
        if not purpose:
            missing_fields.append("Purpose")
            
        if not brief:
            missing_fields.append("Brief")
            
        if missing_fields:
            st.warning(f"Please fill in the required fields: {', '.join(missing_fields)}")      
            
            focus_field                         = missing_fields[0]
            
        else:
            
            # progress_bar                        = st.progress(0)
            
            # for percent_complete in range(100):
            #     time.sleep(0.05)
            #     progress_bar.progress(percent_complete + 1)
                
            payload                             = {"company_name"     : company_name,
                                                   "purpose"          : purpose,
                                                   "occasion"         : occasion,
                                                   "topic"            : topic,
                                                   "brief"            : brief,
                                                   "extra_details"    : extra_details,
                                                   "platform"         : platform,
                                                   "tone"             : tone,
                                                   "target_audience"  : target_audience,
                                                   }

            response                            = requests.post(Config.TEXT_GENERATION_API, json = payload)

            if response.status_code == 200:
                response_json                   = response.json()
                st.session_state.generated_text = response_json.get("Generated_Text", "")

    st.write(st.session_state.generated_text)
        
    text_button_download, text_button_copy, text_button_regenerate, space = st.columns([1.7, 1.7, 1.7, 8])
    
    with text_button_download:

        if st.session_state.generated_text:
            st.download_button(label      = "📥", 
                            data       = st.session_state.generated_text,
                            file_name  = "generated_text.txt",
                            mime       = "text/plain"
                            )
    
    with text_button_copy:
        
        if st.session_state.generated_text:
            if st.button(label = "📄", key = "text_copy"):
                pyperclip.copy(st.session_state.generated_text)
                
    
    with text_button_regenerate:
        
        if st.session_state.generated_text:
            
            if st.button(label = "🔄", key = "text_regenerate"):
                
                payload                     = {"company_name"     : company_name,
                                                "purpose"          : purpose,
                                                "occasion"         : occasion,
                                                "topic"            : topic,
                                                "brief"            : brief,
                                                "extra_details"    : extra_details,
                                                "platform"         : platform,
                                                "tone"             : tone,
                                                "target_audience"  : target_audience,
                                                }
                
                response                    = requests.post(Config.TEXT_GENERATION_API, json = payload)
                
                if response.status_code == 200:
                    response_json                   = response.json()
                    st.session_state.generated_text = response_json.get("Generated_Text", "")
                    st.rerun()


## ----- IMAGE GENERATION SECTION -----
    
with st.expander("Image Generation Section", expanded = False):
    
    st.markdown("<div class='container'>", unsafe_allow_html = True)
    
    st.markdown("<h3 class='subtitle'>Generate your Image Here</h3>", unsafe_allow_html = True)

    image_prompt  = st.text_input("Enter image description:", 
                                  placeholder = "A futuristic city at night...")
    
    image_quality = st.selectbox("Select Image Quality:", 
                                 ["Low", 
                                  "Medium", 
                                  "High"
                                  ]
                                 )
    
    aspect_ratio  = st.radio("Choose Aspect Ratio:", 
                             ["Square (1:1)", 
                              "Portrait (4:5)", 
                              "Landscape (16:9)"
                              ]
                             )
    
    st.text("")   
        
    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None
        st.session_state.image_bytes     = None 
    
    if st.button("Generate Image"):
        
        missing_fields = []

        if not image_prompt: 
            missing_fields.append("Prompt of Image")
            
        if not image_quality:
            missing_fields.append("Image Quality")
            
        if not aspect_ratio:
            missing_fields.append("Aspect Ratio")

        if missing_fields:
            st.warning(f"Please fill in the required fields: {', '.join(missing_fields)}")      
            focus_field               = missing_fields[0]    
            st.session_state["focus"] = focus_field
        
        else:
            
            # progress_bar = st.progress(0)
            
            # for percent_complete in range(100):
            #     time.sleep(0.05)
            #     progress_bar.progress(percent_complete + 1)
                
            payload                              = {"image_prompt"    : image_prompt,
                                                    "image_quality"   :  image_quality,
                                                    "aspect_ratio"    : aspect_ratio,
                                                    } 

            response                             = requests.post(Config.IMAGE_GENERATION_API, json = payload)

            if response.status_code == 200:

                st.session_state.image_bytes     = BytesIO(response.content)
                st.session_state.generated_image = Image.open(st.session_state.image_bytes)
    
    generated_image_section, edit_image_section = st.columns([5,3])
    
    with generated_image_section:
    
        if st.session_state.generated_image is not None:
            
            # Text Inputs
            title           = st.text_input("Title:", "Elegant Timepieces")
            subtitle        = st.text_input("Subtitle:", "Experience the Art of Watchmaking")
            paragraph       = st.text_area("Paragraph:", 
                "Our watches are crafted with precision, blending innovation and tradition. Explore timeless elegance.")
            
            st.image(st.session_state.generated_image, caption = "Generated Image", use_container_width = True)

                    
        image_button_download, image_button_regenerate, space = st.columns([1.5, 1.5, 12])
        
        if st.session_state.generated_image:
            
            with image_button_download:
                
                st.download_button(label      = "📥",
                                data       = st.session_state.image_bytes.getvalue(),
                                file_name  = "generated_image.jpg",
                                mime       = "image/jpeg")
                
        if st.session_state.generated_image:
                
            with image_button_regenerate:
                
                if st.button(label = "🔄", key = "image_regenerate"):
                    
                    payload                              = {"image_prompt"    : image_prompt,
                                                            "image_quality"   :  image_quality,
                                                            "aspect_ratio"    : aspect_ratio,
                                                            } 

                    response                             = requests.post(Config.IMAGE_GENERATION_API, json = payload)

                    if response.status_code == 200:

                        st.session_state.image_bytes     = BytesIO(response.content)
                        st.session_state.generated_image = Image.open(st.session_state.image_bytes)
                        st.rerun()
                        
    
    with edit_image_section:
        
        if st.session_state.generated_image:
        
            
            # Font Selection
            title_font      = st.selectbox("Title Font:", font_options, index = 0)
            subtitle_font   = st.selectbox("Subtitle Font:", font_options, index = 0)
            paragraph_font  = st.selectbox("Paragraph Font:", font_options, index = 0)
            
            # Font Size Selection
            title_size      = st.slider("Title Font Size:", 5, 50, 20)
            subtitle_size   = st.slider("Subtitle Font Size:", 5, 50, 16)
            paragraph_size  = st.slider("Paragraph Font Size:", 5, 50, 12 )

            # Text Positioning
            title_y         = st.slider("Title Y Position:", 0, 500, 70)
            subtitle_y      = st.slider("Subtitle Y Position:", 0, 500, 160)
            paragraph_y     = st.slider("Paragraph Y Position:", 0, 700, 300)

            wrap            = st.slider("Wrap Offset", 0, 500, 70)

            
st.markdown("<hr>", unsafe_allow_html = True) 

st.markdown("""
<p style='text-align: center; font-weight: bold; opacity: 0.6;'>
    © 2025 Made by Itobuz Technologies | All Rights Reserved
</p>
""", unsafe_allow_html = True)