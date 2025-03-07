## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

import io
import os
import sys
import json
import time
import base64
import requests
import textwrap
import pyperclip
from PIL import Image 
from io import BytesIO
import streamlit as st
from PIL import ImageDraw 
from PIL import ImageFont


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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
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
                                    "Personal Brands & Entrepreneurs",
                                    "Custom"
                                    ]
                                   )
    
    if target_audience == "Custom":
        
        target_audience = st.text_input("Custom Target Audience", placeholder = "Enter Your Custom Target Audience")
    
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
                                                   "occasion"         : occasion,
                                                   "topic"            : topic,
                                                   "brief"            : brief,
                                                   "extra_details"    : extra_details,
                                                   "platform"         : platform,
                                                   "tone"             : tone,
                                                   "target_audience"  : target_audience,
                                                   }
            
            start_time                          = time.time()
            
            response                            = requests.post(Config.TEXT_GENERATION_API, json = payload)
            
            end_time                            = time.time()
             
            if response.status_code == 200:
                response_json                   = response.json()
                st.session_state.generated_text = response_json.get("Generated_Text", "")
                
                ## ----- SAVING PAYLOAD AND RESPONSES INTO A JSON FILE -----
                
                os.makedirs(os.path.dirname(Config.LLM_RESPONSE_JSON_FILE_PATH), exist_ok=True)
                
                if os.path.exists(Config.LLM_RESPONSE_JSON_FILE_PATH):
                    with open(Config.LLM_RESPONSE_JSON_FILE_PATH, "r") as f:
                        
                        try:
                            data = json.load(f)
                        
                        except json.JSONDecodeError:
                            data = []
                
                else:
                    data         = []
                
                new_entry        = {"input": payload,"output": st.session_state.generated_text, 'time_taken':end_time-start_time}

                data.append(new_entry)

                with open(Config.LLM_RESPONSE_JSON_FILE_PATH, "w", encoding = "utf-8") as f:
                    json.dump(data, f, indent = 4, ensure_ascii = False)
            

    st.text(st.session_state.generated_text)
        
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
    
    main_content, right_sidebar = st.columns([5, 3])
    
    with main_content:
    
        st.markdown("<div class='container'>", unsafe_allow_html = True)
        
        st.markdown("<h3 class='subtitle'>Generate your Image Here</h3>", unsafe_allow_html = True)

        with st.container():
            st.markdown("<div class='container'>", unsafe_allow_html=True)

            prompt                  = st.text_area("**Enter image description** :red[*] ", 
                                                    placeholder = "A futuristic city at night...")
            
            negative_prompt_options = ["blurry", "low resolution", "distorted", "uncanny valley", "unnatural lighting",  
                                       "grainy", "noisy", "compression artifacts", "pixelated", "watermark",  "text overlay", 
                                       "extra limbs", "extra fingers", "deformed hands", "asymmetrical features",  "overexposed", 
                                       "underexposed", "washed out colors", "oversaturated", "harsh shadows",  "bad framing", 
                                       "unbalanced composition", "distracting background", "awkward perspective",  
                                       "unnatural depth of field", "glitched", "warped faces", "misaligned eyes", 
                                       "mutated anatomy"
                                       ]
            
            negative_prompt         = st.multiselect(label = "**Select your Negative Prompt Options** :red[*]", options = negative_prompt_options)
            
            mask_position           = st.selectbox(label = "**Text Position**", options = ["left", "right"])
            
            logo_presence           = st.selectbox(label = "**Logo Presence** :red[*]", options = ["No", "Yes"])
            
            if logo_presence == "Yes":
                
                logo_area           = st.selectbox(label = "**Logo Position**", options = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]) 
            
            image_size              = st.selectbox(label = "**Image Size**", options = ["1024x1024", "768x768", "512x512", "1024x768", "768x1024", "768x512", "512x768"])
            
            inference_steps         = st.number_input(label = "**Generation Steps**",min_value = 10, max_value = 100, step = 10)
            
            prior_guidance_scale    = st.number_input(label = "**Text Influence**", min_value = 1.0, max_value = 5.0)
            
            background_color        = st.color_picker(label = "Background Color", value =  "#FFFFFF")

            st.text("")   
                
            if "generated_image" not in st.session_state:
                st.session_state.generated_image = None
                st.session_state.image_bytes     = None 
            
            if st.button("Generate Image"):
                
                missing_fields = []
                
                if not prompt:
                    missing_fields.append("Prompt")
                    
                if not negative_prompt:
                    missing_fields.append("Negative Prompt")
                    
                if not logo_presence:
                    missing_fields.append("Logo Presence")

                if missing_fields:
                    st.warning(f"Please fill in the required fields: {', '.join(missing_fields)}")      
                    focus_field               = missing_fields[0]    
                    st.session_state["focus"] = focus_field
                
                else:
                    
                    # progress_bar = st.progress(0)
                    
                    # for percent_complete in range(100):
                    #     time.sleep(0.05)
                    #     progress_bar.progress(percent_complete + 1)
                    
                    width, height = map(int, image_size.split("x"))
                    
                    payload                              = {"prompt"                : prompt,
                                                            "negative_prompt"       : ','.join(negative_prompt),
                                                            "mask_position"         : mask_position,
                                                            "height"                : height,
                                                            "width"                 : width,
                                                            "base_bg_color"         : background_color,
                                                            "inference_step"        : inference_steps,
                                                            "prior_guidance_scale"  : prior_guidance_scale
                                                            } 
                        

                    start_time                           = time.time()
                    
                    response                             = requests.get(Config.IMAGE_GENERATION_API, params = payload)
                    
                    end_time                             = time.time()

                    if response.status_code == 200:

                        st.session_state.image_bytes     = BytesIO(response.content)
                        st.session_state.generated_image = Image.open(st.session_state.image_bytes)
                        
                        ## ----- SAVING PAYLOAD AND RESPONSES INTO A JSON FILE -----
                    
                        os.makedirs(os.path.dirname(Config.IMAGE_RESPONSE_JSON_FILE_PATH), exist_ok = True)
                        
                        if os.path.exists(Config.IMAGE_RESPONSE_JSON_FILE_PATH):
                            with open(Config.IMAGE_RESPONSE_JSON_FILE_PATH, "r") as f:
                                try:
                                    data = json.load(f)
                                
                                except json.JSONDecodeError:
                                    data = []
                        else:
                            data         = []
                        
                        new_entry        = {"input": payload, 'time_taken':end_time-start_time}

                        data.append(new_entry)

                        with open(Config.IMAGE_RESPONSE_JSON_FILE_PATH, "w", encoding = "utf-8") as f:
                            json.dump(data, f, indent = 4, ensure_ascii = False)

                
                # st.session_state.generated_image = Image.open("/Users/it012303/Project/social_media_content_generation/data/MainAfter.jpg")
            
            
            if st.session_state.generated_image is not None:
                
                with right_sidebar:
                    
                    st.markdown("<h5 class='subtitle'>Edit Options</h5>", unsafe_allow_html = True)
                    
                    # Title Inputs
                    title                = st.text_input("Title:", placeholder = "Elegant Timepieces")
                    title_font           = st.selectbox("Title Font:", font_options, index = 0)
                    title_size           = st.number_input("Title Font Size", value = 60)
                    title_x              = st.number_input("Title X Position:", min_value = 0, max_value = 1000, value = 30)
                    title_y              = st.number_input("Title Y Position:", min_value = 0, max_value = 1000, value = 130)
                    title_font_color     = st.color_picker(label = "Title Font Color", value = "#000000")
                    title_wrap           = st.slider("Title Wrap Offset", min_value = 1, max_value = 1000, value = 400)
                    
                    subtitle             = st.text_input("Subtitle:", placeholder = "Experience the Art of Watchmaking")
                    subtitle_font        = st.selectbox("Subtitle Font:", font_options, index = 0)
                    subtitle_size        = st.number_input("Subtitle Font Size", value = 30)
                    subtitle_x           = st.number_input("Subtitle X Position:", min_value = 0, max_value = 1000, value = 30)
                    subtitle_y           = st.number_input("Subtitle Y Position:", min_value = 0, max_value = 1000, value = 300)
                    subtitle_font_color  = st.color_picker(label = "Subtitle Font Color", value = "#000000")
                    subtitle_wrap         = st.slider("Subtitle Wrap Offset", min_value = 1, max_value = 1000, value = 400)
                    
                    paragraph            = st.text_area("Paragraph:", placeholder = "Our watches are crafted with precision, blending innovation and tradition. Explore timeless elegance.")
                    paragraph_font       = st.selectbox("Paragraph Font:", font_options, index = 0)
                    paragraph_size       = st.number_input("Paragraph Font Size", value = 20)
                    paragraph_x          = st.number_input("Paragraph X Position:", min_value = 0, max_value = 1000, value = 30)
                    paragraph_y          = st.number_input("Paragraph Y Position:", min_value = 0, max_value = 1000, value = 770)
                    paragraph_font_color = st.color_picker(label = "Paragraph Font Color", value = "#000000")
                    paragraph_wrap       = st.slider("Paragraph Wrap Offset", min_value = 1, max_value = 1000, value = 400)                            
                    # wrap                 = st.slider("Wrap Offset", min_value = 1, max_value = 1000, value = 400)
                
                text_image               = st.session_state.generated_image.copy()
                scaling_fac              = 1.2
                
                BASE_DIR                 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                IMAGE_LOGO_PATH          = os.path.join(BASE_DIR, "assets", "itobuz-logo.png")
                
                logo                     = Image.open(IMAGE_LOGO_PATH).convert('RGBA')
                resized_logo             = logo.resize((int(150 * scaling_fac),int(40 * scaling_fac)))
                
                logo_alpha               = resized_logo.split()[3]
                
                if logo_area == "Top-Left":
                    logo_position        = (30,30)
                    
                elif logo_area == "Top-Right":
                    logo_position        = (820,30)
                    
                elif logo_area == "Bottom-Left":
                    logo_position        = (30,950)
                    
                elif logo_area == "Bottom-Right":
                    logo_position        = (820,950)
                
                text_image.paste(resized_logo, logo_position, mask = logo_alpha)
                
                draw                     = ImageDraw.Draw(text_image)

                title_font_path          = font_families.get(title_font, None)
                subtitle_font_path       = font_families.get(subtitle_font, None)
                paragraph_font_path      = font_families.get(paragraph_font, None)

                font_title               = ImageFont.truetype(title_font_path, title_size) if title_font_path else ImageFont.load_default()
                font_subtitle            = ImageFont.truetype(subtitle_font_path, subtitle_size) if subtitle_font_path else ImageFont.load_default()
                font_paragraph           = ImageFont.truetype(paragraph_font_path, paragraph_size) if paragraph_font_path else ImageFont.load_default()
                
                char_width_title         = font_title.getbbox("A")[2] - font_title.getbbox("A")[0]  # Width of 'A'
                char_width_subtitle      = font_subtitle.getbbox("A")[2] - font_subtitle.getbbox("A")[0]
                char_width_paragraph     = font_paragraph.getbbox("A")[2] - font_paragraph.getbbox("A")[0]

                title_wrap_width         = title_wrap // char_width_title
                subtitle_wrap_width      = subtitle_wrap // char_width_subtitle
                paragraph_wrap_width     = paragraph_wrap // char_width_paragraph

                title_wrapped            = textwrap.fill(title, width = title_wrap_width)
                subtitle_wrapped         = textwrap.fill(subtitle, width = subtitle_wrap_width)
                paragraph_wrapped        = textwrap.fill(paragraph, width = paragraph_wrap_width)

                draw.text((title_x, title_y), title_wrapped, fill = title_font_color, font = font_title)
                draw.text((subtitle_x, subtitle_y), subtitle_wrapped, fill = subtitle_font_color, font = font_subtitle)
                draw.text((paragraph_x, paragraph_y), paragraph_wrapped, fill = paragraph_font_color, font = font_paragraph)
                
                image_bytes_io = io.BytesIO()
                text_image.save(image_bytes_io, format="JPEG")
                image_bytes_io.seek(0)


                st.image(text_image, use_container_width = True)

                            
            image_button_download, image_button_regenerate, image_edit_button, space = st.columns([1, 1, 1, 7])
            
            if st.session_state.generated_image:
                
                with image_button_download:
                    
                    st.download_button(label      = "📥",
                                    data       = image_bytes_io,
                                    file_name  = "generated_image.jpg",
                                    mime       = "image/jpeg")
                    
                with image_button_regenerate:
                    
                    if st.button(label = "🔄", key = "image_regenerate"):
                        
                        width, height = map(int, image_size.split("x"))
                        
                        payload                             = {"prompt"                : prompt,
                                                            "negative_prompt"       : ','.join(negative_prompt),
                                                            "mask_position"         : mask_position,
                                                            "height"                : height,
                                                            "width"                 : width,
                                                            "base_bg_color"         : background_color,
                                                            "inference_step"        : inference_steps,
                                                            "prior_guidance_scale"  : prior_guidance_scale
                                                            } 
                        

                        response                             = requests.get(Config.IMAGE_GENERATION_API, params = payload)

                        if response.status_code == 200:

                            st.session_state.image_bytes     = BytesIO(response.content)
                            st.session_state.generated_image = Image.open(st.session_state.image_bytes)
                            st.rerun()
    

            
st.markdown("<hr>", unsafe_allow_html = True) 

st.markdown("""
<p style='text-align: center; font-weight: bold; opacity: 0.6;'>
    © 2025 Made by Itobuz Technologies | All Rights Reserved
</p>
""", unsafe_allow_html = True)