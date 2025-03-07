## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

import os
import sys
import time
import json
import base64
import requests
import pyperclip
import streamlit as st
from time import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Append the root directory to sys.path
sys.path.append(ROOT_DIR)

from config.config import Config

st.set_page_config(page_title = "Text Generation", 
                   page_icon  = "✍️", 
                   layout     = "centered")


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
    unsafe_allow_html = True
) 


st.markdown("""
    <style>
        section[data-testid="stSidebarNav"] {
            display: none !important;
        }
    </style>
""", 
unsafe_allow_html = True
)

# Title & Description
st.markdown("<h1 class='title'>✍️ AI-Powered Text Generation</h1>", 
            unsafe_allow_html = True)

st.markdown("<h3 class='subtitle'>📢 Create Social Media Posts Effortlessly!</h3>", 
            unsafe_allow_html = True)

# Centered Container
with st.container():
    st.markdown("<div class = 'container'>", unsafe_allow_html = True)

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
            
        if not extra_details:
            extra_details = ""
            
        if missing_fields:
            st.warning(f"Please fill in the required fields: {', '.join(missing_fields)}")      
            
            focus_field                         = missing_fields[0]
            
        else:
            
            # progress_bar                        = st.progress(0)
            
            # for percent_complete in range(100):
            #     time.sleep(0.3)
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

            start_time                          = time()
            response                            = requests.post(Config.TEXT_GENERATION_API, json = payload)
            end_time                            = time()

            if response.status_code == 200:
                response_json                   = response.json()
                st.session_state.generated_text = response_json.get("generated_text", "")
                
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
            
    space, text_area_space, space = st.columns([1, 5, 1])
    
    with text_area_space:
        
        st.text(st.session_state.generated_text)
        
    text_button_download, text_button_copy, text_button_regenerate, space = st.columns([1, 1, 1, 10])
    
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
                
                payload                     = { "company_name"     : company_name,
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
                    st.session_state.generated_text = response_json.get("generated_text", "")
                    st.rerun()


st.markdown("<hr>", unsafe_allow_html = True) 

st.markdown("""
<p style='text-align: center; font-weight: bold; opacity: 0.6;'>
    © 2025 Made by Itobuz Technologies | All Rights Reserved
</p>
""", unsafe_allow_html = True)
