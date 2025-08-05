# DEPENDENCIES

import os
import base64
import streamlit as st

st.set_page_config(page_title = "BrandSync AI", 
                   page_icon  = "🤖", 
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


# Hide only the default Streamlit sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebarNav"] {
            display: none !important;
        }
    </style>
""", 
unsafe_allow_html = True
)


# Display the title in the second row
st.markdown("<h2 class = 'title'>🚀 Welcome to <span class='highlight'>BrandSync AI </span>: Your AI-Powered Social Media Assistant! 🚀</h2>", unsafe_allow_html=True)

st.markdown("<h3 class = 'highlight'>Struggling to create engaging and brand-aligned social media content?</h3>", unsafe_allow_html=True)


# Homepage Description
st.markdown('<div class = "content-box">', unsafe_allow_html=True)

st.markdown("""
BRANDSYNC AI makes it effortless! Our AI-driven platform generates high-quality text and visuals tailored to your brand, ensuring **consistency, efficiency, and creativity**. 

Simply provide minimal input, and let our advanced AI models craft stunning posts for platforms like **Facebook, LinkedIn, and Instagram.**  

✅ **AI-generated, platform-specific content**  
✅ **Brand-consistent tone and style**  
✅ **Stunning AI-generated images**  
✅ **Simple interface for content preview and download**  
<br>
### 🚀 Supercharge your social media presence with BrandSync AI – Try it today!
""", unsafe_allow_html = True)

st.markdown('</div>', unsafe_allow_html = True)

st.markdown("<hr>", unsafe_allow_html = True) 

st.markdown("""
<p style='text-align: center; font-weight: bold; opacity: 0.6;'>
    © 2025 Made by Itobuz Technologies | All Rights Reserved
</p>
""", unsafe_allow_html = True)

st.markdown("""
<style>
.social-buttons {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 10px;
}

.social-button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: bold;
    color: white;
    text-decoration: none;
    transition: 0.3s;
    width: 150px;
}

.linkedin { background-color: #FFFFFF; }
.facebook { background-color: #FFFFFF; }
.instagram { background-color: #FFFFFF; }

.social-button img {
    width: 20px;
    height: 20px;
}
</style>

<div class="social-buttons">
    <a class="social-button linkedin" href="https://www.linkedin.com/company/itobuz-technologies-pvt-ltd/posts/?feedView=all" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png"/> LinkedIn
    </a>
    <a class="social-button facebook" href="https://www.facebook.com/Itobuz/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/733/733547.png"/> Facebook
    </a>
    <a class="social-button instagram" href="https://www.instagram.com/itobuztechnologies/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174855.png"/> Instagram
    </a>
</div>
""", unsafe_allow_html=True)
