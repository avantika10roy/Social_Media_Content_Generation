# Dependencies
import time
import base64 
import requests
import importlib
import streamlit as st

# Set page configuration
st.set_page_config(page_title = "BrandSync AI", 
                   page_icon  = "assets/icon_1.png", 
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


# Display the logo in the first row
logo_path = "assets/logo.png"  
# Read and encode the image
with open(logo_path, "rb") as img_file:
    encoded_logo = base64.b64encode(img_file.read()).decode()

# Display logo with custom height
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
""", unsafe_allow_html=True)


# Create custom sidebar navigation
st.sidebar.title("Navigation")

# Define pages manually
pages               = {"Home"                       : None,
                       "Generate Text"              : "pages/text_generation.py",
                       "Generate Image"             : "pages/image_generation.py",
                       "Generate Text & Image Both" : "pages/generate_both.py",
                       "Text Generation Docs"       : "pages/text_generation_doc.md",
                       "Text Generation Guide"      : "pages/text_generation_user_guide.md",
                       "Image Generation Docs"      : "pages/image_generation_doc.md",
                       "Image Generation Guide"     : "pages/image_generation_user_guide.md",
                      }

# Define the available pages
selected_page       = st.sidebar.radio(label   = "Go to:", 
                                       options = list(pages.keys()), 
                                       index   = 0)

# Store the selection to prevent duplication
st.session_state["selected_page"] = selected_page  

# Redirect to different pages based on selection
if (pages[selected_page] is None):
    
    # Display the title in the second row
    st.markdown("<h2 class='title'>🚀 Welcome to <span class='highlight'>BrandSync AI </span>: Your AI-Powered Social Media Assistant! 🚀</h2>", unsafe_allow_html=True)

    st.markdown("<h3 class='highlight'>Struggling to create engaging and brand-aligned social media content?</h3>", unsafe_allow_html=True)


    # Homepage Description
    st.markdown('<div class="content-box">', unsafe_allow_html=True)

    st.markdown("""
    BRANDSYNC AI makes it effortless! Our AI-driven platform generates high-quality text and visuals tailored to your brand, ensuring **consistency, efficiency, and creativity**. 

    Simply provide minimal input, and let our advanced AI models craft stunning posts for platforms like **Facebook, LinkedIn, and Instagram.**  

    ✅ **AI-generated, platform-specific content**  
    ✅ **Brand-consistent tone and style**  
    ✅ **Stunning AI-generated images**  
    ✅ **Simple interface for content preview and download**  
    <br>
    ### 🚀 Supercharge your social media presence with BrandSync AI – Try it today!
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif pages[selected_page].endswith(".py"):
    exec(open(pages[selected_page]).read())

else:
    with open(pages[selected_page], "r", encoding="utf-8") as file:
        st.markdown(file.read())
