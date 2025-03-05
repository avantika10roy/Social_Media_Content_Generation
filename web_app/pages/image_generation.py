# Dependencies
import io
import base64
import requests
from PIL import Image
import streamlit as st

# Custom CSS for Center Alignment of the page
st.markdown(
    """
    <style>
        .container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        
        .stTextInput, .stSelectbox, .stRadio, .stButton {
            width: 50% !important;
            text-align: center;
        }
        
        .stButton>button {
            display: block;
            margin: 0 auto;
            font-size: 16px;
            font-weight: bold;
        }
        
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #d4af37;
            text-align: center;
        }
        
        .subtitle {
            font-size: 20px;
            color: #c9a86a;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Placeholder API endpoint (Replace with actual API endpoint)
image_generation_api = "https://api.example.com/generate_image"

# Title & Description
st.markdown("<h1 class='title'>🖼️ Generate AI-Powered Images</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>🎨 Create stunning AI-generated images with ease!</h3>", unsafe_allow_html=True)

# Centered Container
with st.container():
    st.markdown("<div class='container'>", unsafe_allow_html=True)

    # User Inputs 
    image_prompt  = st.text_input("Enter image description:", placeholder="A futuristic city at night...")
    image_quality = st.selectbox("Select Image Quality:", ["Low", "Medium", "High"])
    aspect_ratio  = st.radio("Choose Aspect Ratio:", ["Square (1:1)", "Portrait (4:5)", "Landscape (16:9)"])

    # Button to Generate Image
    if st.button("Generate Image"):
        
        if not image_prompt:
            st.warning("⚠️ Please enter an image description before generating.")

        else:
            # Prepare the API Request Payload
            payload  = {"prompt"       : image_prompt,
                        "quality"      : image_quality,
                        "aspect_ratio" : aspect_ratio
                       }

            response = requests.post(url  = image_generation_api, 
                                     json = payload)

            # Catch the API Response and render it
            if (response.status_code == 200):
                image_data = response.json().get("image_base64", "")
                
                if image_data:
                    # Decode and display the image
                    image_bytes = base64.b64decode(image_data)
                    image       = Image.open(io.BytesIO(image_bytes))

                    st.image(image, caption="🖼️ Generated Image", use_column_width=True)

                    # Configure the download button
                    buffer  = io.BytesIO()
                    image.save(buffer, format = "PNG")
                    byte_im = buffer.getvalue()

                    st.download_button(label     = "📥 Download Image",
                                       data      = byte_im,
                                       file_name = "generated_image.png",
                                       mime      = "image/png"
                                      )
                
                else:
                    st.error("❌ Failed to generate image. Try again.")
            
            else:
                st.error(f"❌ Error {response.status_code}: Unable to generate image.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><hr><center>🚀 Powered by BrandSync AI 🚀</center>", unsafe_allow_html=True)
