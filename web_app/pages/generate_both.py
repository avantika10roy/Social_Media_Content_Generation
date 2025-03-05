# Dependencies
import io
import base64
import requests
from PIL import Image
import streamlit as st

# Custom CSS for better layout
st.markdown(
    """
    <style>
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

# Placeholder API endpoints (Replace with actual API endpoints)
text_generation_api  = "https://api.example.com/generate_text"
image_generation_api = "https://api.example.com/generate_image"

# Page Title
st.markdown("<h1 class='title'>✨ AI-Powered Content Generation</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'> Create Social Media Posts & Generate AI Images</h3>", unsafe_allow_html=True)
st.markdown("<br> </br>", unsafe_allow_html=True)

# Layout for side-by-side input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Text Generation")

    # User Inputs for text generation
    purpose         = st.radio("**For Which Purpose Do You Want To Generate This Post?** :red[*]", ["On A Specific Occasion", "On A Specific Topic"])

    if (purpose == "On A Specific Occasion"):
        occasion    = st.selectbox("**Select The Occasion For The Social Media Post** :red[*]", ["Business & Professional Occasion", "Marketing & Sales-Related Occasion", "Social & Cultural Occasion", "Industry-Specific Days", "Personal & Community Engagement Occasions", "Trend-Based & Fun Occasions"])
    
    elif (purpose == "On A Specific Topic"):
        topic_brief = st.selectbox("**Select The Topic For The Social Media Post** :red[*]", ["Product & Feature-Based Topic", "Marketing & Branding Insights", "Engagement & Community", "Trend-Based Content", "Thought Leadership & Industry Trend", "Customer Success Stories & Testimonials", "Special Day & Holiday"])

    brief           = st.text_area("**Please Provide A Brief Description For This Post** :red[*]")
    extra_details   = st.text_area("**Do You Want To Add Some Brand Specific Information, Keywords, Hashtags, Emojis, or Other Details?**")
    platform        = st.selectbox("**Select Social Media Platform** :red[*]", ["LinkedIn", "Instagram", "Facebook"])
    tone            = st.selectbox("**Select Tone & Style** :red[*]", ["Formal", "Casual", "Promotional", "Inspirational", "Motivational", "Informative", "Educational", "Humorous", "Witty", "Storytelling", "Narrative", "Authoritative", "Thought Leadership"])
    target_audience = st.selectbox("**Select The Target Audience Type For This Post** :red[*]", ["Small & Medium Businesses", "Social Media Managers & Marketers", "Influencers & Content Creators", "Corporate & HR Teams", "Agencies & Freelancers", "Personal Brands & Entrepreneurs"])

    if st.button("Generate Text"):
        
        if not text_prompt:
            st.warning("⚠️ Please enter text before generating.")
        
        else:
            # Prepare payload for API request
            text_gen_payload  = {"purpose"       : purpose,
                                 "occasion"      : occasion if purpose == "On A Specific Occasion" else None,
                                 "topic"         : topic_brief if purpose == "On A Specific Topic" else None,
                                 "brief"         : brief,
                                 "extra_details" : extra_details,
                                 "platform"      : platform,
                                 "tone"          : tone,
                                 "audience"      : target_audience
                                }

            image_response    = requests.post(url  = text_generation_api, 
                                              json = text_gen_payload)

            # Capture the API Response                                 
            if (image_response.status_code == 200):
                st.success("✅ Generated Text:")
                st.write(response.json().get("generated_text", "No text generated."))
            
            else:
                st.error(f"❌ Error {response.status_code}: Unable to generate text.")

with col2:
    st.subheader("🎨 Image Generation")
    image_prompt  = st.text_input("Enter image description:", placeholder="A futuristic city at night...")
    image_quality = st.selectbox("Select Image Quality:", ["Low", "Medium", "High"])
    aspect_ratio  = st.radio("Choose Aspect Ratio:", ["Square (1:1)", "Portrait (4:5)", "Landscape (16:9)"])
    
    if st.button("Generate Image"):
        
        if not image_prompt:
            st.warning("⚠️ Please enter an image description before generating.")
        
        else:
            image_gen_payload = {"prompt"       : image_prompt, 
                                 "quality"      : image_quality, 
                                 "aspect_ratio" : aspect_ratio}

            image_response    = requests.post(url  = image_generation_url, 
                                              json = image_gen_payload)

            # Capture the API response                            
            if (image_response.status_code == 200):
                image_data = response.json().get("image_base64", "")
                
                if image_data:
                    image_bytes = base64.b64decode(image_data)
                    image       = Image.open(io.BytesIO(image_bytes))

                    # Render the image through streamlit
                    st.image(image            = image, 
                             caption          = "🖼️ Generated Image", 
                             use_column_width = True)
                    
                    # Convert the image as buffer
                    buffer      = io.BytesIO()
                    image.save(buffer, format = "PNG")
                    
                    st.download_button(label     = "📥 Download Image", 
                                       data      = buffer.getvalue(), 
                                       file_name = "generated_image.png", 
                                       mime      = "image/png")
                
                else:
                    st.error("❌ Failed to generate image. Try again.")
            
            else:
                st.error(f"❌ Error {response.status_code}: Unable to generate image.")

# Footer
st.markdown("<br><hr><center>🚀 Powered by BrandSync AI 🚀</center>", unsafe_allow_html=True)
