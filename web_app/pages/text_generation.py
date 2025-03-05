# Dependencies
import requests
import streamlit as st

# ---- Custom CSS for Center Alignment ----
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

# Placeholder API (Replace with actual API)
text_generation_api = "https://api.example.com/generate_text"

# Title & Description
st.markdown("<h1 class='title'>✍️ AI-Powered Text Generation</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>📢 Create Social Media Posts Effortlessly!</h3>", unsafe_allow_html=True)

# Centered Container
with st.container():
    st.markdown("<div class='container'>", unsafe_allow_html=True)

    # User Inputs
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

    # Button to Generate Text
    if st.button("Generate Text"):
        if not brief:
            st.warning("⚠️ Please provide a brief description for the post.")
        else:
            # API Request
            payload  = {"purpose"       : purpose,
                        "occasion"      : occasion if purpose == "On A Specific Occasion" else None,
                        "topic"         : topic_brief if purpose == "On A Specific Topic" else None,
                        "brief"         : brief,
                        "extra_details" : extra_details,
                        "platform"      : platform,
                        "tone"          : tone,
                        "audience"      : target_audience
                       }
            
            # Send the request to the API endpoint
            response = requests.post(url  = text_generation_api, 
                                     json = payload)
 
            # Capture the API response
            if (response.status_code == 200):
                generated_text = response.json().get("generated_text", "")

                if generated_text:
                    # Display generated text
                    st.markdown("<h3 class='subtitle'>📝 Generated Content</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='output-box'>{generated_text}</div>", unsafe_allow_html=True)

                    # Copy & Download Option
                    st.code(generated_text, language = "markdown")
                    
                    st.download_button(label     = "📥 Download as Text File",
                                       data      = generated_text,
                                       file_name = "generated_text.txt",
                                       mime      = "text/plain"
                                      )
                
                else:
                    st.error("❌ Failed to generate text. Try again.")
            
            else:
                st.error(f"❌ Error {response.status_code}: Unable to generate text.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><hr><center>🚀 Powered by BrandSync AI 🚀</center>", unsafe_allow_html=True)
