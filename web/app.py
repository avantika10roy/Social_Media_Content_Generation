import streamlit as st

pg = st.navigation([
    st.Page("pages/home.py", 
            title = "Home", 
            icon  = "🏠"),
    
    st.Page("pages/text_generation.py", 
            title = "Generate Text", 
            icon  = "✍️"),
    
    st.Page("pages/image_generation.py", 
            title = "Generate Image", 
            icon  = "🖼️"),
    
    st.Page("pages/generate_both.py", 
            title = "Generate Both", 
            icon  = "🚀"),
    
    st.Page("pages/text_generation_doc.py", 
            title = "Text Generation Documentation", 
            icon  = "📚"),
    
    st.Page("pages/text_generation_user_guide.py", 
            title = "Text Generation User Manual", 
            icon  = "🔍"),
    
    st.Page("pages/image_generation_doc.py",
            title = "Image Generation Documentation", 
            icon  = "📚"),
    
    st.Page("pages/image_generation_user_guide.py", 
            title = "Image Generation User Manual", 
            icon  = "🔍")
])
pg.run()

