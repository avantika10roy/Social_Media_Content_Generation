# AI-Powered Social Media Content Generator Web App

## Overview
The **AI-Powered Social Media Content Generator** is a **Streamlit-based** web application that allows users to generate AI-driven **text** and **images** for social media platforms like **LinkedIn, Instagram, and Facebook**. It leverages **LLM-based text generation** and **AI-driven image synthesis** to help users create **brand-consistent, engaging, and platform-optimized** content.

---

## ✨ Features
✅ AI-powered **text generation** for professional, creative, and brand-aligned posts  
✅ AI-driven **image generation** with fonts, colors, and layout customization  
✅ **Multi-platform** content adaptation for LinkedIn, Instagram, and Facebook  
✅ **User-friendly** interface built with Streamlit  
✅ **Pre-configured themes and styles** for AI-generated content  
✅ **Downloadable** text and image content  

---

## 🚀 Installation

### Prerequisites
Ensure you have **Python 3.8+** installed.

### Clone the Repository
```bash
git clone https://github.com/your-repo/ai-social-media-content-generator.git
cd ai-social-media-content-generator/web
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```

## 📂 Project Structure
```plaintext
web/
├── app.py                           # Main entry point for the web application
├── assets/
│   └── logo.png                      # Application logo
├── fonts/                             # Font files for image customization
├── pages/
│   ├── __init__.py
│   ├── home.py                        # Home page
│   ├── text_generation.py             # AI-powered text generation page
│   ├── image_generation.py            # AI-powered image generation page
│   ├── generate_both.py               # Page for generating both text and images
│   ├── text_generation_doc.py         # Documentation for text generation
│   ├── text_generation_user_guide.py  # User guide for text generation
│   ├── image_generation_doc.py        # Documentation for image generation
│   ├── image_generation_user_guide.py # User guide for image generation
├── pydantic_inputs.py                 # Input validation for API requests
├── pydantic_outputs.py                # Output validation for API responses
├── social_media_content_generator.py  # FastAPI backend for text generation
├── text_generation_inference.py       # Text generation model and inference logic
```

## 📌 Web Application Pages

### 🏠 Home (`home.py`)
- Provides an **overview** of the app  
- **Explains** AI-powered content generation  
- Contains **social media links** for user engagement  

### ✍️ Text Generation (`text_generation.py`)
- **Generates AI-powered social media posts**  
- Users select **tone, audience, and platform**  
- Posts are **brand-aligned** and **customizable**  
- **Download and copy** functionality included  

### 🖼️ Image Generation (`image_generation.py`)
- **Creates AI-generated images** for posts  
- Allows **customization of text, fonts, and colors**  
- Supports **various image styles**  
- **Downloadable high-resolution** images  

### 🚀 Generate Both (`generate_both.py`)
- **Combines text and image generation** in one workflow  
- Users can create **a complete social media post**  
- **Fully customizable** text and image options  

### 📚 Documentation & User Guides
- **`text_generation_doc.py` & `text_generation_user_guide.py`**: Guides for text generation  
- **`image_generation_doc.py` & `image_generation_user_guide.py`**: Guides for image generation  

## ⚙️ Backend & API Services

### 📜 Text Generation API (`social_media_content_generator.py`)
- **FastAPI-based backend**  
- Exposes an **API endpoint** for generating AI-driven text  

#### API Endpoint:
```http
POST /generate_text
```

#### Request Example:
```json
{
    "company_name": "Itobuz Technologies",
    "occasion": "Product Launch",
    "topic": "",
    "brief": "Introducing our AI-powered automation tool!",
    "extra_details": "Include the benefits of AI-driven automation.",
    "platform": "LinkedIn",
    "tone": "Professional",
    "target_audience": "Tech Entrepreneurs"
}
```

#### Response Example:
```json
{
    "generated_text": "🚀 Big news! Our AI-powered automation tool is here to change the game!..."
}
```

### 🔍 AI Model & Inference (`text_generation_inference.py`)
- **Loads pre-trained LLM models**  
- **Generates structured and engaging social media posts**  
- **Uses prompt engineering** to optimize AI-generated text  
- **Filters & cleans hashtags** before returning responses  

---

### 🛠️ Logging & Debugging
- All logs are stored in the **logs/** directory.  
- If any page crashes, refer to **Streamlit logs** for debugging.  
