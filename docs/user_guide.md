<div align="center">
  <h1>AI-Powered Social Media Content Generation System</h1>
  <p>Integrated Text & Image Generation for Your Social Media Needs</p>
</div>

---

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Text Generation Module](#text-generation-module)
4. [Image Generation Module](#image-generation-module)
5. [Advanced Features & Integration](#advanced-features--integration)
6. [Troubleshooting & FAQs](#troubleshooting--faqs)
7. [Appendices](#appendices)

---

## 1. Introduction

Welcome to the **AI-Powered Social Media Content Generation System**. This guide provides a unified overview of both our text and image generation functionalities—designed to help you create engaging, brand-consistent social media content effortlessly.

**Key Objectives:**
- **Automated Content Creation:** Generate context-aware posts and visuals.
- **Brand Consistency:** Maintain your unique style, voice, and tone.
- **Cross-Platform Adaptation:** Tailor content for LinkedIn, Instagram, Facebook, and more.
- **Streamlined Workflow:** Seamlessly combine text and image outputs for complete posts.

---

## 2. Getting Started

### Prerequisites
- **Python 3.10+** installed.
- All required dependencies (see `requirements.txt`).
- Basic command-line knowledge for running scripts and servers.

### Installation & Setup
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/satyaki-itobuz/social_media_content_generation.git
    cd social_media_content_generation
    ```

2. **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    .\venv\Scripts\activate   # For Windows
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Accessing the Platform
- **Web Interface:**  
  Navigate to the `web/` directory and start the API server:
    ```bash
    cd web/
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
    ```
- **Dashboard Navigation:**  
  Use the left sidebar on your dashboard to access the text and image generation features.

---

## 3. Text Generation Module

### Purpose & Use Cases
The text generation module transforms your inputs into engaging, platform-specific content. Whether you need professional posts for LinkedIn or creative content for Instagram, this module adapts to your brand's voice.

### Step-by-Step Guide

1. **Access the Generator:**
   - Navigate to the **"Generate Text"** page on the dashboard.

2. **Define Your Content:**
   - **Occasion/Topic:** Specify the event or subject (e.g., product launch, webinar).
   - **Description:** Provide background details and key points.
   - **Keywords & Hashtags:** Add relevant terms and social media tags.
   - **Brand Voice Elements:** Mention signature phrases or tone guidelines.

3. **Refine Your Communication Style:**
   - **Tone Selection:** Choose from professional, casual, enthusiastic, or persuasive options.
   - **Target Audience:** Specify demographics and audience interests.
   - **Content Length:** Decide between concise or detailed posts.
   - **Platform Optimization:** Select the social media platform to tailor the style.

4. **Generate and Edit:**
   - Review your inputs.
   - Click the **'Generate'** button.
   - Edit and refine the generated text as needed using built-in tools.

### Best Practices
- **Detailed Inputs:** The more context you provide, the better the AI output.
- **Hashtag Strategy:** Combine common and unique hashtags for optimal engagement.
- **Iterate:** Generate multiple versions and refine for best results.

---

## 4. Image Generation Module

### Purpose & Use Cases
This module converts your descriptive prompts into visually compelling images that align with your brand aesthetics. It is ideal for creating graphics that complement your social media posts.

### Step-by-Step Guide

1. **Access the Generator:**
   - Open the **"Generate Image"** option from the dashboard’s left sidebar.

2. **Craft Your Prompt:**
   - Enter a detailed description including key elements like style, mood, and color scheme.
   - **Example:** "A serene mountain landscape at sunset with snow-capped peaks, warm golden light, photorealistic style."

3. **Select Negative Prompts:**
   - Use the dropdown menu to select negative prompts that exclude unwanted elements.

4. **Customize Text Position (if applicable):**
   - Choose the text overlay position (e.g., heading, subheading, paragraph) on the image.

5. **Adjust Image Settings:**
   - **Image Size:** Select the desired dimensions.
   - **Generation Steps:** Specify the number of steps for image refinement.
   - **Text Influence:** Set a value (1–5) to balance adherence to your text prompt versus creative freedom.

6. **Generate Your Image:**
   - Click the **'Generate Image'** button.
   - Preview, refine, and download the final image.

### Best Practices
- **Be Specific:** Detailed descriptions yield higher-quality visuals.
- **Experiment:** Adjust generation parameters to optimize output.
- **Maintain Branding:** Include brand colors and stylistic elements consistently.

---

## 5. Advanced Features & Integration

### Customization & Fine-Tuning
- **Fine-Tuning Models:** Customize both text and image generation models (e.g., using LoRA for model fine-tuning) to capture your brand’s unique style.
- **Parameter Adjustments:** Tweak settings such as generation steps and text influence for precise outputs.

### Integration Workflow
1. **Generate Text:** Create your post content using the text generation module.
2. **Generate Image:** Produce a complementary visual using the image generation module.
3. **Combine Outputs:** Merge the text and image to form a cohesive, brand-consistent social media post.

### API & Automation
- Use the provided RESTful API and frontend interface for batch processing and real-time previews.
- Refer to the API documentation in the `/docs` directory for integration details.

---

## 6. Troubleshooting & FAQs

### Common Issues and Solutions

- **Text Generation:**
  - *Issue:* Output is too generic.
    - **Solution:** Enhance inputs with more detailed context and brand specifics.
  - *Issue:* Tone does not match the desired style.
    - **Solution:** Adjust tone settings and include clear examples.

- **Image Generation:**
  - *Issue:* Images lack desired detail.
    - **Solution:** Enrich your prompt with additional descriptors and adjust generation steps.
  - *Issue:* Unwanted visual elements appear.
    - **Solution:** Select appropriate negative prompts to filter out undesired content.

### FAQs

- **How do I merge text and image outputs?**  
  Use the integration workflow above to align text and visual elements into a complete social media post.

- **Can I customize generation parameters?**  
  Yes. Both modules offer adjustable settings such as tone, content length, image size, generation steps, and text influence.

- **Where do I find more technical details?**  
  Refer to the README and the documentation in the `/docs` directory for extended technical and API references.

### Support
For further assistance, please consult the provided documentation or contact our support team as detailed in the project README.

---

## 7. Appendices

### Glossary
- **Negative Prompt:** Instructions to omit specific elements from the generated image.
- **Text Influence:** A parameter controlling how closely the image adheres to the text prompt.
- **Generation Steps:** The number of iterations the model uses to refine its output.

### Additional Resources
- **[Project README](../README.md):** Overall project overview and technical specifications.
- **[Text Generation User Guide](../web/pages/text_generation_user_guide.md)**
- **[Image Generation User Guide](../web/pages/image_generation_user_guide.md)**
- **API Documentation:** Detailed reference available in the `/docs` directory.

---

<div align="center">
  <p>Made with ❤️ by the AI-ML-2024 Intern Team</p>
  <a href="#top">↑ Back to Top</a>
</div>
