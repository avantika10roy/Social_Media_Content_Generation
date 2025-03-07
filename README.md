<div align="center">

# 🤖 AI-Powered Social Media Content Generation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-009688.svg)](https://fastapi.tiangolo.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Transform your social media presence with AI-powered content generation*

[Features](#features) • [Installation](#installation) • [Documentation](#documentation) • [Usage](#usage) • [Contributing](#contributing)

</div>

---

## 🌟 Overview

The AI-Powered Social Media Content Generation System is a cutting-edge solution that harnesses the power of Large Language Models (LLMs) and Stable Diffusion to revolutionize social media content creation. This system automatically generates platform-specific posts while maintaining brand consistency and engagement quality.

### 🎯 Key Objectives

- Automated, context-aware social media content generation
- Brand-consistent content creation
- Streamlined content workflow optimization
- Cross-platform content adaptation

## ✨ Features

### 🤖 Core Capabilities

- **Intelligent Content Generation**
  - Context-aware post creation
  - Platform-specific formatting
  - Brand voice preservation
  - Hashtag optimization

- **Smart Integration**
  - RESTful API
  - Intuitive frontend interface
  - Real-time preview

## 🛠️ Technology Stack

### Core Technologies
- Python 3.10+
- PyTorch
- Transformers
- Diffusers
- FastAPI
- Streamlit

### AI Models
- **Text Generation**: FALCON3-1B-Instruct
- **Caption Generation**: BLIP
- **Image Generation**: Kandinsky 2-2-Decoder-Inpaint

## 📋 Prerequisites

Before you begin using the AI-Powered Social Media Content Generation System, ensure that your environment is properly set up. You will need to install the following tools and libraries:

### System Requirements:
- **Python**: Python 3.10 or later
- **Operating System**: Linux, macOS, or Windows (All platforms supported)

### Software Requirements:
1. **Python** (3.10+): This project is built with Python 3.10 or newer. You can download Python from the official website:
   - [Python Download](https://www.python.org/downloads/)

2. **Chromedriver**: Required for Selenium-based web scraping. It can be installed using the following command (depending on your OS):
   - For **Windows**: Download from [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads)
   - For **Linux/macOS**: Install via a package manager or download the appropriate version from the above link.


## 🚀 Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/satyaki-itobuz/social_media_content_generation.git
cd social_media_content_generation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```


## 📁 Project Structure

```plaintext
Social_Media_Content_Generation/
├── LICENSE                                # MIT License for the project
├── README.md                              # Executive summary of the project and results
├── base_model_downoader.py                # Downloads SDXL Base Model
├── changelog.txt                          # Contains the description of the changes from the starting and done by whom
├── config                                 # Centralized Configuration module for the whole project
├── credentials
├── data                                   # All type of data in each stage has been saved here
│   ├── cleaned_data                       # All cleaned data in JSON format
│   ├── curated_data                       # After merging data from all platforms, saved here in one unified place             
│   ├── extracted_features_data            # Extracted features using BLIP and CLIP along with original features
|   ├── logo_identification_result         # Adds logo position in image 
|   ├── mixed_curated                      # Adds raw post content
|   ├── preprocessed_data                  # Combines all preprocessed features in json format
│   ├── raw_data                           # All raw Scraped data      
|   └── logo.jpg       
├── docs                                   # A centralized folder for keeping all project related documents for future purpose
├── notebooks                              # Containing all jupyter notebooks for experimentation
├── requirements.txt                       # Required python dependencies
├── results
|   ├── evaluation_results                 # Results of evaluation
|   |   └── falcon3_1b_instruct_eval.json
|   └── llm_results                        # Fine-tuned models here
├── scrape_raw_data.py                     # Run file for data collection by scraper module
├── setup.sh                               # Project environment setup 
├── src                                    # All source codes 
|   ├── custom_dataset                     # Custom dataset for LLM
|   ├── data_cleaner                       # Centralized module for data cleaning for whole project
|   ├── data_curator                       # Centralized module for data curation for whole project
|   ├── data_preprocessor                  # Centralized module for data preprocessing for whole project
|   ├── feature_engineering                # Centralized module for feature engineering for whole project
|   ├── identify_logo                      # Identifies the presence of logo in the images
|   ├── model_finetuners                   # Model fine-tuning functionalities
|   ├── model_inference                    # Model inference functionalities
|   ├── prompts                            # Storing prompts that generate good results
|   ├── scraper                            # Centralized scraper module 
|   └── utils                              # Unified utility module for any other utilities than model related tasks
└── web_app
    ├── assets
    └── pages
```

## 💻 API Usage

```bash
# Go to the web folder
cd web/

# Start the API server
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Performance Metrics

| Metric | Target | Current |
|--------|---------|---------|
| Response Time | <30s | 60s |
| Platform Support | 3 | 3 |

## 📚 Documentation

Comprehensive documentation is available in the `/docs` directory:

- 📖 [System Architecture](docs/AI_Powered_Social_Media_Content_Generation_System.pdf)
- 👥 [User Guide](docs/user_guide.md)

## 🗺️ Future Roadmap

### Phase 1: Scalability
- [ ] Multi-user support
- [ ] Platform expansion

### Phase 2: Advanced Features
- [ ] Analytics dashboard

### Phase 3: Integrations
- [ ] Direct social media posting


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AI-ML-2024 Internship Program Team
- Open-source AI Model Providers:
  - FALCON3 by Technology Innovation Institute(TII) (Abu Dhabi)
  - Kandinsky by Accubits

---

<div align="center">

**Made with ❤️ by the AI-ML-2024 Intern Team**

[↑ Back to Top](#)

</div>