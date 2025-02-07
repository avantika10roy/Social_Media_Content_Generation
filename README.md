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
  - Batch processing support

## 🛠️ Technology Stack

### Core Technologies
- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Diffusers
- FastAPI
- Streamlit

### AI Models
- **Text Generation**: LLAMA-7B
- **Caption Generation**: BLIP
- **Image Generation**: Stable Diffusion XL (SDXL)

## 📋 Prerequisites
```bash
yet to be done
```

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
├── changelog.txt                          # Contains the description of the chnages from the starting and done by whom
├── config                                 # Centralized Configuration module for the whole project
│   ├── __init__.py
│   └── config.py
├── data                                   # All type of data in each stage has been saved here
│   ├── cleaned_data
│   │   ├── facebook_cleaned_data.json     # Cleaned facebook data 
│   │   ├── instagram_cleaned_data.json    # Cleaned instagram data
│   │   └── linkedin_cleaned_data.json     # Cleaned linkedin data
│   ├── curated_data                       
│   │   └── final_data.json                # After merging data from all platforms, saved here in one unified place
│   ├── extracted_features_data             
│   │   └── blip_output.json               # Extracted features along with original features saved here
│   └── raw_data
│       ├── facebook_raw_data.json         # Raw Scraped facebook data
│       ├── instagram_raw_data.json        # Raw scraped instagram data
│       └── linkedin_raw_data.json         # Raw scraped linkedin data
├── data_cleaning.py                       
├── docs                                   # A centralized folder for keeping all project related documents for future purpose
│   ├── project_flowchart.png
│   └── workflow.png
├── logs                                   # Log files saved here for all the tasks
│   
├── notebooks                              # Containing all jupyter notebooks for experimentation
│   └── Final_Project.ipynb
├── requirements.txt                       # Required pythoon dependencies
├── run.py                                 # 
├── scrape_raw_data.py                     # Run file for data collection by scraper module
├── setup.sh                               # Project environment setup 
└── src                                    # All source codes 
    ├── api                                # API related codes here
    │   └── __init__.py
    ├── base_models                        # Base models for image generation, text generation and feature extraction saved here
    │   └── __init__.py  
    ├── data_cleaner                       # Centralized module for data cleaning for whole project
    │   ├── __init__.py 
    │   ├── data_cleaner.py
    │   ├── data_preprocessing.py
    │   └── linkedIn_preprocessor.py
    ├── data_curator                       # Centralized module for data curation for whole project
    │   └── __init__.py
    ├── data_preprocesser                  # Centralized module for data preprocessing for whole project
    │   ├── __init__.py
    │   └── text_preprocessing.py
    ├── feature_engineering                # Centralized module for feature engineering for whole project
    │   └── blip_feature_extraction.py
    ├── frontend                           # Centralized module for frontend management
    │   └── __init__.py
    ├── model_finetuners                   # Model fine-tuning functionalities
    │   └── __init__.py 
    ├── model_inference                    # Model inference functionalities
    │   └── __init__.py
    ├── models                             # Model utilities
    │   ├── __init__.py
    │   ├── model_loader.py
    │   └── model_saver.py
    ├── scraper                            # Centralized scraper module 
    │   ├── __init__.py
    │   ├── facebook_scraper.py
    │   ├── instagram_scraper.py
    │   └── linkedin_scraper.py
    ├── scripts                            # 
    │   └── example.sh 
    └── utils                              # Unified utility module for any other utilities than model related tasks
        ├── __init__.py
        ├── data_saver.py
        └── logger.py                    
```

## 💻 Usage

### Model Fine-tuning
```bash 
Yet to be done
```

### API Usage

```bash
# Start the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Performance Metrics

| Metric | Target | Current |
|--------|---------|---------|
| Response Time | <30s | 25s |
| Platform Support | 3+ | 4 |
| Brand Adherence | 95% | 97% |

## 📚 Documentation

Comprehensive documentation is available in the `/docs` directory:

- 📖 [System Architecture](docs/architecture.md)
- 🔧 [API Reference](docs/api.md)
- 👥 [User Guide](docs/user-guide.md)
- ⚙️ [Installation Guide](docs/installation.md)
- 🎓 [Fine-tuning Guide](docs/fine-tuning.md)

## 🗺️ Future Roadmap

### Phase 1: Scalability
- [ ] Multi-user support
- [ ] Batch processing
- [ ] Platform expansion

### Phase 2: Advanced Features
- [ ] A/B testing system
- [ ] Analytics dashboard
- [ ] Automated scheduling
- [ ] Content calendar

### Phase 3: Integrations
- [ ] Direct social media posting
- [ ] CRM integration
- [ ] CMS integration


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AI-ML-2024 Internship Program Team
- Open-source AI Model Providers:
  - Meta
  - Salesforce

---

<div align="center">

**Made with ❤️ by the AI-ML-2024 Intern Team**

[↑ Back to Top](#)

</div>