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
│   ├── __init__.py
│   └── config.py
├── credentials
|   └── client_secret.json
├── data                                   # All type of data in each stage has been saved here
│   ├── cleaned_data
│   │   ├── facebook_cleaned_data.json     # Cleaned facebook data 
│   │   ├── instagram_cleaned_data.json    # Cleaned instagram data
│   │   └── linkedin_cleaned_data.json     # Cleaned linkedin data
│   ├── curated_data                       
│   │   └── final_data.json                # After merging data from all platforms, saved here in one unified place
│   ├── extracted_features_data             
│   │   ├── blip_output.json               # Extracted features using BLIP along with original features saved here
|   |   └── clip_output.json               # Extracted features using CLIP along with original features saved here
|   ├── logo_identification_result
|   |   └── output_with_logo_info_and_uploads.json       # Adds logo position in image
|   ├── mixed_curated
|   |   └── mixed_curated.json             # Adds raw post content
|   ├── preprocessed_data
|   |   ├── preprocessed_data.json         # Combines all preprocessed features in json format
|   |   └── preprocessed_data2.json
│   ├── raw_data
│   |   ├── facebook_raw_data.json         # Raw Scraped facebook data
│   |   ├── instagram_raw_data.json        # Raw scraped instagram data
│   |   └── linkedin_raw_data.json         # Raw scraped linkedin data
|   └── logo.jpg
├── data_processor.py                       
├── docs                                   # A centralized folder for keeping all project related documents for future purpose
│   ├── project_flowchart.png
│   └── workflow.png
├── logs                                   # Log files saved here for all the tasks
├── llm_evaluation.py                      # Evaluate the performance of LLM
├── llm_modular.py                         # Modular LLM finetuning code
├── llm_run.py                             # LLM Inference
├── notebooks                              # Containing all jupyter notebooks for experimentation
|   ├── FLAN-T5.ipynb
│   └── LLM_Experiments.ipynb
├── requirements.txt                       # Required python dependencies
├── results
|   ├── evaluation_results                 # Results of evaluation
|   |   └── falcon3_1b_instruct_eval.json
|   └── llm_results
|   |   ├── fine_tuning_results_v1/checkpoint-115
|   |   ├── flan_t5_base_fine_tuning_results_v1
|   |   ├── pipeline_finetuning_v9
|   |   └── __init__.py
├── run.py                                 # 
├── scrape_raw_data.py                     # Run file for data collection by scraper module
├── setup.sh                               # Project environment setup 
├── src                                    # All source codes 
|   ├── custom_dataset                     # Custom dataset for LLM
|   │   └── llm_dataset.py
|   ├── data_cleaner                       # Centralized module for data cleaning for whole project
|   │   ├── README.md                      # Clean and preprocess scraped social media content
|   │   ├── __init__.py 
|   │   ├── image_cleaner.py
|   │   └── text_cleaner.py
|   ├── data_curator                       # Centralized module for data curation for whole project
|   │   ├── README.md                      # Merge, refine, and structure cleaned social media data
|   │   ├── __init__.py
|   │   ├── data_curation.py
|   │   └── mix_curator.py
|   ├── data_preprocessor                  # Centralized module for data preprocessing for whole project
|   │   ├── README.md                      # Transform, augment, and standardize both text and image data
|   │   ├── __init__.py
|   │   ├── image_augmentor.py
|   │   ├── llm_finetune_data_preprocessor.py
|   │   └── text_preprocessing.py
|   ├── feature_engineering                # Centralized module for feature engineering for whole project
|   │   ├── blip_feature_extraction.py
|   │   └── clip_feature_extraction.py
|   ├── frontend                           # Centralized module for frontend management
|   │   └── __init__.py
|   ├── identify_logo                           
|   │   └── logo_identification.py
|   ├── model_finetuners                   # Model fine-tuning functionalities
|   │   ├── __init__.py
|   │   ├── flan_t5_finetuner.py
|   │   ├── llm_fine_tuner.py
|   │   └── t5_lora_finetuning.py
|   ├── model_inference                    # Model inference functionalities
|   │   ├── __init__.py
|   │   ├── flan_t5_inference.py
|   │   ├── llm_inference.py
|   │   └── t5_inference.py
|   ├── prompts                            # Storing prompts that generate good results
|   │   └── __init__.py
|   │   ├── prompts.py
|   ├── scraper                            # Centralized scraper module 
|   │   ├── __init__.py
|   │   ├── facebook_scraper.py
|   │   ├── instagram_scraper.py
|   │   └── linkedin_scraper.py
|   ├── scripts                            # 
|   │   └── example.sh 
|   └── utils                              # Unified utility module for any other utilities than model related tasks
|       ├── __init__.py
|       ├── color_themes.py
|       ├── data_saver.py
|       ├── download_from_drive.py
|       ├── set_seed.py
|       └── logger.py 
├── T5_run.py 
└── web_app
    ├── assets
    |   └── logo.png
    ├── pages
    ├── __init__.py
    ├── app.py
    ├── image_generation_inference.py
    ├── pydantic_inputs.py
    ├── pydantic_outputs.py
    ├── social_media_content_generator.py
    └── text_generation_inference.py
```

## 💻 Usage

### Model Fine-tuning
```bash 
Yet to be done
```

### API Usage

```bash
# Go to the web folder
cd web/

# Start the API server
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Performance Metrics

| Metric | Target | Current |
|--------|---------|---------|
| Response Time | <30s | 25s |
| Platform Support | 3+ | 4 |

## 📚 Documentation

Comprehensive documentation is available in the `/docs` directory:

- 📖 [System Architecture](docs/architecture.md)
- 🔧 [API Reference](docs/api.md)
- 👥 [User Guide](docs/user_guide.md)
- 🎓 [Fine-tuning Guide](docs/fine-tuning.md)

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
  - Meta
  - Salesforce

---

<div align="center">

**Made with ❤️ by the AI-ML-2024 Intern Team**

[↑ Back to Top](#)

</div>